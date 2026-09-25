"""Resolve a free-form edit request into an operation, a target, and an intent.

The user is free to phrase a request however they like. This module never rejects
a prompt for "wrong phrasing": every request resolves to some localized
operation, and the guarantee that the rest of the frame stays untouched is enforced
downstream by the mask + composite + hard-restore, not by constraining the words.

The design follows the instruction-editing literature (ZONE, DM-Align): extract the
subject the edit is about, ground it open-vocabulary, and confine the change to
that region. Concretely:

    recolour -> tracked matte + hue transform (a colour was requested)
    remove -> temporal propagation + plate (a deletion was requested)
    replace -> masked diffusion, subject -> thing (a swap was requested)
    edit -> masked diffusion, instruction-guided (anything else)

edit is the catch-all so an unusual instruction still produces a localized,
surroundings-preserving change instead of an error. The subject is a free-text
phrase (e.g. "the girl on the left", "her dress") handed to CLIPSeg, so the
vocabulary is open; the COCO detector is only a faster, tighter path when the
subject happens to be a class it knows.
"""

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from preserve.prompt import COLOR_HUE_RANGES, TargetSpec, parse_prompt


class EditOperation(StrEnum):
    RECOLOR = "recolor"
    REMOVE = "remove"
    REPLACE = "replace"
    EDIT = "edit"


REMOVE_VERBS = (
    "remove",
    "delete",
    "erase",
    "eliminate",
    "take out",
    "get rid of",
    "clear out",
    "cut out",
    "hide",
)

RECOLOR_VERBS = (
    "recolor",
    "recolour",
    "colorize",
    "colourize",
    "repaint",
    "paint",
    "tint",
    "dye",
)

REPLACE_VERBS = (
    "replace",
    "swap",
    "substitute",
    "turn into",
    "convert into",
    "morph into",
)

# Verbs that introduce an edit but whose result decides the operation.
_CHANGE_VERBS = ("change", "make", "turn", "set", "convert", "transform")

ALL_COLORS = set(COLOR_HUE_RANGES) | {"black", "white", "grey", "gray", "silver"}

# Words that carry no subject in a result clause, so "a bright red one" is still
# just a colour request while "a yellow taxi" names a new subject.
_FILLER_WORDS = {
    "a",
    "an",
    "the",
    "it",
    "one",
    "instead",
    "colour",
    "color",
    "coloured",
    "colored",
    "shade",
    "please",
    "now",
    "version",
    "and",
    "of",
    "bright",
    "dark",
    "light",
    "pale",
    "deep",
    "into",
    "to",
    "with",
}

# Leading words to strip when isolating the subject phrase for grounding.
_ACTION_PREFIXES = tuple(
    sorted(
        {
            *REMOVE_VERBS,
            *RECOLOR_VERBS,
            *REPLACE_VERBS,
            *_CHANGE_VERBS,
            "only",
            "just",
            "keep only",
        },
        key=len,
        reverse=True,
    )
)


@dataclass
class EditSpec:
    """A fully resolved edit instruction."""

    operation: EditOperation
    target: TargetSpec
    # Free-text subject phrase for open-vocabulary grounding (always populated).
    target_phrase: str = ""
    # For RECOLOR: the colour to paint the target.
    new_color: str | None = None
    # For REPLACE: what the target should become.
    replacement: str | None = None
    # For EDIT: the instruction handed to the diffusion prompt as-is.
    instruction: str | None = None
    raw_prompt: str = ""

    def describe(self) -> str:
        subject = self.target_phrase or self.target.describe()
        if self.operation is EditOperation.RECOLOR:
            return f"recolor {subject} to {self.new_color}"
        if self.operation is EditOperation.REPLACE:
            return f"replace {subject} with {self.replacement}"
        if self.operation is EditOperation.EDIT:
            return f"edit {subject}: {self.instruction}"
        return f"remove {subject}"

    @property
    def result_color(self) -> str | None:
        """The colour the result should have, when one was named.

        For a recolour that is the requested colour; for a replacement it is a
        colour named inside the replacement phrase ("a yellow taxi"), so the
        scorer can check the edit actually landed in that colour.
        """
        if self.operation is EditOperation.RECOLOR:
            return self.new_color
        if self.replacement:
            words = set(re.findall(r"[a-z]+", self.replacement.lower()))
            matches = sorted(words & ALL_COLORS, key=lambda w: self.replacement.lower().find(w))
            return matches[0] if matches else None
        return None


_PIVOT = re.compile(r"\b(?:with|into|to)\b")
_TRAILING_COLOR = re.compile(
    rf"\b({'|'.join(sorted(ALL_COLORS, key=len, reverse=True))})\b"
    rf"(?:\s+(?:{'|'.join(sorted(_FILLER_WORDS, key=len, reverse=True))}))*"
    r"\s*[.!]?\s*$"
)
_PROTECT_CLAUSE = re.compile(
    r"\b(?:and\s+|but\s+)?(?:keep|preserve|retain|leave|protect|maintain)\b.*$"
)


def _split_clauses(prompt: str) -> tuple[str, str]:
    """Return (selection_text, result_text).

    The result clause is what the edit should produce; the selection clause names
    what it applies to. Splitting them keeps a requested outcome ("...a red truck")
    from being mistaken for another filter on which object to select.
    """
    lowered = prompt.lower().strip()

    protect_match = _PROTECT_CLAUSE.search(lowered)
    protect_text = protect_match.group(0) if protect_match else ""
    head = lowered[: protect_match.start()] if protect_match else lowered

    pivots = list(_PIVOT.finditer(head))
    if pivots:
        pivot = pivots[-1]
        return f"{head[: pivot.start()]} {protect_text}".strip(), head[pivot.end() :].strip()

    trailing = _TRAILING_COLOR.search(head)
    if trailing:
        return f"{head[: trailing.start()]} {protect_text}".strip(), trailing.group(1)

    return f"{head} {protect_text}".strip(), ""


def _subject_phrase(selection: str) -> str:
    """Reduce a selection clause to the subject phrase a grounder should locate.

    Strips a leading action verb and any protect clause but keeps determiners and
    spatial words ("the girl on the left"), which help a text-conditioned segmenter.
    """
    text = _PROTECT_CLAUSE.sub("", selection).strip()

    changed = True
    while changed:
        changed = False
        for prefix in _ACTION_PREFIXES:
            if text.startswith(prefix + " "):
                text = text[len(prefix) :].strip()
                changed = True
        # Drop a dangling leading determiner left by verb removal.
        for lead in ("the ", "a ", "an ", "this ", "that ", "of "):
            if text.startswith(lead):
                text = text[len(lead) :].strip()
                changed = True

    # A trailing negation clause ("... not the whole crowd") is context, not the
    # subject, so cut it — the subject is what precedes it.
    text = re.split(r"\bnot\b", text, maxsplit=1)[0].strip()
    return text


def _target_from_selection(prompt: str, selection: str) -> TargetSpec:
    """Parse the selection clause into the COCO/colour spec, if any applies."""
    full = parse_prompt(prompt)
    selected = parse_prompt(selection) if selection.strip() else full

    targets = selected.targets - full.protect
    if not targets:
        targets = full.targets - full.protect

    return replace(
        selected,
        targets=targets,
        colors=selected.colors,
        protect=full.protect,
        raw_prompt=prompt,
    )


def _without_result_color(target: TargetSpec, new_color: str | None) -> TargetSpec:
    if new_color is None or new_color not in target.colors:
        return target
    return replace(target, colors=target.colors - {new_color})


def classify_edit(prompt: str) -> EditSpec:
    """Resolve a free-form edit request. Never raises: unknown -> localized edit."""
    lowered = prompt.lower().strip()
    selection, result = _split_clauses(prompt)
    target = _target_from_selection(prompt, selection)
    subject = _subject_phrase(selection) or target.describe() or lowered

    if any(verb in lowered for verb in REMOVE_VERBS):
        return EditSpec(EditOperation.REMOVE, target, target_phrase=subject, raw_prompt=prompt)

    result_words = re.findall(r"[a-z]+", result)
    result_colors = [w for w in result_words if w in ALL_COLORS]
    substance = [w for w in result_words if w not in ALL_COLORS and w not in _FILLER_WORDS]

    # A result naming a thing is a replacement even if it also has a colour
    # ("...a red truck"); a result that is only a colour is a recolour.
    if substance:
        return EditSpec(
            EditOperation.REPLACE,
            target,
            target_phrase=subject,
            replacement=result,
            raw_prompt=prompt,
        )

    if result_colors:
        return EditSpec(
            EditOperation.RECOLOR,
            _without_result_color(target, result_colors[-1]),
            target_phrase=subject,
            new_color=result_colors[-1],
            raw_prompt=prompt,
        )

    if result and any(verb in lowered for verb in REPLACE_VERBS):
        return EditSpec(
            EditOperation.REPLACE,
            target,
            target_phrase=subject,
            replacement=result,
            raw_prompt=prompt,
        )

    # Nothing recolour/remove/replace-shaped: treat the whole request as an
    # instruction-guided localized edit on the subject. Surroundings are still
    # preserved by the composite, so this is safe for any phrasing.
    return EditSpec(
        EditOperation.EDIT,
        target,
        target_phrase=subject,
        instruction=prompt.strip(),
        raw_prompt=prompt,
    )


# What an attached object hides on its wearer. Reveal fills and the removal
# residue are phrased with this instead of "own surface", which reads as skin
# to an inpainting model (audit 2026-09-20: pale scalp).
REVEAL_SURFACES: dict[str, str] = {
    "cap": "short natural hair, hairline and forehead",
    "hat": "short natural hair, hairline and forehead",
    "beanie": "short natural hair, hairline and forehead",
    "helmet": "short natural hair, hairline and forehead",
    "hood": "short natural hair and the back of the head",
    "glasses": "eyes, eyebrows and the bridge of the nose",
    "sunglasses": "eyes, eyebrows and the bridge of the nose",
    "popsicle": "closed mouth and lips",
    "lollipop": "closed mouth and lips",
    "straw": "closed mouth and lips",
    "cigarette": "closed mouth and lips",
    "mask": "nose, mouth and chin",
    "scarf": "neck and collar",
    "logo": "plain fabric of the garment with its folds and shading",
    "print": "plain fabric of the garment with its folds and shading",
    "text": "plain fabric of the garment with its folds and shading",
    "badge": "plain fabric of the garment with its folds and shading",
    "patch": "plain fabric of the garment with its folds and shading",
    "sticker": "plain surface underneath",
    "necklace": "bare neck and collarbones",
    "watch": "bare wrist",
    "bracelet": "bare wrist",
    "earring": "bare earlobe",
    "earrings": "bare earlobes",
    "backpack": "the person's back and shoulders in their clothes",
}


# Targets that sit on a flat surface (a print on fabric, a sticker): the
# keyframe editor erases them classically first, or it keeps the print's
# outline as an embossed ghost (audit 2026-09-21: logo ring on a plain tee).
FLAT_SURFACE_NOUNS = {
    "logo",
    "print",
    "text",
    "badge",
    "patch",
    "sticker",
    "graffiti",
    "sign",
    "label",
}


def is_flat_surface_target(target_phrase: str | None) -> bool:
    noun = (target_phrase or "").split()[-1].lower() if target_phrase else ""
    return noun.rstrip("s") in FLAT_SURFACE_NOUNS or noun in FLAT_SURFACE_NOUNS


def is_wearable(target_phrase: str | None) -> bool:
    """Whether the target is something worn or held against a person (has a reveal surface)."""
    return reveal_surface(target_phrase, "person") != "the person's own surface"


def reveal_surface(target_phrase: str | None, parent: str) -> str:
    noun = (target_phrase or "object").split()[-1].lower()
    singular = (
        noun[:-2] if noun.endswith("es") and noun[:-2] in REVEAL_SURFACES else noun.rstrip("s")
    )
    return REVEAL_SURFACES.get(noun) or REVEAL_SURFACES.get(singular, f"the {parent}'s own surface")


# Nouns whose plural form names one object; "his glasses" is one pair.
_PLURALIA_TANTUM = {
    "glasses",
    "sunglasses",
    "pants",
    "shorts",
    "jeans",
    "scissors",
    "headphones",
    "earphones",
    "trousers",
    "tights",
    "binoculars",
}


def wants_multiple(prompt: str) -> bool:
    """Whether the prompt names more than one instance ("their caps", "both hats", "all the cars")."""
    words = re.findall(r"[a-z]+", prompt.lower())
    if any(
        w
        in {
            "their",
            "all",
            "both",
            "every",
            "each",
            "these",
            "those",
            "two",
            "three",
            "four",
            "several",
            "multiple",
        }
        for w in words
    ):
        return True
    if any(
        w in {"his", "her", "its", "my", "your", "our", "this", "that", "a", "an", "one"}
        for w in words
    ):
        return False
    nouns = [
        w
        for w in words
        if len(w) > 2
        and w not in {"remove", "delete", "erase", "the", "from", "with", "replace", "make"}
    ]
    return any(
        w.endswith("s") and w not in _PLURALIA_TANTUM and not w.endswith("ss") for w in nouns
    )
