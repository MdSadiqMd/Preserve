"""Parse natural-language edit prompts into a structured target specification.

A prompt like "remove blue cars but keep the trucks intact" resolves to:
    targets={"car"}, colors={"blue"}, protect={"truck"}
The pipeline then only builds a mask over car instances whose pixels are
predominantly blue, leaving every other detected object untouched.
"""

import re
from dataclasses import dataclass, field

# Maps user vocabulary onto COCO class names. Plural and colloquial forms are
# listed explicitly because we match on word boundaries, not stems.
CLASS_SYNONYMS: dict[str, str] = {
    "car": "car",
    "cars": "car",
    "sedan": "car",
    "sedans": "car",
    "vehicle": "car",
    "vehicles": "car",
    "automobile": "car",
    "automobiles": "car",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "van": "truck",
    "vans": "truck",
    "bus": "bus",
    "buses": "bus",
    "busses": "bus",
    "coach": "bus",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "man": "person",
    "woman": "person",
    "human": "person",
    "humans": "person",
    "train": "train",
    "trains": "train",
    "dog": "dog",
    "dogs": "dog",
    "cat": "cat",
    "cats": "cat",
    "bird": "bird",
    "birds": "bird",
    "bench": "bench",
    "benches": "bench",
    "traffic light": "traffic light",
    "traffic lights": "traffic light",
    "stop sign": "stop sign",
    "stop signs": "stop sign",
    "fire hydrant": "fire hydrant",
    "boat": "boat",
    "boats": "boat",
    "airplane": "airplane",
    "airplanes": "airplane",
    "plane": "airplane",
    "planes": "airplane",
}

# Hue ranges on OpenCV's 0-179 scale. Red wraps around the origin, so it gets
# two intervals. Achromatic colours are matched on saturation/value instead and
# carry no hue interval.
COLOR_HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 22)],
    "yellow": [(23, 33)],
    "green": [(34, 85)],
    "cyan": [(86, 95)],
    "blue": [(96, 130)],
    "purple": [(131, 155)],
    "violet": [(131, 155)],
    "pink": [(156, 169)],
    "magenta": [(156, 169)],
}

ACHROMATIC_COLORS = {"white", "black", "gray", "grey", "silver"}

REMOVAL_VERBS = {"remove", "delete", "erase", "clear", "eliminate", "take out", "get rid of"}
PROTECT_VERBS = {"keep", "preserve", "retain", "leave", "protect", "maintain"}


@dataclass
class TargetSpec:
    """Structured description of what an edit prompt asks to change."""

    targets: set[str] = field(default_factory=set)
    colors: set[str] = field(default_factory=set)
    protect: set[str] = field(default_factory=set)
    raw_prompt: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.targets

    def describe(self) -> str:
        colour = " ".join(sorted(self.colors))
        target = ", ".join(sorted(self.targets)) or "nothing"
        text = f"{colour} {target}".strip()
        if self.protect:
            text += f" (protecting {', '.join(sorted(self.protect))})"
        return text


def _find_classes(text: str) -> list[tuple[int, str]]:
    """Return (position, coco_class) for every class mention.

    Longer phrases are matched first and claim their character span, so
    "traffic light" is not also reported as a separate shorter match.
    """
    hits: list[tuple[int, str]] = []
    claimed: list[tuple[int, int]] = []

    for phrase in sorted(CLASS_SYNONYMS, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            if any(match.start() < end and start < match.end() for start, end in claimed):
                continue
            claimed.append((match.start(), match.end()))
            hits.append((match.start(), CLASS_SYNONYMS[phrase]))

    return sorted(hits)


def _find_colors(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    known = set(COLOR_HUE_RANGES) | ACHROMATIC_COLORS
    for colour in known:
        for match in re.finditer(rf"\b{re.escape(colour)}\b", text):
            hits.append((match.start(), colour))
    return sorted(hits)


def _split_clauses(text: str) -> list[str]:
    """Split on conjunctions that typically separate an action from an exception."""
    return [c for c in re.split(r"\b(?:but|while|and|,|;|however|except)\b", text) if c.strip()]


def parse_prompt(prompt: str) -> TargetSpec:
    """Turn a free-form edit instruction into a TargetSpec.

    Clauses containing a protect verb ("keep the trucks intact") contribute to
    protect; every other clause contributes to targets. When no clause
    carries a removal verb we treat all mentioned classes as targets, so a bare
    "the blue cars" still does the obvious thing.
    """
    text = prompt.lower().strip()
    spec = TargetSpec(raw_prompt=prompt)

    clauses = _split_clauses(text)
    saw_protect_clause = False

    for clause in clauses:
        classes = [c for _, c in _find_classes(clause)]
        if not classes:
            continue

        is_protect = any(verb in clause for verb in PROTECT_VERBS)
        if is_protect:
            saw_protect_clause = True
            spec.protect.update(classes)
        else:
            spec.targets.update(classes)
            spec.colors.update(c for _, c in _find_colors(clause))

    # A class named in a protect clause must never also be a target.
    spec.targets -= spec.protect

    # Fallback: no clause split produced targets (e.g. "keep trucks, blue cars out")
    if not spec.targets and not saw_protect_clause:
        spec.targets.update(c for _, c in _find_classes(text))
        spec.colors.update(c for _, c in _find_colors(text))
        spec.targets -= spec.protect

    return spec
