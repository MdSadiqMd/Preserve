"""Compositing operations with preservation guarantees.

Implements the preservation invariant from validated-solution.md:
  F_t = where(R_t, C_t, S_t)
  Q_t(x,y)=1 => F_t(x,y)=S_t(x,y)

Now with full color management and linear-light compositing per §7.
"""

import numpy as np
from numpy.typing import NDArray

from preserve.color import (
    ColorMetadata,
    composite_linear,
    hard_restore_protected_linear,
)


def composite_over(
    source: NDArray[np.uint8],
    patch: NDArray[np.uint8],
    alpha: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Composite patch over source using alpha.

    Standard source-over: C = A P + (1-A) S
    (Legacy sRGB-space compositing, kept for backward compatibility)
    """
    if alpha.ndim == 2:
        alpha = alpha[..., np.newaxis]

    src_f = source.astype(np.float32)
    patch_f = patch.astype(np.float32)

    result = alpha * patch_f + (1.0 - alpha) * src_f
    return np.clip(result, 0, 255).astype(np.uint8)


def hard_restore_protected(
    candidate: NDArray[np.uint8],
    source: NDArray[np.uint8],
    protected: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Hard-restore protected samples from source.

    Uses indexed assignment (not multiply/add) to guarantee exactness.
    This is the final preservation enforcement step.
    """
    result = candidate.copy()

    if protected.ndim == 2 and result.ndim == 3:
        protected_broadcast = np.broadcast_to(protected[..., np.newaxis], result.shape)
        np.copyto(result, source, where=protected_broadcast)
    else:
        np.copyto(result, source, where=protected)

    return result


def composite_with_preservation(
    source: NDArray[np.uint8],
    generated: NDArray[np.uint8],
    alpha: NDArray[np.float32],
    allowed: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Full composite pipeline with hard preservation guarantee.

    1. Composite generated patch over source using alpha
    2. Hard-restore all protected (non-allowed) pixels from source
    """
    composited = composite_over(source, generated, alpha)
    protected = ~allowed
    return hard_restore_protected(composited, source, protected)


def composite_with_preservation_linear(
    source: NDArray[np.uint8],
    generated: NDArray[np.uint8],
    alpha: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    *,
    source_meta: ColorMetadata | None = None,
    generated_meta: ColorMetadata | None = None,
    out_meta: ColorMetadata | None = None,
    working_space: str = "linear_srgb",
    fg_premultiplied: bool = False,
) -> NDArray[np.uint8]:
    """Full composite pipeline with color management and linear-light compositing.

    1. Convert source and generated to linear working space
    2. Composite in linear light (source-over)
    3. Hard-restore protected pixels in output space
    4. Convert to output color space

    This implements the exact pipeline from validated-solution.md §7.1.
    """
    from preserve.color import ColorSpace

    # Parse working space
    ws = ColorSpace(working_space) if isinstance(working_space, str) else working_space

    # Default metadata
    if source_meta is None:
        source_meta = ColorMetadata()
    if generated_meta is None:
        generated_meta = ColorMetadata()
    if out_meta is None:
        out_meta = source_meta

    # Step 1: Linear-light compositing
    composited = composite_linear(
        background=source,
        foreground=generated,
        alpha=alpha,
        fg_premultiplied=fg_premultiplied,
        bg_meta=source_meta,
        fg_meta=generated_meta,
        out_meta=out_meta,
        working_space=ws,
    )

    # Step 2: Hard-restore protected samples in output space
    protected = ~allowed
    result = hard_restore_protected_linear(
        candidate=composited,
        source=source,
        protected=protected,
        candidate_meta=out_meta,
        source_meta=source_meta,
        out_meta=out_meta,
    )

    return result


def composite_sequence(
    sources: list[NDArray[np.uint8]],
    generated: list[NDArray[np.uint8]],
    alphas: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    keepout: NDArray[np.bool_] | None = None,
    *,
    source_meta: ColorMetadata | None = None,
    generated_meta: ColorMetadata | None = None,
    out_meta: ColorMetadata | None = None,
    working_space: str = "linear_srgb",
    use_linear_compositing: bool = False,
) -> list[NDArray[np.uint8]]:
    """Composite a sequence of frames with preservation.

    Args:
        sources: Source frames
        generated: Generated frames
        alphas: Alpha masks (T, H, W) float32
        allowed: Allowed edit region masks (T, H, W) bool
        keepout: Optional keepout masks for occluder restoration
        source_meta: Source color metadata
        generated_meta: Generated frames color metadata
        out_meta: Output color metadata
        working_space: Working color space for compositing
        use_linear_compositing: Whether to use linear-light compositing (new) or legacy sRGB

    Returns:
        Composited frames
    """
    from preserve.edits.coherence import restore_occluders

    if len(sources) != len(generated):
        raise ValueError("Source and generated frame counts must match")
    if len(sources) != len(alphas):
        raise ValueError("Source and alpha counts must match")

    if use_linear_compositing:
        composite_fn = composite_with_preservation_linear
        kwargs = dict(
            source_meta=source_meta,
            generated_meta=generated_meta,
            out_meta=out_meta,
            working_space=working_space,
        )
    else:
        composite_fn = composite_with_preservation
        kwargs = {}

    composited = [
        composite_fn(s, g, a, r, **kwargs)
        for s, g, a, r in zip(sources, generated, alphas, allowed, strict=True)
    ]
    return restore_occluders(composited, sources, keepout)


def composite_sequence_simple(
    sources: list[NDArray[np.uint8]],
    generated: list[NDArray[np.uint8]],
    alphas: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    keepout: NDArray[np.bool_] | None = None,
) -> list[NDArray[np.uint8]]:
    """Simple wrapper for backward compatibility."""
    return composite_sequence(sources, generated, alphas, allowed, keepout)
