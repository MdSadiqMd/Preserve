"""Apple-silicon numerics workarounds shared by every diffusion backend."""

import torch


def wrap_scheduler_step_on_cpu(pipe) -> None:
    """Run the scheduler's update on the CPU in fp32, keeping tensors on the device.

    UniPC's higher-order update amplifies MPS fp32 rounding in its coefficient
    math roughly 5e4x (measured on the VACE path: identical inputs, 6e-7 in,
    0.028 out), which compounds over steps until a render is haze or noise. The
    transformer, VAE and text encoder are bit-exact on MPS; the scheduler's
    tensors are tiny, so stepping on the CPU costs nothing and restores
    CPU-exact output at full MPS speed.
    """
    from diffusers.schedulers.scheduling_utils import SchedulerOutput

    orig_step = pipe.scheduler.step

    def cpu_step(model_output, t, sample, *args, **kwargs):
        out = orig_step(
            model_output.float().cpu(),
            float(t) if torch.is_tensor(t) else t,
            sample.float().cpu(),
            *args,
            **kwargs,
        )
        prev = (out[0] if isinstance(out, tuple) else out.prev_sample).to(sample.device)
        return SchedulerOutput(prev_sample=prev)

    pipe.scheduler.step = cpu_step


def enable_flash_attention() -> bool:
    """Patch torch SDPA with a tiled O(n) flash kernel on MPS (mps-flash-attn).

    PyTorch's MPS SDPA materialises the full score matrix: a 720p Wan render
    (17 frames -> 17.6k tokens, 24 heads) needs tens of GB for one attention
    call and was SIGKILLed in the first transformer step (measured 2026-09-19).
    Numerics match torch SDPA to bf16 precision (max abs diff 2e-3 on 4k tokens).
    Returns False when the kernel is unavailable (non-MPS, missing package).
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        from mps_flash_attn import replace_sdpa
    except ImportError:
        return False
    replace_sdpa()
    return True


def bound_vae_memory(vae) -> None:
    """Tile the Wan VAE and flush the MPS cache after every encoder/decoder call.

    The causal 3D convolutions allocate MPSGraph workspace outside torch's
    allocator: one full-width 720p decoder call needs ~42GB of it (measured
    2026-09-19, 17 frames: driver 52GB, SIGKILL alongside the 24GB DiT).
    Tiles shrink each call and the flush keeps the freed blocks from
    fragmenting the pool; the same decode then peaks at 8GB in 156s.
    """
    vae.enable_tiling()
    for module in (vae.encoder, vae.decoder):
        original = module.forward

        def forward(*args, _original=original, **kwargs):
            out = _original(*args, **kwargs)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            return out

        module.forward = forward
