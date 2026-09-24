# Preserve

Generate a video from a prompt, then change one part of it — with a pixel-level
guarantee that nothing else moves.

```
"a red sports car driving on an empty highway"   -> generates a clip
"make the car blue"                              -> only the car changes
```

## Architecture

The core principle: **AI proposes pixels inside an authorized work region; the application enforces preservation outside.**

Each edit is routed to the least generative method that can express it
(`validated-solution.md` 3.3):

| Prompt | Method | Uses a model? |
|---|---|---|
| "make the car blue" | tracked matte + colour transform | no |
| "remove the car" | background plate from other frames, MiniMax-Remover for unknown pixels | yes |
| "remove his cap" | attached object: the region (object, its shadow on the wearer, a holding hand) is erased classically, FLUX.2 klein completes the wearer on the middle frame (hair, eyes, mouth), flow-warped copies anchor the clip ends (or klein edits an end frame directly when flow cannot carry the fill), VACE propagates between anchors | yes |
| "remove the logo" | flat-surface target: same path, whole print erased before the image editor so no embossed outline survives | yes |
| "replace the car with a taxi" | klein proposes the replacement on one frame at native resolution (its footprint may grow the region), VACE propagates between anchors | yes |
| "remove their caps" | every strong grounded instance is tracked and refined separately | yes |

Every edit produces two files: a **lossless master** the preservation guarantee is
proved against, and a playable H.264 derivative. A lossy encode changes samples
everywhere, so it can never carry the exactness claim.

```
immutable source asset
  → define allowed/protected regions (matte + wearer shadow + held hand + colour rim,
      each pixel marked with why it is in the edit; `PRESERVE_DEBUG_DIR` dumps the matte)
  → recover known pixels from real frames where possible
  → generate only genuinely unknown content, inside a native-resolution crop:
      image editor on the middle frame → flow-warped anchors at both ends → video model between them
  → composite patch over original
  → hard-restore protected source samples
  → verify preservation before output (0 changed pixels outside mask)
```

Every stage is audited visually, not by its score: `backend/tests/edge_suite.sh` renders the
attached-object cases (cap, popsicle, glasses, walking hat, shirt logo) to comparison sheets, and
`docs/decisions.md` records what each change measured.

## System Requirements

- **Apple Silicon Mac** with 48GB+ unified memory (M4 Max recommended)
- macOS 12.3+ for MPS support
- Python 3.11+
- Node.js 20+
- pnpm
- uv

## Quick Start

```bash
# Install dependencies
just install

# Models are already downloaded to backend/models/propainter/
# If not, run: just download-models

# Start development servers
just dev
```

Backend: http://localhost:8000  
Client: http://localhost:3000

## Project Structure

```
preserve/
├── backend/                 # Python FastAPI backend
│   ├── src/preserve/
│   │   ├── api.py          # REST API endpoints
│   │   ├── pipeline.py     # Main editing pipeline
│   │   ├── composite.py    # Compositing with preservation
│   │   ├── verify.py       # Preservation verification
│   │   ├── mask.py         # Mask creation/manipulation
│   │   ├── video.py        # Video I/O
│   │   ├── config.py       # Settings with auto device detection
│   │   ├── models_config.yaml  # Model configuration
│   │   ├── inpaint/        # Inpainting backends
│   │   │   └── propainter.py   # ProPainter (ICCV 2023)
│   │   ├── propainter_model/   # ProPainter model code
│   │   └── raft/           # RAFT optical flow
│   ├── models/             # Model weights
│   │   └── propainter/
│   │       ├── ProPainter.pth
│   │       ├── recurrent_flow_completion.pth
│   │       └── raft-things.pth
│   └── tests/              # Pytest tests
├── client/                 # TanStack Start frontend
│   └── src/
│       ├── routes/         # File-based routing
│       ├── components/     # React components
│       └── lib/            # API client, types
├── problem.md              # Original problem statement
└── validated-solution.md   # Research-backed architecture
```

## Removal Backends

### MiniMax-Remover (default)

MiniMax-Remover (NeurIPS 2025) is a Wan-1.3B-class video DiT trained specifically
to fill a masked region with background instead of regenerating the object.
12 steps, no prompt, no CFG. Weights: `zibojia/minimax-remover`
(CC BY-NC 4.0, non-commercial) under `backend/models/minimax-remover/`.

Removal composes three stages: real background recovered from other frames
where the camera motion registers, the generative fill for pixels no frame
reveals, and a membrane seam match so the fill's tone meets the source.

Targets attached to another detected object (a cap on a head, glasses on a
face) skip the plate entirely, since the hidden surface is the wearer, and go
to a prompted reveal fill on the replacement backend.

### ProPainter

ProPainter (ICCV 2023) propagation-first inpainting remains available with
`PRESERVE_INPAINT_BACKEND=propainter` for footage with a static background
visible in other frames.

Configuration in `backend/src/preserve/models_config.yaml` (`inpainting:`).

## Generation

Wan2.2-TI2V-5B at 720p@24fps. On Apple silicon the backend keeps the whole
stack resident (CPU offload leaks MPS memory), steps the scheduler on the CPU,
patches flash attention over MPS SDPA and tiles the VAE with a cache flush;
see `backend/src/preserve/mps.py`.

## Preservation Guarantees

The system enforces strict preservation:

1. **Hard sample restoration**: Protected pixels copied from source via indexed assignment
2. **Verification**: Every output tested for exact pixel equality outside allowed region
3. **Failure policy**: Jobs fail if any protected pixel changes

Test output:
```
tests/test_propainter_integration.py::test_propainter_preservation_guarantee PASSED
```

## API

```
GET  /health                 Health check with device info
GET  /config                 Current configuration

POST /videos/upload          Upload video file
GET  /videos/{id}            Get video metadata
GET  /videos/{id}/stream     Stream video file
GET  /videos/{id}/frame/{ms} Get frame at timestamp as JPEG

POST /jobs                   Create edit job
GET  /jobs/{id}              Get job status
GET  /jobs/{id}/result       Download result video
GET  /jobs                   List recent jobs
```

## Configuration

Environment variables (or `.env` in backend/):

```bash
PRESERVE_DEVICE=auto         # auto, mps, cuda, or cpu
PRESERVE_DTYPE=float16       # float32, float16, bfloat16
PRESERVE_INPAINT_BACKEND=minimax  # or propainter
PRESERVE_DEBUG=false
PRESERVE_MODEL_DIR=./models
```

## Testing

```bash
# Run all tests
just test

# Run specific test
cd backend && uv run pytest tests/test_propainter_integration.py -v
```

## Research References

- [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover) - NeurIPS 2025
- [ProPainter](https://github.com/sczhou/ProPainter) - ICCV 2023
- [Lucy Edit](https://huggingface.co/decart-ai/Lucy-Edit-1.1-Dev) - instruction video editing on Wan2.2-5B
- See `validated-solution.md` for complete reference list with 31 citations
