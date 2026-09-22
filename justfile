set dotenv-load

default:
    @just --list

install:
    cd backend && uv sync
    cd client && pnpm install

dev:
    just backend & just client

backend:
    cd backend && uv run python -m preserve

client:
    cd client && pnpm dev

test:
    cd backend && uv run pytest tests/ -v --ignore=tests/test_sample_video.py --ignore=tests/test_generate_and_edit.py

# Full loop against real models: generate a clip, edit one region, score it.
# Writes data/generated.mp4, data/generated-edited.mp4 and a comparison image.
test-loop:
    cd backend && uv run python tests/test_generate_and_edit.py

# Same, reusing the previously generated clip instead of regenerating it.
test-loop-reuse:
    cd backend && uv run python tests/test_generate_and_edit.py --reuse

fmt:
    cd backend && uv run ruff format src
    cd client && pnpm run format

clean:
    rm -rf backend/.venv backend/data
    rm -rf client/node_modules client/.vinxi client/.output

download-models:
    cd backend && uv run python -m preserve.scripts.download_models

routes:
    cd client && pnpm generate-routes
