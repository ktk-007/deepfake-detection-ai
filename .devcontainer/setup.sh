#!/usr/bin/env bash
# One-time setup when the codespace is created: CPU PyTorch, app packages, CLIP backbones.
set -e
cd "$(dirname "$0")/.."
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
  || pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r .devcontainer/requirements-cpu.txt \
  || pip install open_clip_torch timm opencv-python-headless numpy pillow scikit-learn tqdm PyYAML huggingface_hub safetensors fastapi uvicorn python-multipart
pip install facenet-pytorch==2.6.0 --no-deps
python -c "
from unmask.app.pipeline import find_models
from unmask.train.loop import load_model_checkpoint
for m in find_models():
    if m.path:
        load_model_checkpoint(m.path, 'cpu')
        print('ready:', m.label)
"
