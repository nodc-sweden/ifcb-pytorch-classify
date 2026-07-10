# Installation

Requires Python 3.11–3.12, PyTorch, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

## Get the code

Clone the repository and enter it first — the install commands below do an
editable install of the local checkout (`uv pip install -e .`):

```bash
git clone https://github.com/nodc-sweden/ifcb-pytorch-classify.git
cd ifcb-pytorch-classify
```

## CPU only

**Linux/macOS:**

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

**Windows:**

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
```

## With CUDA

PyTorch from PyPI is CPU-only. To get CUDA support, install torch first from the
[PyTorch wheel index](https://pytorch.org/get-started/locally/) for your CUDA
version, then install the package:

**Linux/macOS:**

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # adjust to your CUDA version
uv pip install -e .
```

**Windows:**

```powershell
uv venv
.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # adjust to your CUDA version
uv pip install -e .
```

## Verify the install

Check that the package imports and see whether PyTorch found a GPU:

```bash
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
ifcb-classify --help
```

`CUDA available: True` means training will use the GPU automatically. `False`
is expected for a CPU-only install (the walkthrough still works, just slower);
if you installed the CUDA build and still see `False`, see
[Troubleshooting → PyTorch doesn't see my GPU](troubleshooting.md#pytorch-doesnt-see-my-gpu).

## Optional extras

```bash
uv pip install -e ".[mlflow]"   # MLflow experiment tracking
uv pip install -e ".[wandb]"    # Weights & Biases experiment tracking
uv pip install -e ".[chains]"   # YOLO chain counting
uv pip install -e ".[dev]"      # Development tools (pytest, ruff)
uv pip install -e ".[docs]"     # Build this documentation site locally
```

## Running the tests

```bash
python -m pytest tests/ -v
```
