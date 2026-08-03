# Installation

Requires Python 3.11 to 3.14, PyTorch, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

## Get the code

Clone the repository and enter it first, because the install commands below do
an editable install of the local checkout (`uv pip install -e .`):

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

On Linux, the PyPI wheel already bundles CUDA (it pulls the NVIDIA runtime
packages in as dependencies), so the plain install above is normally enough. On
Windows and macOS the PyPI wheel is CPU-only.

Use the [PyTorch wheel index](https://pytorch.org/get-started/locally/) when you
need a CUDA version other than the bundled one, or to force a CPU-only build on
Linux (`--index-url https://download.pytorch.org/whl/cpu`). Install torch first,
then the package:

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
uv pip install -e ".[dev]"      # Development tools (pytest, pytest-cov, ruff)
uv pip install -e ".[docs]"     # Build this documentation site locally
```

## Running the tests

The test tools come from the `dev` extra above, so install it first:

```bash
uv pip install -e ".[dev]"
python -m pytest tests/ -v
```
