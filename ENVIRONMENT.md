# Exercise environment

Run commands from the repository directory. Conda manages Python and common
packages; pip installs exactly one platform-specific PyTorch build.
The `nodefaults` entry prevents inherited global channels from being added.
The pthreads OpenBLAS build avoids a duplicate OpenMP runtime with the macOS
PyTorch wheel.
Only run the installation step after environment creation succeeds; explicit
`conda run -n dna-exercises` commands cannot silently install into base.

## Create the shared environment

```sh
conda env create -f environment.yml
conda activate dna-exercises
```

## Install PyTorch: choose ONE

Apple Silicon macOS (MPS if available; otherwise CPU):

```sh
conda run -n dna-exercises python -m pip install -r requirements-torch-macos.txt
```

Linux with NVIDIA GPU, using the CUDA 12.6 build:

```sh
nvidia-smi
conda run -n dna-exercises python -m pip install -r requirements-torch-cuda126.txt
```

Check the GPU model and driver against the selected PyTorch build before
installing. CUDA 12.6 is an example pinned build, not a determination that an
unknown server is compatible. Mac cannot use NVIDIA CUDA. Official PyTorch
wheels provide runtime dependencies; compiling CUDA extensions may additionally
require a local CUDA toolkit. Do not install multiple torch variants together.

Linux/Windows without GPU:

```sh
conda run -n dna-exercises python -m pip install -r requirements-torch-cpu.txt
```

## Notebook kernel

Select the `dna-exercises` Python interpreter/kernel in VS Code. If it does not
appear, register it:

```sh
python -m ipykernel install --user --name dna-exercises --display-name "Python (dna-exercises)"
```

## Verify in a notebook cell

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.ones(4, device=device)
print("Selected:", device, "sum:", x.sum().item())
```

A successful tensor operation checks basic device execution, not the whole
training workload. Some MPS operations may need separate compatibility checks.

## Reproducibility

The common dependency ranges allow platform-specific resolution; environment.yml
is not a complete lockfile. After installation and verification, capture an exact
snapshot on EACH platform (replace `<platform>` with `macos` or `linux-cuda126`):

```sh
conda env export --no-builds > environment-resolved-<platform>.yml
python -m pip freeze > pip-resolved-<platform>.txt
python -m pip check
```

Resolved snapshots describe their originating platform; do not use a Mac snapshot
as a Linux CUDA lockfile. Keep the specification and the chosen torch requirements
file together. First exercise code must still use only the standard library.

Official build instructions:
https://pytorch.org/get-started/previous-versions/

## Training visualization

TensorBoard is included in environment.yml. For an already-created environment,
install the new dependency explicitly into that environment:

```sh
conda run -n dna-exercises python -m pip install 'tensorboard>=2,<3'
conda run -n dna-exercises python -m pip check
```

Alternatively, synchronize the full Conda specification (without pruning the
separately installed PyTorch build):

```sh
conda env update -n dna-exercises -f environment.yml
```

Launch from the project directory:

```sh
conda run -n dna-exercises tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

For a remote server, forward its port over SSH instead of exposing TensorBoard
publicly. Refresh the per-platform dependency snapshot after changes.
