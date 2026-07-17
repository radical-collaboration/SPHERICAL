#!/bin/bash
# =============================================================================
# SGDES environment setup — Delta HPC (NCSA)
#
# Creates the Python 3.11 venv, clones repos, and installs all dependencies
# to reproduce /u/mgoliyad1/ve/sgdes.
#
# Usage:
#   bash delta_env_setup.sh [--env-dir DIR] [--sgdes-dir DIR] [--spherical-dir DIR]
#
# Defaults:
#   ENV_DIR       = /u/$USER/ve/sgdes
#   SGDES_DIR     = /scratch/bblj/$USER/SGDES
#   SPHERICAL_DIR = /scratch/bblj/$USER/SPHERICAL
# =============================================================================
# Only apply strict error handling when run as a subprocess (bash script.sh),
# NOT when sourced (. ./script.sh) — sourcing with set -e exits the login shell
# on any error, which looks like being logged out of the machine.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Parse optional overrides ──────────────────────────────────────────────────
ENV_DIR="${ENV_DIR:-/u/${USER}/ve/sgdes}"
SGDES_DIR="${SGDES_DIR:-/scratch/bblj/${USER}/sgdes}"
SPHERICAL_DIR="${SPHERICAL_DIR:-/scratch/bblj/${USER}/SPHERICAL}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-dir)       ENV_DIR="$2";       shift 2 ;;
        --sgdes-dir)     SGDES_DIR="$2";     shift 2 ;;
        --spherical-dir) SPHERICAL_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PY="${ENV_DIR}/bin/python3.11"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  ENV_DIR       = ${ENV_DIR}"
echo "  SGDES_DIR     = ${SGDES_DIR}"
echo "  SPHERICAL_DIR = ${SPHERICAL_DIR}"
echo "================================================================="

# ── 0. Clone repositories ─────────────────────────────────────────────────────
echo ""
echo "── Step 0: Cloning repositories ──"

if [ ! -d "${SGDES_DIR}/.git" ]; then
    echo "Cloning SGDES → ${SGDES_DIR}"
    git clone git@github.com:WEIFZH/SGDES.git "${SGDES_DIR}"
else
    echo "SGDES already cloned at ${SGDES_DIR}"
fi

if [ ! -d "${SPHERICAL_DIR}/.git" ]; then
    echo "Cloning SPHERICAL → ${SPHERICAL_DIR}"
    git clone git@github.com:radical-collaboration/SPHERICAL.git "${SPHERICAL_DIR}"
else
    echo "SPHERICAL already cloned at ${SPHERICAL_DIR}"
fi


# ── 1. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Creating venv ──"

# The Cray PE Python (3.11.7) is itself packaged by conda-forge and reports
# sys.version as '3.11.13 | packaged by conda-forge | ...', which its own
# platform.py regex cannot parse — a Cray PE packaging bug that breaks
# get-pip.py, cloudpickle, and other tools that call platform.python_version().
# Use python3.11 from the anaconda3 module (conda-forge Python with the patched
# platform.py) to create the venv instead.
BASE_PY=$(command -v python3.11 2>/dev/null || true)
if [ -z "${BASE_PY}" ]; then
    echo "python3.11 not in PATH — loading anaconda3 module..."
    module load cray-python/3.11.7 2>/dev/null || true
    BASE_PY=$(command -v python3.11 2>/dev/null || true)
fi
if [ -z "${BASE_PY}" ]; then
    echo "ERROR: python3.11 not found even after loading anaconda3."
    echo "       Run: module load anaconda3"
    exit 1
fi
echo "Using Python: ${BASE_PY} ($(${BASE_PY} --version))"

if [ ! -x "${PY}" ]; then
    echo "Creating venv at ${ENV_DIR}..."
    "${BASE_PY}" -m venv "${ENV_DIR}"
else
    echo "venv already exists at ${ENV_DIR}"
fi

# Convenience symlinks
ln -sf "${ENV_DIR}/bin/python3.11" "${ENV_DIR}/bin/python"  2>/dev/null || true
ln -sf "${ENV_DIR}/bin/python3.11" "${ENV_DIR}/bin/python3" 2>/dev/null || true

echo "Python: $("${PY}" --version)"

# ── 2. Bootstrap pip / setuptools ────────────────────────────────────────────
echo ""
echo "── Step 2: Bootstrapping pip ──"
"${PY}" -m pip install -q --upgrade pip wheel
"${PIP}" install -q --force-reinstall "setuptools<71" sympy

# ── 3. Core ML stack: PyTorch 2.4.0+cu121, TensorFlow, JAX ──────────────────
# transformers >= 4.44 requires PyTorch >= 2.4; use 2.4.0 to match bridges2
# and keep PyG wheel URLs consistent.
echo ""
echo "── Step 3: PyTorch 2.4.0+cu121 ──"
"${PIP}" install -q \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "── Step 3b: PyTorch Geometric (cu121, torch 2.4.0) ──"
"${PIP}" install -q \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/pyg_lib-0.4.0%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_scatter-2.1.2%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_sparse-0.6.18%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_cluster-1.6.3%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    torch-geometric

echo ""
echo "── Step 3c: TensorFlow 2.16.2 + JAX 0.4.25 ──"
"${PIP}" install -q \
    "tensorflow==2.16.2" \
    "tf-keras==2.16.0" \
    "tensorflow-io-gcs-filesystem==0.37.1"

"${PIP}" install -q \
    "jax==0.4.25" \
    "jaxlib==0.4.25" \
    "ml-dtypes==0.3.2"

# ── 4. Dragon / Rhapsody / Radical ────────────────────────────────────────────
echo ""
echo "── Step 4: Dragon HPC + Rhapsody + Radical ──"
# dragonhpc and rhapsody-py are versioned in pyproject.toml and installed via step 7.
# Install only radical.asyncflow here (not in pyproject.toml extras).
"${PIP}" install -q \
    "radical.asyncflow==0.3.1"

# Re-pin torch + numpy after steps 3b/4 which may upgrade them via transitive deps.
"${PIP}" install -q --force-reinstall \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"${PIP}" install -q --force-reinstall "numpy>=1.26.3,<1.27" "ml-dtypes==0.3.2"

# ── 5. Bio/chem and ML utilities ──────────────────────────────────────────────
echo ""
echo "── Step 5: Bio/ML packages ──"
"${PIP}" install -q \
    "accelerate==1.13.0" \
    "transformers==4.57.6" \
    "huggingface_hub==0.36.2" \
    "safetensors==0.7.0" \
    "datasets==2.21.0" \
    "tokenizers==0.22.2" \
    "pytorch-lightning==2.6.1" \
    "deepspeed==0.17.6" \
    "fair-esm==2.0.0" \
    "biopython==1.81" \
    "biotite==0.39.0" \
    "scikit-learn==1.4.1.post1" \
    "scipy==1.17.1" \
    "numpy>=1.26.3,<2.0.0" \
    "pandas" \
    "rdkit==2024.9.6" \
    "e3nn==0.5.9" \
    "torch-geometric" \
    "xgboost==1.7.6" \
    "umap-learn==0.5.7" \
    "pynndescent==0.6.0" \
    "einops==0.8.2" \
    "ema-pytorch==0.7.9" \
    "rotary-embedding-torch==0.8.9" \
    "x-transformers==2.17.9" \
    "torchmetrics==1.9.0" \
    "gin-config==0.5.0" \
    "autograd==1.8.0" \
    "autograd-gamma==0.5.0" \
    "absl-py==2.4.0" \
    "rna-fm==0.2.2" \
    "sentencepiece==0.1.99" \
    "psutil==7.2.2" \
    "GitPython==3.1.46" \
    "pyfiglet==0.8.post1" \
    "icecream==2.2.0" \
    "PyYAML==6.0.3" \
    "tqdm==4.67.3" \
    "rich==14.3.3" \
    "click==8.3.2" \
    "requests==2.33.1" \
    "fsspec[http]==2024.6.1" \
    "h5py==3.14.0" \
    "pillow==12.2.0" \
    "matplotlib" \
    "seaborn==0.13.2" \
    "plotly==6.6.0" \
    "bokeh==3.9.0" \
    "tensorboard>=2.16,<2.17" \
    "tensorboardX==2.6.4" \
    "hydra-core==1.3.2" \
    "omegaconf" \
    "pydantic==2.12.5" \
    "typing_extensions==4.15.0" \
    "python-dateutil>=2.7" \
    "tornado>=6.1" \
    "poetry-core"

# vit-pytorch 1.19.1 requires torch>=2.4 (now satisfied); install without deps to
# avoid pulling in a newer torch that would override the cu121 pin.
"${PIP}" install -q --no-deps "vit-pytorch==1.19.1"
# calm 0.1.1 from GitHub (PyPI only has 0.1.3+; this commit has no tornado/iso8601 conflicts).
"${PIP}" install -q "git+https://github.com/martinez-zacharya/CaLM.git@2b3a9b8985b0940ed8277e4d11d21e530b6aaf34"

# ── 6. TRILL from SGDES fork (editable) ───────────────────────────────────────
echo ""
echo "── Step 6: TRILL from SGDES fork ──"
"${PIP}" install -q -e "${SGDES_DIR}"

# amortized_bo uses absolute imports — expose its parent via .pth
echo "${SGDES_DIR}/trill/utils/abo" > \
    "${ENV_DIR}/lib/python3.11/site-packages/amortized_bo.pth"
echo "amortized_bo.pth written"

# Re-pin torch after TRILL which upgrades it to latest.
"${PIP}" install -q --force-reinstall \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 7. SPHERICAL (editable) ────────────────────────────────────────────────────
echo ""
echo "── Step 7: SPHERICAL ──"
"${PIP}" install -q -e "${SPHERICAL_DIR}[esm2,dragon,dev,plotting]"

# ── 8. Pin versions that downstream installs may upgrade ─────────────────────
# Steps 6–7 (TRILL + SPHERICAL extras) can upgrade torch and numpy beyond the
# versions installed in steps 3 and 5.  Force-reinstall pins them back down.
echo ""
echo "── Step 8: Re-pinning critical versions ──"

# Re-pin PyTorch (must use cu121 index to get the correct CUDA variant).
"${PIP}" install -q --force-reinstall \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Re-pin NumPy and other TF/sklearn/numba-constrained packages.
# numpy must be <1.27 (numba 0.58.1) and <2.0 (TF 2.16.2, sklearn 1.4.x).
"${PIP}" install -q --force-reinstall \
    "numpy>=1.26.3,<1.27" \
    "protobuf>=3.20.3,<5.0.0dev" \
    "setuptools<71" \
    "tensorboard>=2.16,<2.17" \
    "ml-dtypes==0.3.2" \
    "python-dateutil>=2.7" \
    "tornado>=6.1" \
    "fsspec[http]==2024.6.1" \
    "cloudpickle>=3.0"

# pynvml is a deprecated wrapper around nvidia-ml-py that triggers a FutureWarning
# from torch.cuda.  nvidia-ml-py (already a direct dep) provides the same API.
"${PIP}" uninstall -q -y pynvml 2>/dev/null || true

# ── 9. foldseek GPU binary ────────────────────────────────────────────────────
echo ""
echo "── Step 9: foldseek GPU binary ──"
if [ ! -x "${ENV_DIR}/bin/foldseek" ]; then
    # Use the known-good binary from the conda env (same version that passes GPU search).
    CONDA_FOLDSEEK="/u/mgoliyad1/conda_env/sgdes/bin/foldseek"
    if [ -x "${CONDA_FOLDSEEK}" ]; then
        cp "${CONDA_FOLDSEEK}" "${ENV_DIR}/bin/foldseek"
        chmod +x "${ENV_DIR}/bin/foldseek"
        echo "foldseek binary copied from conda env"
    else
        FOLDSEEK_TGZ="/tmp/foldseek-linux-gpu.tar.gz"
        [ ! -f "${FOLDSEEK_TGZ}" ] && curl -L \
            "https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-gpu.tar.gz" \
            -o "${FOLDSEEK_TGZ}"
        tar -xzf "${FOLDSEEK_TGZ}" -C "${ENV_DIR}/bin" \
            --strip-components=2 foldseek/bin/foldseek
        chmod +x "${ENV_DIR}/bin/foldseek"
        echo "foldseek GPU binary installed from GitHub"
    fi
else
    echo "foldseek already installed, skipping"
fi

# ── 10. seqkit binary ─────────────────────────────────────────────────────────
echo ""
echo "── Step 10: seqkit v2.8.2 ──"
if [ ! -x "${ENV_DIR}/bin/seqkit" ]; then
    SEQKIT_TGZ="/tmp/seqkit_linux_amd64.tar.gz"
    [ ! -f "${SEQKIT_TGZ}" ] && curl -L \
        "https://github.com/shenwei356/seqkit/releases/download/v2.8.2/seqkit_linux_amd64.tar.gz" \
        -o "${SEQKIT_TGZ}"
    tar -xzf "${SEQKIT_TGZ}" -C "${ENV_DIR}/bin" seqkit
    chmod +x "${ENV_DIR}/bin/seqkit"
    echo "seqkit installed"
else
    echo "seqkit already installed, skipping"
fi

# ── 11. Apply slurm patch  ────────────────────────────────────────────────────────────────
echo ""
echo "── Verifying installation ──"
"${PY}" ${SPHERICAL_DIR}/workflows/apply_slurm_patch.py

# ── 12. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "── Verifying installation ──"
_check() {
    local label="$1"; shift
    if out=$("$@" 2>&1); then
        echo "${label}: OK  (${out})"
    else
        echo "WARNING: ${label} failed"
        echo "  ${out}" | head -3
    fi
}

_check "pkg_resources"     "${PY}" -c "import pkg_resources; print('pkg_resources')"
_check "numpy"             "${PY}" -c "import numpy as np; print(np.__version__)"
_check "torch"             "${PY}" -c "import torch; print(torch.__version__, 'cuda=' + str(torch.cuda.is_available()))"
_check "tensorflow"        "${PY}" -c "import tensorflow as tf; print(tf.__version__)"
_check "jax"               "${PY}" -c "import jax; print(jax.__version__)"
_check "trill"             "${PY}" -c "import trill; print('ok')"
_check "radical.asyncflow" "${PY}" -c "import radical.asyncflow; print('ok')"
_check "rhapsody"          "${PY}" -c "import rhapsody; print('ok')"
_check "foldseek"          "${ENV_DIR}/bin/foldseek" version
_check "seqkit"            "${ENV_DIR}/bin/seqkit" version

echo ""
echo "================================================================="
echo "Setup complete."
echo ""
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo "================================================================="
