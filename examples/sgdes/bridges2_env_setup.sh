#!/bin/bash
# ===================================================================
# Bridges2 TRILL/SGDES setup script
# Uses full paths to conda env executables — does NOT require conda init.
#
# Usage:
#   bash env_setup.sh
# ===================================================================

# --- 0. Load required modules ---
module load anaconda3 || true
module load cuda      || true

# --- 1. Set environment variables ---
export CONDA_ENV="${PROJECT}/conda_env"
export TRILL_DIR="${PROJECT}/sgdes/SGDES"
export SPHERICAL_DIR="${PROJECT}/htp/SPHERICAL"

ENV_DIR="${CONDA_ENV}/sgdes"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

# Prefer conda's libstdc++ over system RHEL8 (fixes GLIBCXX_3.4.26 missing)
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

# --- 2. Create conda env (recreate if python missing or wrong version) ---
if [ ! -x "${PY}" ] || ! "${PY}" -c "import sys; assert sys.version_info[:2] == (3,11)" 2>/dev/null; then
    echo "Creating/recreating conda env at ${ENV_DIR} (Python 3.11)..."
    # Remove stale/incomplete directory so conda create doesn't fail
    if [ -d "${ENV_DIR}" ]; then
        echo "  Removing incomplete env dir ${ENV_DIR}..."
        rm -rf "${ENV_DIR}"
    fi
    conda create -y -p "${ENV_DIR}" python=3.11
fi
echo "Using Python: $("${PY}" --version)"

# --- 3. Update pip, setuptools, wheel ---
echo "Upgrading pip/setuptools/wheel..."
"${PY}" -m pip install -q --upgrade pip wheel
"${PIP}" install -q --force-reinstall "setuptools<71" sympy

# --- 4. Install conda packages (binary tools) ---
echo "Installing conda packages..."
conda install -y -p "${ENV_DIR}" \
    -c conda-forge -c bioconda \
    seqkit ambertools openbabel \
    vina smina fpocket \
    "numpy>=1.26.4,<2.0" pandas scikit-learn \
    "sentencepiece>=0.1.99,<0.2" \
    psutil libstdcxx-ng \
    "cudnn>=8.9,<9" \
    cmake git

# --- 4b. Install foldseek GPU binary from GitHub releases ---
# The conda foldseek binary is CPU-only and cannot generate ProstT5 SS predictions
# (empty _db_ss). The GPU build from GitHub bundles PyTorch+CUDA and works on V100.
echo "Installing foldseek GPU binary..."
if [ ! -x "${ENV_DIR}/bin/foldseek" ]; then
    FOLDSEEK_TGZ="/tmp/foldseek-linux-gpu.tar.gz"
    if [ ! -f "${FOLDSEEK_TGZ}" ]; then
        curl -L \
            "https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-gpu.tar.gz" \
            -o "${FOLDSEEK_TGZ}"
    fi
    tar -xzf "${FOLDSEEK_TGZ}" -C "${ENV_DIR}/bin" --strip-components=2 foldseek/bin/foldseek
    chmod +x "${ENV_DIR}/bin/foldseek"
    echo "foldseek GPU binary installed at ${ENV_DIR}/bin/foldseek"
else
    echo "foldseek already installed, skipping"
fi
"${PIP}" install -q "scipy==1.13.1"
# --- 5. Install PyTorch 2.4.0 + CUDA 12.1 ---
echo "Installing PyTorch 2.4.0+cu121..."
"${PIP}" install -q --upgrade --force-reinstall \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# --- 6. Install Python packages for TRILL ---
echo "Installing Python packages for TRILL..."
"${PIP}" install -q \
    "pyfiglet==0.8.post1" \
    "transformers==4.47.0" \
    "loguru==0.7.3" \
    "GitPython==3.1.46" \
    "fair-esm==2.0.0" \
    "icecream==2.1.10" \
    "pytorch-lightning==2.4.0" \
    "absl-py==2.1.0" \
    "gin-config==0.5.0" \
    "pillow==10.4.0" \
    "biotite==0.39.0" \
    "numpy>=1.26.3,<2.0.0" \
    "fsspec[http]==2024.6.1" \
    sympy \
    rna-fm \
    "tensorflow==2.16.2" \
    "tf-keras==2.16.0" \
    "jax[cuda12]==0.4.25" \
    "jaxlib==0.4.25" \
    "ml-dtypes==0.3.2"

# Reinstall scikit-learn via pip so it links against pip-installed numpy
# (conda sklearn binary may have been compiled against a different numpy ABI)
"${PIP}" install -q --force-reinstall "scikit-learn==1.4.1.post1"

# --- 7. Install PyTorch Geometric (cp311, torch 2.4.0+cu121) ---
echo "Installing PyTorch Geometric..."
"${PIP}" install -q \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_scatter-2.1.2%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_sparse-0.6.18%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_cluster-1.6.3%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/pyg_lib-0.4.0%2Bpt24cu121-cp311-cp311-linux_x86_64.whl" \
    torch-geometric

# --- 8. Install flash-attn (pre-built wheel, torch2.4+cu12, cp311) ---
echo "Installing flash-attn..."
"${PIP}" install -q \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# --- 9. Install TRILL from SGDES fork (editable) ---
echo "Installing TRILL from ${TRILL_DIR}..."
"${PIP}" install -q poetry-core
"${PIP}" install -e "${TRILL_DIR}"

# amortized_bo uses absolute imports — add its parent dir to sys.path via .pth
echo "${TRILL_DIR}/trill/utils/abo" > \
    "${ENV_DIR}/lib/python3.11/site-packages/amortized_bo.pth"

# --- 10. Install SPHERICAL (editable) ---
if [ -d "${SPHERICAL_DIR}" ]; then
    echo "Installing SPHERICAL..."
    "${PIP}" install -q -e "${SPHERICAL_DIR}"
fi

# --- 10b. Re-pin after TRILL install (trill can upgrade these) ---
# setuptools>=72 removes pkg_resources; pyfiglet (used by trill) requires it.
# tensorboard: trill upgrades it to 2.20, but TF 2.16 requires <2.17.
# numpy: trill/transformers may upgrade to 2.x, but JAX 0.4.25 requires <2.0.
# ml-dtypes: must be rebuilt against the re-pinned numpy ABI.
# python-dateutil: conda may install 2.5.x; matplotlib requires >=2.7.
# tornado: calm 0.1.4 pulls in tornado 4.3 which uses collections.MutableMapping
#           removed in Python 3.10+; tornado>=6 uses collections.abc.MutableMapping.
"${PIP}" install -q --force-reinstall \
    "setuptools<71" \
    "tensorboard>=2.16,<2.17" \
    "numpy>=1.26.3,<2.0.0" \
    "ml-dtypes==0.3.2" \
    "python-dateutil>=2.7" \
    "tornado>=6.1"  
    

# --- 11. Verify ---
echo ""
echo "── Verifying installation ──"
"${PY}" -c "import pkg_resources; print('pkg_resources OK')"
"${PY}" -c "import numpy as np; print(f'numpy={np.__version__}')"
"${PY}" -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')"
"${PY}" -c "import jax; print(f'jax={jax.__version__}')"
"${PY}" -c "import trill; print('trill OK')"
"${ENV_DIR}/bin/foldseek" version 2>/dev/null && echo "foldseek OK" || echo "foldseek not found"
"${ENV_DIR}/bin/seqkit"   version 2>/dev/null && echo "seqkit OK"   || echo "seqkit not found"

echo ""
echo "Setup complete. Activate with:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${ENV_DIR}"
