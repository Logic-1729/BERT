#!/usr/bin/env bash
# Setup conda environment and install dependencies for BERT + interpretability project.
#
# Usage:
#     bash scripts/setup_env.sh
#
# What it does:
#     1. Creates conda env "bert-interp" (Python 3.10) if not exists
#     2. Installs PyTorch (CUDA 12.1) + all requirements
#     3. Configures pip mirror if in China
#
set -euo pipefail

ENV_NAME="bert-interp"
PYTHON_VER="3.10"

# ---- helpers ----
red()  { echo -e "\033[31m$*\033[0m"; }
green(){ echo -e "\033[32m$*\033[0m"; }
bold() { echo -e "\033[1m$*\033[0m"; }

# ---- pip mirror (Tsinghua, works reliably in China) ----
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
pip_flags="-i ${PIP_MIRROR}"

# ---- step 1: create conda env ----
if conda info --envs | grep -q "^${ENV_NAME} "; then
    green "[1/5] conda env '${ENV_NAME}' already exists, skip creation."
else
    bold "[1/5] Creating conda env '${ENV_NAME}' (python=${PYTHON_VER}) ..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
    green "       Done."
fi

# ---- step 2: activate ----
bold "[2/5] Activating '${ENV_NAME}' ..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "       Python: $(which python)"

# ---- step 3: install PyTorch (CUDA 12.1) ----
bold "[3/5] Installing PyTorch (CUDA) ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ${pip_flags}
green "       Done."

# ---- step 4: install project deps ----
bold "[4/5] Installing project dependencies (shap may take 3-5 minutes) ..."
pip install -r requirements.txt ${pip_flags}
green "       Done."

# ---- step 5: verify ----
bold "[5/5] Verifying installation ..."
python -c "
import torch; print(f'  PyTorch {torch.__version__}  | CUDA: {torch.cuda.is_available()}')
import transformers; print(f'  Transformers {transformers.__version__}')
import shap; print(f'  SHAP {shap.__version__}')
import lime; print(f'  LIME OK')
"
green "       All OK."

# ---- done ----
echo ""
echo "============================================"
green " Setup complete. Environment: ${ENV_NAME}"
echo "============================================"
echo ""
echo "If you are in China, set the HuggingFace mirror before downloading data:"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo ""
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/prepare_thucnews.py"
echo "  python -m src.train --config configs/bert_thucnews.yaml"
