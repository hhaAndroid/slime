#!/bin/bash
set -ex

# All code will be placed here
export BASE_DIR="/mnt/shared-storage-user/huanghaian/code/slime_package"
cd "${BASE_DIR}"

# ---- pins ----
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
export ENV_NAME="slime_megatron"

# ---- create/activate env ----
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n "${ENV_NAME}" python=3.12 pip -c conda-forge -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# CUDA will be installed into this conda prefix
export CUDA_HOME="$CONDA_PREFIX"

# ---- install CUDA 12.9 toolchain + NCCL + cuDNN ----
conda install -n "${ENV_NAME}" cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
conda install -n "${ENV_NAME}" -c conda-forge cudnn -y

# ---- pytorch cu129 + related pins ----
# NOTE: The original script comment says "prevent installing cuda 13.0 for sglang",
# but it pins cuda-python==13.1.0. Keep identical behavior for reproducibility.
pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# ---- sglang ----
cd "${BASE_DIR}"
if [ ! -d "${BASE_DIR}/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git
fi
cd "${BASE_DIR}/sglang"
git fetch --all
git checkout "${SGLANG_COMMIT}"
pip install -e "python[all]"

pip install cmake ninja

# flash-attn (compile)
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.1

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# ---- Megatron-LM ----
cd "${BASE_DIR}"
if [ ! -d "${BASE_DIR}/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd "${BASE_DIR}/Megatron-LM"
git fetch --all --recurse-submodules
git checkout "${MEGATRON_COMMIT}"
pip install -e .

# ---- slime ----
cd "${BASE_DIR}"
# Prefer using the local slime_package copy if present; otherwise clone upstream into BASE_DIR.
# Your workspace shows slime already under ${BASE_DIR}/slime, so this should just reuse it.
# if [ ! -d "${BASE_DIR}/slime" ]; then
#   git clone https://github.com/THUDM/slime.git "${BASE_DIR}/slime"
# fi
export SLIME_DIR="${BASE_DIR}/slime"
cd "${SLIME_DIR}"
pip install -e .

pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# Compatibility pins
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# ---- apply patches ----
cd "${BASE_DIR}/sglang"
git apply "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch"
cd "${BASE_DIR}/Megatron-LM"
git apply "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch"