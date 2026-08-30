# SF3D installs CUDA-backed texture_baker/uv_unwrapper submodules at build time.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
ENV USE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    USE_NATIVE_ARCH=0
WORKDIR /worker
COPY sf3d-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG SF3D_REPOSITORY=https://github.com/Stability-AI/stable-fast-3d.git
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake ninja-build && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --recurse-submodules --shallow-submodules ${SF3D_REPOSITORY} /opt/stable-fast-3d \
    && pip install --no-cache-dir -U setuptools==69.5.1 wheel \
    && cd /opt/stable-fast-3d && pip install --no-cache-dir -r requirements.txt
COPY sf3d-worker/worker.py .
CMD ["python", "worker.py"]
