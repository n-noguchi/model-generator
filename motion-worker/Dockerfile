FROM blender_runtime AS blender
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
RUN apt-get update && apt-get install -y --no-install-recommends git libxi6 libxrender1 libxfixes3 libxkbcommon0 libsm6 libgl1 libegl1 libxext6 libxxf86vm1 libxcursor1 libxrandr2 libwayland-client0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 gdown==5.2.0 requests==2.32.3 pytest==8.3.5 \
    && pip install --no-cache-dir git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
ARG MDM_COMMIT=ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778
RUN git clone https://github.com/GuyTevet/motion-diffusion-model.git /opt/mdm && cd /opt/mdm && git checkout ${MDM_COMMIT}
COPY motion-worker/humanml-only.patch /tmp/humanml-only.patch
RUN cd /opt/mdm && git apply --unidiff-zero /tmp/humanml-only.patch
COPY --from=blender /opt/blender /opt/blender
ENV PATH=/opt/blender:${PATH} PYTHONPATH=/opt/mdm PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
WORKDIR /worker
COPY motion-worker/*.py ./
COPY sf3d-worker/gpu_runtime.py ./
CMD ["python", "worker.py"]
