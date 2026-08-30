FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl xz-utils libx11-6 libxxf86vm1 libxcursor1 libxi6 \
    libxrandr2 libxinerama1 libegl1 libwayland-client0 libxkbcommon0 \
    libdbus-1-3 libgl1 libglib2.0-0 libsm6 libice6 libfontconfig1 \
    libxrender1 libxfixes3 libxext6 libgomp1 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -o /tmp/blender.tar.xz \
    && tar -xJf /tmp/blender.tar.xz -C /opt \
    && mv /opt/blender-4.2.0-linux-x64 /opt/blender \
    && rm /tmp/blender.tar.xz \
    && python3 -m pip install --no-cache-dir requests
# /opt/blender retains the official tarball's bundled license files.
ENV PATH=/opt/blender:${PATH}
WORKDIR /worker
COPY worker.py process.py ./
CMD ["python3", "worker.py"]
