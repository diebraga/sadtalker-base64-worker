# ── Lean SadTalker serverless worker ──────────────────────────────────────
# Uses the official RunPod PyTorch base (~7 GB compressed) instead of the
# monolithic drvpn image (~15-25 GB).  Model checkpoints are NOT baked in;
# handler.py downloads them on first cold start via ensure_checkpoints().
# Subsequent cold starts skip the download (files already on disk if a
# network volume is attached, or are re-downloaded in ~3-5 min otherwise).
# ---------------------------------------------------------------------------

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── Clone SadTalker ────────────────────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/OpenTalker/SadTalker.git

WORKDIR /app/SadTalker

# ── Python dependencies ────────────────────────────────────────────────────
# torch/torchvision are already in the base; we only add what SadTalker needs
RUN pip install --no-cache-dir \
        face_alignment==1.3.5 \
        imageio==2.33.1 \
        imageio-ffmpeg==0.4.9 \
        librosa \
        numba \
        "numpy<2" \
        opencv-python-headless \
        pillow \
        pyyaml \
        scikit-image \
        scipy \
        safetensors \
        yacs \
        kornia \
        tqdm \
        gfpgan \
        basicsr \
        realesrgan \
        facexlib \
        runpod \
    && pip cache purge

# Create checkpoint dirs (ensure_checkpoints() fills them at runtime)
RUN mkdir -p /app/SadTalker/checkpoints /app/SadTalker/gfpgan/weights

# ── Worker code ────────────────────────────────────────────────────────────
COPY handler.py /app/SadTalker/handler.py

WORKDIR /app/SadTalker
CMD ["python", "-u", "handler.py"]
