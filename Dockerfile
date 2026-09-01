# ── Multi-stage SadTalker serverless worker ───────────────────────────────
#
# Stage 1 (build): pytorch/pytorch devel — has nvcc + build tools for any
#   packages that need to compile C/CUDA extensions at install time.
# Stage 2 (runtime): pytorch/pytorch runtime — no compiler, ~5 GB compressed
#   vs the ~14 GB devel image that was causing RunPod pull timeouts.
#
# Model checkpoints are NOT baked in; handler.py downloads them at first
# cold start via ensure_checkpoints().
# ─────────────────────────────────────────────────────────────────────────

# ── Stage 1: build ────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS build

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/OpenTalker/SadTalker.git

WORKDIR /app/SadTalker

# Core numeric / media deps
RUN pip install --no-cache-dir \
        "numpy<2" \
        scipy \
        scikit-image \
        pillow \
        imageio==2.33.1 \
        imageio-ffmpeg==0.4.9 \
    && pip cache purge

# Audio deps
RUN pip install --no-cache-dir \
        librosa \
        numba \
    && pip cache purge

# Vision / face deps
RUN pip install --no-cache-dir \
        opencv-python-headless \
        face_alignment==1.3.5 \
        kornia \
        safetensors \
        pyyaml \
        yacs \
        tqdm \
    && pip cache purge

# Face restoration deps
RUN pip install --no-cache-dir \
        basicsr \
        gfpgan \
        realesrgan \
        facexlib \
    && pip cache purge

# RunPod serverless SDK
RUN pip install --no-cache-dir runpod && pip cache purge

# Create checkpoint dirs (ensure_checkpoints() fills them at runtime)
RUN mkdir -p /app/SadTalker/checkpoints /app/SadTalker/gfpgan/weights

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Runtime system libraries only — no compiler, no git
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from build stage
COPY --from=build /opt/conda/lib/python3.10/site-packages \
                  /opt/conda/lib/python3.10/site-packages

# Copy SadTalker repo (including checkpoint dirs) from build stage
COPY --from=build /app /app

# ── Worker code ───────────────────────────────────────────────────────────
COPY handler.py /app/SadTalker/handler.py

WORKDIR /app/SadTalker
CMD ["python", "-u", "handler.py"]
