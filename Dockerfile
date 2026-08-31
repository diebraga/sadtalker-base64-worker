FROM drvpn/runpod_serverless_sadtalker_worker:latest

# Pre-download SadTalker + GFPGAN checkpoints at build time so sync_checkpoints() is a no-op at runtime
RUN mkdir -p /app/SadTalker/checkpoints /app/SadTalker/gfpgan/weights && \
    curl -fsSL "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar" \
         -o /app/SadTalker/checkpoints/mapping_00109-model.pth.tar && \
    curl -fsSL "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar" \
         -o /app/SadTalker/checkpoints/mapping_00229-model.pth.tar && \
    curl -fsSL "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" \
         -o /app/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors && \
    curl -fsSL "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors" \
         -o /app/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors && \
    curl -fsSL "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth" \
         -o /app/SadTalker/gfpgan/weights/alignment_WFLW_4HG.pth && \
    curl -fsSL "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
         -o /app/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth && \
    curl -fsSL "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
         -o /app/SadTalker/gfpgan/weights/GFPGANv1.4.pth && \
    curl -fsSL "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
         -o /app/SadTalker/gfpgan/weights/parsing_parsenet.pth

COPY handler.py /app/SadTalker/handler.py
