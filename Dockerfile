FROM drvpn/runpod_serverless_sadtalker_worker:latest

# Pre-download SadTalker + GFPGAN checkpoints at build time using Python (curl/wget not in base image)
RUN python3 -c "
import urllib.request, os
os.makedirs('/app/SadTalker/checkpoints', exist_ok=True)
os.makedirs('/app/SadTalker/gfpgan/weights', exist_ok=True)
files = [
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar', '/app/SadTalker/checkpoints/mapping_00109-model.pth.tar'),
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar', '/app/SadTalker/checkpoints/mapping_00229-model.pth.tar'),
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors', '/app/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors'),
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors', '/app/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth', '/app/SadTalker/gfpgan/weights/alignment_WFLW_4HG.pth'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', '/app/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth'),
    ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', '/app/SadTalker/gfpgan/weights/GFPGANv1.4.pth'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', '/app/SadTalker/gfpgan/weights/parsing_parsenet.pth'),
]
for url, path in files:
    print('Downloading', url)
    urllib.request.urlretrieve(url, path)
    print('Saved', path)
"

COPY handler.py /app/SadTalker/handler.py
