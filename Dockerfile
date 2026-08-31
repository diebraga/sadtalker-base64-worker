FROM drvpn/runpod_serverless_sadtalker_worker:latest

# Copy and run the download script (curl/wget not available in base image)
COPY download_checkpoints.py /tmp/download_checkpoints.py
RUN python3 /tmp/download_checkpoints.py && rm /tmp/download_checkpoints.py

COPY handler.py /app/SadTalker/handler.py
