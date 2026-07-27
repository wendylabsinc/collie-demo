FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      alsa-utils espeak-ng \
      python3 python3-pip python3-venv python3-gi \
      gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0 \
      gstreamer1.0-tools gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good gstreamer1.0-alsa \
      gstreamer1.0-pipewire pipewire-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv --system-site-packages /app/venv
ENV PATH="/app/venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./

EXPOSE 8099
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8099"]
