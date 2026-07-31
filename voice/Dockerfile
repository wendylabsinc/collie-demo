# syntax=docker/dockerfile:1.6

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG CYCLONEDDS_VERSION=0.10.5

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config \
      gcc python3-dev portaudio19-dev libportaudio2 \
      libsrtp2-1 libffi-dev libssl-dev \
    && git clone --depth 1 --branch ${CYCLONEDDS_VERSION} \
      https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC=ON \
    && cmake --build /tmp/cyclonedds/build -j2 \
    && cmake --install /tmp/cyclonedds/build \
    && ldconfig \
    && rm -rf /tmp/cyclonedds /var/lib/apt/lists/*

ENV CYCLONEDDS_HOME=/usr/local \
    PYTHONUNBUFFERED=1 \
    GO2_IP=192.168.123.161 \
    COLLIE_BASE_URL=http://127.0.0.1:8096 \
    COLLIE_VOICE_PORT=8098 \
    COLLIE_VOICE_AUTO_START=1 \
    COLLIE_CAMERA_JPEG_QUALITY=85 \
    COLLIE_BARK_UUID=161387de-21ab-4f0b-b4e9-97124b000d06

WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip uninstall -y opencv-python \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
      "opencv-python-headless>=4.8,<5" \
    && python3 -c 'import cv2; print(cv2.__version__)'

COPY __init__.py camera_broker.py commands.py main.py ./

EXPOSE 8098
CMD ["python3", "main.py"]
