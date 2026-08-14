# syntax=docker/dockerfile:1.6

FROM python:3.11-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CYCLONEDDS_VERSION=0.10.5
ARG UNITREE_SDK2_PYTHON_REF=e4cd91f051aaa77a70600e3d2bf7f50889db1980

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config portaudio19-dev \
      espeak-ng \
    && git clone --depth 1 --branch ${CYCLONEDDS_VERSION} \
      https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
      -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
    && cmake --build /tmp/cyclonedds/build -j2 \
    && cmake --install /tmp/cyclonedds/build \
    && python3 -m pip install --no-cache-dir --no-deps \
      "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@${UNITREE_SDK2_PYTHON_REF}" \
    && ldconfig \
    && mkdir -p /opt/thermal-audio \
    && espeak-ng -s 145 -w /opt/thermal-audio/woof_low_battery.wav \
      "Low battery" \
    && rm -rf /tmp/cyclonedds /var/lib/apt/lists/*

RUN python3 -m venv /opt/thermal-venv \
    && CYCLONEDDS_HOME=/usr/local /opt/thermal-venv/bin/pip install --no-cache-dir --no-deps \
      "cyclonedds==0.10.2" \
      "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@${UNITREE_SDK2_PYTHON_REF}" \
    && /opt/thermal-venv/bin/pip install --no-cache-dir \
      "unitree_webrtc_connect==2.1.2"

FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/libddsc.so* /usr/local/lib/
COPY --from=builder /opt/thermal-venv /opt/thermal-venv
COPY --from=builder /opt/thermal-audio /opt/thermal-audio

ENV PYTHONUNBUFFERED=1 \
    PATH=/opt/thermal-venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib \
    CYCLONEDDS_HOME=/usr/local \
    WOOF_THERMAL_PORT=8102 \
    WOOF_THERMAL_STATE_DIR=/state \
    GO2_IP=192.168.123.161 \
    GO2_NETWORK_INTERFACE=enP8p1s0

WORKDIR /app
COPY monitor.py ./

EXPOSE 8102
CMD ["python3", "monitor.py"]
