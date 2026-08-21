#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# MiniMax H3's 720p/768P profile is 1344x768 and the model always generates at
# 24 FPS. This requests 240 frames (~10 seconds). CP4 is represented by the
# repository's Ulysses/context-parallel degree. The bundled astronaut image is
# used by default; provide another image as the first argument or set IMAGE.
# Override MODEL, PROMPT, OUTPUT, or PYTHON from the environment when needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODEL="${MODEL:-MiniMaxAI/MiniMax-H3}"
DEFAULT_IMAGE="${SCRIPT_DIR}/astronaut.jpg"
IMAGE="${IMAGE:-${1:-${DEFAULT_IMAGE}}}"
PROMPT="${PROMPT:-The astronaut slowly rises from the lunar dust, turns toward the camera, and begins walking toward a distant ridge as Earth glows above the horizon; the camera makes a gentle cinematic push-in, with suit lights, drifting dust, subtle radio chatter, breathing, and wind moving naturally.}"
OUTPUT="${OUTPUT:-minimax_h3_fl2va_720p.mp4}"
PYTHON="${PYTHON:-python}"

if [[ ! -f "${IMAGE}" ]]; then
  echo "Image not found: ${IMAGE}" >&2
  echo "Usage: IMAGE=/path/to/first_frame.png bash ${BASH_SOURCE[0]}" >&2
  echo "   or: bash ${BASH_SOURCE[0]} /path/to/first_frame.png" >&2
  exit 2
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"

cd "${REPO_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/image_to_video.py" \
  --model "${MODEL}" \
  --image "${IMAGE}" \
  --prompt "${PROMPT}" \
  --enable-cpu-offload \
  --tensor-parallel-size 2 \
  --ulysses-degree 4 \
  --vae-patch-parallel-size 8 \
  --vae-use-tiling \
  --height 768 \
  --width 1344 \
  --num-frames 240 \
  --num-inference-steps 50 \
  --guidance-scale 1.0 \
  --flow-shift 12.0 \
  --fps 24 \
  --output "${OUTPUT}" \
  --extra-body '{"task":"fl2va","flow_shift":12.0,"audio_flow_shift":3.0}'
