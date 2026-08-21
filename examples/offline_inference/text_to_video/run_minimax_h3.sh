#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# MiniMax H3's 720p/768P profile is 1344x768 and the model always generates at
# 24 FPS. This requests 240 frames (~10 seconds). CP4 is represented by the
# repository's Ulysses/context-parallel degree. Override MODEL, PROMPT, OUTPUT,
# or PYTHON from the environment when needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODEL="${MODEL:-MiniMaxAI/MiniMax-H3}"
DEFAULT_PROMPT="EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCASTThe shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. The light is warm, early sun reflecting off the camera lens. The faint hum of chatter and distant drilling fills the air. The reporter, composed but visibly excited, looks directly into the camera, microphone in hand. Reporter (live): \"Thank you, Sylvia. And yes — this is a sentence I never thought I'd say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!\" He gestures slightly toward the field behind him. Reporter (grinning): \"If my cameraman can pan over, you'll see what all the excitement's about.\" The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening in the morning light. The camera shakes slightly, trying to stay focused through the chaos. Reporter (off-screen, shouting over the noise): \"There it is, folks — the moment New Castle will never forget!\" The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the entire scene — the small-town skyline silhouetted against the wild fountain of oil."
PROMPT="${PROMPT:-${DEFAULT_PROMPT}}"
OUTPUT="${OUTPUT:-minimax_h3_t2va_720p.mp4}"
PYTHON="${PYTHON:-python}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"

cd "${REPO_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/text_to_video.py" \
  --model "${MODEL}" \
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
  --extra-body '{"task":"t2va","aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'
