#!/usr/bin/env bash
set -euo pipefail

# Re-generate the final analytics video (GPU: Apple MPS)
if [[ -f /Users/zhujinpeng/Desktop/yumaoqiu/.venv/bin/python ]]; then
  PY_BIN=/Users/zhujinpeng/Desktop/yumaoqiu/.venv/bin/python
else
  PY_BIN=python3
fi

"${PY_BIN}" /Users/zhujinpeng/Desktop/yumaoqiu/pipeline_code/overlay_player_analytics.py \
  --video_path /Users/zhujinpeng/Desktop/yumaoqiu/output_tracknet/41.mp4 \
  --output_path /Users/zhujinpeng/Desktop/yumaoqiu/output_final/41_analysis.mp4 \
  --ball_csv /Users/zhujinpeng/Desktop/yumaoqiu/output_tracknet/41_ball.csv \
  --yolo_model '/Users/zhujinpeng/Library/Mobile Documents/com~apple~CloudDocs/pipeline_repro_bundle/weights/yolov8s-pose.pt' \
  --tracker_cfg bytetrack.yaml \
  --detect_interval 1 \
  --no_select_court_points \
  --court_points '352,232,613,232,719,525,244,525' \
  --device mps \
  --trail_jump_split_px 80 \
  --no_draw_pose
