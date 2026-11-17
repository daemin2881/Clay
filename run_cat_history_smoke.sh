#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo '사용법: bash run_cat_history_smoke.sh 체크포인트경로 [--viewer]'
  exit 2
fi
CHECKPOINT_PATH="$1"
shift
PROJECT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_PATH"
BASE_MAP_PATH="../marigold-v2/output/final_cat_height/cat_height_512_m.npy"
EAR_MASK_PATH="../sam2/results/bear_ear_polyline_mask/left_ear_mask.png"
for INPUT_PATH in "$CHECKPOINT_PATH" "$BASE_MAP_PATH" "$EAR_MASK_PATH"; do
  if [ ! -f "$INPUT_PATH" ]; then
    echo "입력 파일을 찾을 수 없습니다: $INPUT_PATH"
    exit 2
  fi
done
RUN_TAG="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
RUN_ROOT="results/cat_history_smoke_${RUN_TAG}"
mkdir -p "$RUN_ROOT"
export VS_MATERIAL=medium
export VS_TIP_CORRECTION=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python visual_sculpting.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --target CAT_BASE \
  --target-heightmap "$BASE_MAP_PATH" \
  --roi-mask "$EAR_MASK_PATH" \
  --output-dir "$RUN_ROOT/sculpt" \
  --history-dir "$RUN_ROOT/history" \
  --max-actions 190 --stop-after-actions 5 --force-full-actions \
  --plan-horizon 40 --mpc-execute 5 \
  --candidates 64 --candidate-batch-size 1 \
  --refine-steps 0 --save-every 5 \
  "$@" 2>&1 | tee "$RUN_ROOT/sculpt.log"
python validate_history_replay.py \
  --history-dir "$RUN_ROOT/history" --starts all \
  --sequence-length 1 --max-drift-mm 0.5 \
  2>&1 | tee "$RUN_ROOT/replay_single.log"
python validate_history_replay.py \
  --history-dir "$RUN_ROOT/history" --starts 1 \
  --sequence-length 5 --max-drift-mm 0.5 \
  2>&1 | tee "$RUN_ROOT/replay_sequence5.log"
echo "완료: $RUN_ROOT"
