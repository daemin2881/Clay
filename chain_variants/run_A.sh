#!/bin/bash
# A 체인(목표 A, 자동 영역 10개) 실행기. 변형(VARIANT) 한 줄만 브랜치마다 다르다: git diff 로 변형 차이를 바로 볼 수 있다.
#   J = 반사실 없음, J 로만 확정      (2026-11-17 A/B 결과 최선: A +15.1%, 하트 +16.4%, 단계 시간 1x)   <- main 작업 트리에서 쓰는 방법
#   C = T 와 같은 반사실, 확정은 J 로만 (A +10.0%, 하트 +16.4%, 단계 시간 약 3x)                       <- 브랜치 variant/C-commit-by-j
#   T = 원래 방법(반사실 + T 추적 확정)  (A +7.7%, 하트 +5.5%, 단계 시간 약 3x)                        <- 브랜치 variant/T-original
VARIANT=C
VARIANT_FLAGS="--commit-by-j"
# 사용: bash chain_variants/run_A.sh   (USE_MPS=0 이면 MPS 없이, VIEWER=0 이면 뷰어 없이)
cd "$(dirname "$0")/.."
PY=${PY:-/home/airlab/research/clay_genesis/.venv/bin/python}
if [ "${USE_MPS:-1}" = 1 ] && [ -d results/mps_smoke/pipe ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$PWD/results/mps_smoke/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/results/mps_smoke/log
fi
[ "${VIEWER:-1}" = 1 ] && { export DISPLAY=${DISPLAY:-:1}; VIEWER_FLAG=--viewer; } || VIEWER_FLAG=
echo "[$(date +%F' '%T)] A 체인 시작: 변형 $VARIANT ($VARIANT_FLAGS)"
exec $PY -u cf_chain.py --active all --rotate --repeat --skip-stuck $VIEWER_FLAG \
  --regions results/case_A/regions_A_auto_v1_raw --out results/case_A/chain_A_rot \
  --cond E5 --rounds 200 --beam 1 --seed 1 --stop-mae 1.0 --patience 8 \
  --extra="--plan-refine 40 --plan-seq 3 --long-screen model --screen-verify 3 --tree-budget-min 20 --seq-budget-min 45 --t-jband 0.1 --ctrl-valid-len 79 --screen-budget-actions 160 --min-commit-gain 0.01 $VARIANT_FLAGS"
