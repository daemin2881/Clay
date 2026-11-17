#!/bin/bash
# 하트 체인(뒤집힌 하트 목표: 두 덩이 위로, 끝 아래로, 영역 9개) 실행기. 변형 한 줄만 브랜치마다 다르다(run_A.sh 머리말 참조).
VARIANT=C
VARIANT_FLAGS="--commit-by-j"
# 사용: bash chain_variants/run_heart.sh   (USE_MPS=0 이면 MPS 없이)
cd "$(dirname "$0")/.."
PY=${PY:-/home/airlab/research/clay_genesis/.venv/bin/python}
if [ "${USE_MPS:-1}" = 1 ] && [ -d results/mps_smoke/pipe ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$PWD/results/mps_smoke/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/results/mps_smoke/log
fi
echo "[$(date +%F' '%T)] 하트 체인 시작: 변형 $VARIANT ($VARIANT_FLAGS)"
exec $PY -u cf_chain.py --active all --rotate --repeat --skip-stuck \
  --history-dir results/heart_history190_20261115_203518/history --upto 190 \
  --regions results/case_heart/regions_flipped_k190 --target-npy results/case_heart/heart_target_raw_flipud_512_m.npy \
  --out results/case_heart/chain_heart_flip \
  --cond E5 --rounds 200 --beam 1 --seed 1 --stop-mae 1.0 --patience 8 \
  --extra="--plan-refine 40 --plan-seq 3 --long-screen model --screen-verify 3 --tree-budget-min 20 --seq-budget-min 45 --t-jband 0.1 --screen-budget-actions 160 --screen-max-len 100 --ctrl-valid-len 190 --min-commit-gain 0.01 $VARIANT_FLAGS"
