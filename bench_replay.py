#!/usr/bin/env python3
"""MPS 속도 측정용: 하트 이력의 행동 #S 부터 N개를 재생하는 시간만 잰다(결과 파일 없음, 창 없음, 빠른 높이맵 적용). 같은 스크립트를 여러 개 동시에 돌려서 MPS 켠/끈 처리량을 비교한다."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
S, N = int(sys.argv[1]), int(sys.argv[2])
root = ROOT / "results/heart_history190_20261115_203518/history"
det = json.loads((root / "manifest.json").read_text())["signature"]["details"]
os.environ["VS_MATERIAL"] = det["material"]; os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"; os.environ["VS_SHOW_VIEWER"] = "0"
import vs_simulator as sim
from vs_heightmap_fast import apply_fast_height; apply_fast_height(sim)
from vs_history import GenesisStateAdapter
import counterfactual_replay as cfr
ad = GenesisStateAdapter(sim)
pre, records, rngs = cfr.load_plan(root, S)
t0 = time.perf_counter(); cfr.run_replay(ad, pre, records, rngs, S, S + N - 1, {}); dt = time.perf_counter() - t0
print(f"BENCH pid={os.getpid()} {N}행동 {dt:.1f}초 ({dt / N:.1f}초/행동)", flush=True)
