#!/usr/bin/env python3
"""MPS 로 돌린 재생이 MPS 없이 돌린 재생과 같은 결과인지 확인한다(하트 이력의 행동 #S 부터 N개를 재생, 기록과 비교).
사용(각각 따로 실행하고 마지막에 비교):
  python mps_equal_check.py run --tag nompi            # MPS 환경변수 없이
  CUDA_MPS_PIPE_DIRECTORY=... CUDA_MPS_LOG_DIRECTORY=... python mps_equal_check.py run --tag mps
  python mps_equal_check.py compare --tags nompi,mps
판정: 두 결과의 평균 차이가 정상 재생 잡음(약 0.005 mm) 안이고, 기록 대비 어긋남이 두 쪽에서 같은 수준이면 같다고 본다.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ab_cf_vs_j" / "mps_equal"
HIST = ROOT / "results/heart_history190_20261115_203518/history"


def run(a):
    det = json.loads((HIST / "manifest.json").read_text())["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import vs_simulator as sim
    from vs_heightmap_fast import apply_fast_height
    apply_fast_height(sim)
    from vs_history import GenesisStateAdapter
    import counterfactual_replay as cfr
    ad = GenesisStateAdapter(sim)
    S, E = a.start, a.start + a.n - 1
    pre, records, rngs = cfr.load_plan(HIST, S)
    t0 = time.perf_counter()
    h, _ = cfr.run_replay(ad, pre, records, rngs, S, E, {})
    dt = time.perf_counter() - t0
    rec = np.load(HIST / f"action_{E:06d}" / "transition.npz")["after_m"].astype(np.float64)
    d = np.abs(np.asarray(h, np.float64) - rec) * 1000
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{a.tag}.npy", np.asarray(h, np.float32))
    mps = "MPS" if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") else "MPS 없음"
    print(f"{a.tag} ({mps}): #{S}~#{E} {a.n}행동 {dt:.1f}초 | 기록 대비 평균 {d.mean():.4f} mm, p99.9 {np.quantile(d, 0.999):.3f}, 최대 {d.max():.3f}", flush=True)


def compare(a):
    t1, t2 = a.tags.split(",")
    h1, h2 = np.load(OUT / f"{t1}.npy").astype(np.float64), np.load(OUT / f"{t2}.npy").astype(np.float64)
    d = np.abs(h1 - h2) * 1000
    ok = float(d.mean()) <= 0.005
    print(f"{t1} vs {t2}: 평균 차이 {d.mean():.4f} mm, p99.9 {np.quantile(d, 0.999):.3f}, 최대 {d.max():.3f}, 다른 화소(>0.01mm) {int((d > 0.01).sum())}개 -> {'같다(정상 재생 잡음 안)' if ok else '다르다: MPS 를 믿지 않는다'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--start", type=int, default=182)
    r.add_argument("--n", type=int, default=5)
    c = sub.add_parser("compare")
    c.add_argument("--tags", required=True)
    a = ap.parse_args()
    (run if a.cmd == "run" else compare)(a)


if __name__ == "__main__":
    main()
