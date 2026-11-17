"""rollback_cf.py 결과(H_control, H_cf*)를 다른 팔과 같은 형식(start_m.npy, final_m.npy)으로 바꾼다.

'과거 행동 하나를 다르게 했더라면' 지금 상태가 어떻게 달랐을지를 현재 상태 보정 팔과 같은 눈금으로 비교하기 위해서다.
  start = 기록된 최종 상태 H_T
  final = H_T + (H_cf - H_control)       (같은 엔진에서 다시 돌린 control 을 빼서 재생 드리프트를 상쇄)
사용: python cf_as_arm.py --rollback-dir results/.../rollback29 --history-dir H --out-dir results/.../cf_arm
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback-dir", required=True, type=Path)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    last = len([d for d in a.history_dir.glob("action_??????") if d.is_dir()])
    h_t = np.load(a.history_dir / f"action_{last:06d}" / "transition.npz")["after_m"].astype(np.float32)
    h_ctrl = np.load(a.rollback_dir / "H_control.npy")
    dd = np.abs(h_ctrl - h_t) * 1000
    print(f"control 재생 vs 기록: 최대 {dd.max():.3f}mm, 0.5mm 초과 {(dd > 0.5).sum()}화소, p99.9 {np.quantile(dd, 0.999):.4f}mm")
    for f in sorted(a.rollback_dir.glob("H_*.npy")):
        name = f.stem[2:]
        if name in ("control", "replicate"):
            continue
        out = a.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "start_m.npy", h_t)
        np.save(out / "final_m.npy", (h_t + (np.load(f) - h_ctrl)).astype(np.float32))
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
