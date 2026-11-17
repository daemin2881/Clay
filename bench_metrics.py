"""조형 결과를 타깃 간에 공정하게 비교하는 지표 (읽기 전용, numpy만 사용).

전체 MAE는 부조 영역이 작으면 둔감하고, 시작 오차가 큰 타깃(cat)과 작은 타깃(A)을 비교할 수 없다.
그래서 모든 값을 '아무것도 안 했을 때(h0)의 오차'로 나눈다. 1.0 = 변화 없음, 0 = 완벽, 1 초과 = 악화.

  relief_ratio : 부조 영역(footprint) 안 MAE / h0의 같은 값
  bg_ratio     : 부조 밖(배경) MAE / h0의 같은 값  (배경을 망가뜨리지 않았는지)
  global_ratio : 전체 MAE / h0의 같은 값
  progress_cos : 부조 안에서 (h-h0)와 (target-h0)의 코사인 유사도 (변화 방향이 맞는지, 크기 무관)
  relief_gain  : 1 - relief_ratio  (클수록 좋음)

사용: python3 -I bench_metrics.py HISTORY_DIR [HISTORY_DIR ...] [--every 10]
"""
import argparse
import re
from pathlib import Path

import numpy as np

SIZE_THRESH_M = 5e-4   # 부조 영역 = 타깃이 배경 중앙값보다 0.5mm 이상 높은 곳


def border_median(a):
    return float(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))


def metrics(h, h0, t, fp):
    def m(x, mask):
        return float(np.abs(x - t)[mask].mean())
    bg = ~fp
    r0, b0, g0 = m(h0, fp), m(h0, bg), float(np.abs(h0 - t).mean())
    r, b, g = m(h, fp), m(h, bg), float(np.abs(h - t).mean())
    dh, dt = (h - h0)[fp], (t - h0)[fp]
    cos = float((dh * dt).sum() / (np.linalg.norm(dh) * np.linalg.norm(dt) + 1e-12))
    return dict(relief_ratio=r / r0, bg_ratio=b / b0, global_ratio=g / g0,
                progress_cos=cos, relief_gain=1 - r / r0, relief_mae_mm=r * 1000, relief0_mae_mm=r0 * 1000)


def load(root):
    dirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and re.fullmatch(r"action_\d{6}", p.name))
    t = np.load(Path(root) / "target_used_512_m.npy").astype(np.float64)
    return dirs, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=0, help="N행동마다 중간 값도 출력")
    a = ap.parse_args()
    print(f"{'이력':<46}{'행동':>5}{'부조비':>8}{'배경비':>8}{'전체비':>8}{'cos':>7}{'부조MAE':>9}{'h0':>8}")
    for root in a.history:
        dirs, t = load(root)
        fp = t > border_median(t) + SIZE_THRESH_M
        h0 = np.load(dirs[0] / "transition.npz")["before_m"].astype(np.float64)
        name = root.parent.name if root.name == "history" else root.name
        picks = sorted(set(list(range(a.every, len(dirs), a.every)) + [len(dirs)])) if a.every else [len(dirs)]
        for n in picks:
            h = np.load(dirs[n - 1] / "transition.npz")["after_m"].astype(np.float64)
            r = metrics(h, h0, t, fp)
            print(f"{name[:44]:<46}{n:>5}{r['relief_ratio']:>8.3f}{r['bg_ratio']:>8.3f}{r['global_ratio']:>8.3f}"
                  f"{r['progress_cos']:>7.3f}{r['relief_mae_mm']:>8.3f}m{r['relief0_mae_mm']:>7.3f}m")
        print(f"  (부조 영역 {fp.mean():.1%}, 타깃 최대 부조 {(t.max() - border_median(t)) * 1000:.1f}mm)")


if __name__ == "__main__":
    main()
