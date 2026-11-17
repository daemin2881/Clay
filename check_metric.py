"""보상 목표 J 가 '눈으로 보는 조형 진행'과 같은 방향인지, 보상을 믿기 전에 이력으로 검증한다.

조형 이력의 여러 시점(t)에서 영역 집합에 대한 J(cf_compensate 와 같은 정의)를 계산한다. 조형이 진행될수록 목표에 가까워지므로
t 가 클수록 J 가 낮아야 한다(적어도 평평한 시작 상태보다 K 번째가 낮아야 한다). 그렇지 않은 지표를 줄이도록 보상하면 눈에 보이는 개선과 어긋난다.
--blur-tools 0 이면 흐리지 않은 원본 픽셀 지표(점토 입자 질감 포함)로 같은 검사를 한다.

사용: python check_metric.py --history-dir H --regions DIR [--names a,b] [--ts 1,20,40,80,120,190] [--blur-tools 0.4] [--ref-t 80]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

import cf_common as cc
from region_set import RegionSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--names", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--ref-t", type=int, help="J 의 기준(=1.0) 상태. 기본: 영역 폴더의 h_ref")
    ap.add_argument("--blur-tools", type=float, nargs="+", default=[0.4, 0.0], help="검사할 흐림 크기들(도구 반경 배수)")
    a = ap.parse_args()
    root = a.history_dir.resolve()
    n = cc.n_actions(root)
    ts = [int(x) for x in a.ts.split(",")] if a.ts else sorted({1, *range(10, n + 1, max(n // 10, 1)), n})
    g = cc.geometry(root)
    rs = RegionSet.load(a.regions, names=[x for x in a.names.split(",") if x] or None,
                        ref_default=np.load(root / f"action_{a.ref_t:06d}" / "transition.npz")["after_m"] if a.ref_t else np.zeros((512, 512)))
    maps = {t: np.load(root / f"action_{t:06d}" / "transition.npz")["after_m"].astype(np.float64) for t in ts}
    tgt = cc.load_target(root).astype(np.float64)
    bgm = lambda x: float(np.median(np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]])))
    t_ref = a.ref_t or ts[-1]
    h_ref = np.load(root / f"action_{t_ref:06d}" / "transition.npz")["after_m"].astype(np.float64)
    print("시점별 비교는 '순수 목표'(목표를 각 시점 배경 높이에 맞춘 것)를 모든 영역에 쓴다. 영역 목표(match_target)는 기준 상태의 영역 밖 내용을 포함해서 기준 시점에 유리하다.")
    from scipy import ndimage
    for bt in a.blur_tools:
        sigma = cc.band_sigma(g["tool_r_px"], bt)

        def comp(h):
            goal = tgt - bgm(tgt) + bgm(h)
            pre = cc.appearance_pre(rs.masks, {nm: goal for nm in rs.names}, sigma)
            return cc.appearance(h, pre, g["tool_r_px"], sigma)
        v0 = comp(h_ref)
        norm = {nm: (max(v0[nm][2], 0.1), max(v0[nm][0], 0.1), max(v0[nm][1], 0.1)) for nm in rs.names}
        res = {}
        for t in ts:
            v = comp(maps[t])
            c3 = [np.mean([v[nm][i] / norm[nm][j] for nm in rs.names]) for i, j in ((2, 0), (0, 1), (1, 2))]
            res[t] = (sum(c3), *c3)
        print(f"\n== 흐림 {bt:g} x 도구 반경 ({sigma:.1f}px) | 기준 시점 {t_ref} 의 J = 3.00 | 영역 {rs.names} ==")
        print(f"{'t':>5}{'J':>8}{'형태':>8}{'기울기':>8}{'테두리':>8}   (영역 평균, 기준 대비 비율)")
        for t in ts:
            print(f"{t:>5}{res[t][0]:>8.3f}{res[t][1]:>8.2f}{res[t][2]:>8.2f}{res[t][3]:>8.2f}")
        r = stats.spearmanr(ts, [res[t][0] for t in ts])[0]
        best = min(ts, key=lambda t: res[t][0])
        print(f"스피어만(t, J) = {r:+.2f}  | J 가 가장 낮은 시점: {best}  | 평평한 시작({ts[0]}) J {res[ts[0]][0]:.2f} -> 마지막({ts[-1]}) J {res[ts[-1]][0]:.2f}")


if __name__ == "__main__":
    main()
