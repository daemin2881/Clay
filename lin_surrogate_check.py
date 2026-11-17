"""선형 근사(기록된 높이 변화로 '그 행동을 다르게 했다면'을 추정)가 Genesis 재생 결과와 얼마나 맞는지, 과거로 얼마나 멀리까지 맞는지 잰다.

  C_gen = H_변형 - H_control        (Genesis 재생 캐시: s<시작>_control.npy, s<시작>_<변형>.npy; cf_compensate 가 만든다)
  C_lin = 배율 x dH_t               (dH_t = 그 행동이 기록한 높이 변화. skip -1, 깊이 x0.5 -> -0.5, 깊이 x1.5 -> +0.5)
행동 접촉영역을 도구 반경 2배 넓힌 창 안에서 상관계수(Pearson)와 기울기(C_gen ~= k C_lin)를 낸다.
이 근사로 재생 후보를 거르는(싼 선별 -> Genesis 확인) 방식은 근사가 멀리 있는 행동에서도 맞을 때만 믿을 수 있다. 근사는 선별 전용이며 반사실 결과로 쓰지 않는다.

사용: python lin_surrogate_check.py --history-dir H --upto K --cache H/../cf_replay_cache/K<K>
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy import ndimage

import cf_common as cc
import first_beam as fb

FAC = {"skip": -1.0, "l1_d0.5": -0.5, "l1_d1.5": 0.5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out-view", type=Path, required=True, help="읽기 전용 이력 보기를 만들 새 폴더(임시)")
    a = ap.parse_args()
    view = cc.make_view(a.history_dir, a.upto, a.out_view)
    g = cc.geometry(view)
    _, actions, before, after = fb.load_history(view)
    K = len(actions)
    cache = a.cache or (a.history_dir.resolve().parent / "cf_replay_cache" / f"K{K}")
    rows, stats = [], []
    for fc in sorted(cache.glob("s*_control.npy")):
        s0 = int(re.match(r"s(\d+)_control", fc.name).group(1))
        hc = np.load(fc).astype(np.float64)
        win = ndimage.binary_dilation(fb.footprint(g, actions[s0 - 1]), iterations=int(round(2 * g["tool_r_px"])))
        dH = (after[s0 - 1] - before[s0 - 1]).astype(np.float64)
        for vn, fac in FAC.items():
            fv = cache / f"s{s0}_{vn}.npy"
            if not fv.exists():
                continue
            Cg = (np.load(fv).astype(np.float64) - hc)[win]
            Cl = (fac * dH)[win]
            if Cg.std() < 1e-9 or Cl.std() < 1e-9:
                continue
            corr = float(np.corrcoef(Cg, Cl)[0, 1])
            k = float((Cg * Cl).sum() / (Cl * Cl).sum())
            rows.append((K - s0 + 1, s0, vn, corr, k, float(np.sqrt((Cg ** 2).mean()) * 1000), float(np.sqrt((Cl ** 2).mean()) * 1000)))
            stats.append(dict(L=K - s0 + 1, s0=s0, v=vn, n=Cg.size, sgg=float((Cg ** 2).sum()), sxy=float((Cg * Cl).sum()), sxx=float((Cl ** 2).sum())))
    rows.sort()
    print(f"이력 {a.history_dir.name} 시점 {K} | 캐시 {cache} | 창 = 접촉영역 +- 도구 반경 2배")
    print(f"{'재생 길이':>8} {'시작':>5} {'변형':<8} {'상관':>6} {'기울기 k':>9} {'Genesis rms':>12} {'선형 rms':>9}   (k=1 이면 크기까지 일치)")
    for L, s0, vn, c, k, rg, rl in rows:
        print(f"{L:>8} {'#' + str(s0):>5} {vn:<8} {c:>+6.2f} {k:>9.2f} {rg:>10.3f}mm {rl:>7.3f}mm")
    if rows:
        for lo, hi in ((0, 20), (21, 50), (51, 200)):
            sel = [r for r in rows if lo <= r[0] <= hi]
            if sel:
                print(f"재생 길이 {lo:>3}~{hi:<3}: {len(sel):2d}개 | 상관 평균 {np.mean([r[3] for r in sel]):+.2f} (최소 {min(r[3] for r in sel):+.2f}) | 기울기 평균 {np.mean([r[4] for r in sel]):.2f}")
            else:
                print(f"재생 길이 {lo:>3}~{hi:<3}: 아직 없음")
        regress(stats)


def regress(stats):
    """회귀 보정: 변형마다 이득 g_v 를 둬서 C_gen ~= g_v * (배율 x dH_t) 로 맞춘다(원점 지나는 최소제곱).
    g_v=1 이 고정 배율 선형 근사. 시작 행동 하나를 통째로 빼고 나머지로 맞춘 g_v 로 그 행동을 예측해(leave-one-start-out) 보정 전/후 오차(RMSE)를 비교한다.
    표본이 시작 행동 몇 개뿐이라 매개변수 하나(g_v)만 쓴다."""
    print("\n[회귀 보정] 변형별 이득 g_v (C_gen ~= g_v x 선형 근사), leave-one-start-out 검증")
    starts = sorted({s["s0"] for s in stats})
    for vn in FAC:
        sv = [s for s in stats if s["v"] == vn]
        if len({s["s0"] for s in sv}) < 3:
            print(f"  {vn:<8}: 시작 행동 {len({s['s0'] for s in sv})}개뿐이라 건너뜀")
            continue
        g_all = sum(s["sxy"] for s in sv) / sum(s["sxx"] for s in sv)
        e_lin, e_fit, n_tot = 0.0, 0.0, 0
        for h in sorted({s["s0"] for s in sv}):
            tr = [s for s in sv if s["s0"] != h]
            te = [s for s in sv if s["s0"] == h][0]
            g = sum(s["sxy"] for s in tr) / sum(s["sxx"] for s in tr)
            e_lin += te["sgg"] - 2 * te["sxy"] + te["sxx"]
            e_fit += te["sgg"] - 2 * g * te["sxy"] + g * g * te["sxx"]
            n_tot += te["n"]
        r_lin, r_fit = np.sqrt(e_lin / n_tot) * 1000, np.sqrt(e_fit / n_tot) * 1000
        print(f"  {vn:<8}: 이득 g={g_all:.2f} | 예측 오차 RMSE 고정배율 {r_lin:.3f}mm -> 회귀 보정 {r_fit:.3f}mm ({100 * (1 - r_fit / r_lin):+.0f}%) | 시작 행동 {len({s['s0'] for s in sv})}개")


if __name__ == "__main__":
    main()
