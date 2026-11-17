"""과거 힌트 보상 vs 기존 베이스 프로그램(계획기) 재실행 비교: 같은 출발 상태, 같은 새 목표, 같은 점수로 '누른 횟수'당 결과를 잰다.

  기준선(base): continue_sculpt.py 로 이력 K 번째 상태에서 계획기를 그대로 다시 돌린 것 (시드 여러 개). 단계마다 states/step_XXX.npy.
  과거 힌트(hint): cf_compensate 의 확정 상태 states/iter_XXX.npy. 확정 하나가 푸시 1개(단일) 또는 여러 개(2-step PAIR, PLANSEQ)이므로 rounds.json 의 action 길이로 누른 횟수를 센다.
  점수: 영역 점수 J (형태/기울기/테두리 + 구덩이 꼬리, K 번째 상태 = 3.00, 낮을수록 좋음) — 요청으로 만든 새 목표 기준. 원래 목표는 보지 않는다.
  함께 보는 것: 영역 평균 높이 변화(목표 진행률), 영역 최저점, 영역 주변 고리 평균 변화(부수 변화), 영역 밖 최대 변화.
  기준선은 시드 평균±표준편차, 과거 힌트는 실행 1회(반복이 없어 분산을 모른다).
  또한 베이스 프로그램은 학습 모델로 앞을 내다보고 한 번에 5행동을 실행하지만, 과거 힌트 방식은 후보를 Genesis 로 실행해 보고 되돌려서(시뮬레이터에서만 가능한 검증) 고른다: 계산 비용(후보 평가 수)도 같이 보고한다.

사용: python cmp_hint_vs_base.py --history-dir H --upto 80 --regions DIR --names groove_top --base D/base_s1 D/base_s2 ... --hint results/case_A/cf_edit_A80_v22_hint/cf [--hint ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import cf_common as cc
import cf_compensate as cfc
from region_set import RegionSet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--names", default="")
    ap.add_argument("--base", nargs="*", default=[], type=Path, help="continue_sculpt 출력 폴더들 (시드별)")
    ap.add_argument("--hint", nargs="*", default=[], type=Path, help="cf_compensate 출력 폴더들(cf/) (이름=경로 로 라벨 지정 가능)")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--fig", type=Path, help="눈으로 비교하는 그림(PNG)을 저장: 시작 / 기준선 / 과거 힌트의 높이맵·음영·목표와의 차이")
    ap.add_argument("--fig-pushes", type=int, help="그림에서 비교할 누른 횟수(기본: 과거 힌트의 마지막 확정 횟수)")
    a = ap.parse_args()
    root = a.history_dir.resolve()
    K = a.upto or cc.n_actions(root)
    h_K = np.load(root / f"action_{K:06d}" / "transition.npz")["after_m"].astype(np.float64)
    view = cc.make_view(root, K, Path("/tmp") / f"cmp_view_{root.parent.name}")
    g = cc.geometry(view)
    R = g["tool_r_px"]
    rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_default=h_K)
    cfg = cfc.build_cfg(rs, h_K, 0.0, int(round(0.94 * R)), int(round(1.17 * R)), R, w_tail=1.0)
    n0 = rs.names[0]
    m = rs.masks[n0]
    union = cfg["union"]
    ring = ndimage.binary_dilation(union, iterations=int(round(0.94 * R))) & ~union
    goal_move = float((rs.targets[n0] - h_K)[m].mean())

    def stats(h):
        s = cfc.score(h, cfg, h_K)
        d = h - h_K
        return dict(J=s["J"], region=cfc.region_J(n0, s["vis"], cfg), mean_mm=float(d[m].mean() * 1000), prog=float(d[m].mean() / goal_move),
                    std_mm=float((d[m] - (rs.targets[n0] - h_K)[m]).std() * 1000),
                    min_mm=float(d[m].min() * 1000), ring_mm=float(np.abs(d)[ring].mean() * 1000), out_max_mm=float(np.abs(d)[~ring & ~union].max() * 1000))

    res = {"base": {}, "hint": {}}
    maps = {"base": {}, "hint": {}}
    for d in a.base:
        for p in sorted((d / "states").glob("step_*.npy")):
            k = int(p.stem.split("_")[1])
            hh = np.load(p).astype(np.float64)
            res["base"].setdefault(k, []).append(stats(hh))
            maps["base"].setdefault(k, []).append(hh)
    for d in a.hint:
        rounds = [r for r in json.loads((d / "rounds.json").read_text()) if r.get("committed")]
        pushes, cum = {}, 0
        for i, r in enumerate(rounds, 1):
            act = r["action"]
            cum += len(act) if isinstance(act[0], (list, tuple)) else 1
            pushes[i] = cum
        summ = json.loads((d / "summary.json").read_text())
        cache_evals = sum(len(json.loads(p.read_text())) for p in sorted(d.glob("eval_round_*.json")))
        for i, cum in pushes.items():
            hh = np.load(d / "states" / f"iter_{i:03d}.npy").astype(np.float64)
            res["hint"].setdefault(cum, []).append(dict(stats(hh), family=rounds[i - 1].get("family"), evals=cache_evals, arm=d.parent.name))
            maps["hint"].setdefault(cum, []).append(hh)
    print(f"이력 {root.parent.name} K={K} 영역 {n0} | 새 목표: 영역 평균 {goal_move * 1000:+.2f} mm | 점수 J: 시작 = 3.00, 낮을수록 좋음")
    print(f"\n{'누른 횟수':>8} | {'기준선(계획기) J':>20} {'진행률':>7} {'최저점':>8} {'주변':>7} | {'과거 힌트 J':>12} {'진행률':>7} {'최저점':>8} {'주변':>7} {'출처':>8}")
    ks = sorted(set(res["base"]) | set(res["hint"]))
    for k in ks:
        b, h = res["base"].get(k), res["hint"].get(k)
        bs = ("%8.3f±%-6.3f %6.0f%% %7.1fmm %6.2fmm" % (np.mean([x["J"] for x in b]), np.std([x["J"] for x in b]), 100 * np.mean([x["prog"] for x in b]), np.mean([x["min_mm"] for x in b]), np.mean([x["ring_mm"] for x in b]))) if b else " " * 47
        hs = ("%12.3f %6.0f%% %7.1fmm %6.2fmm %8s" % (h[0]["J"], 100 * h[0]["prog"], h[0]["min_mm"], h[0]["ring_mm"], h[0]["family"])) if h else ""
        print(f"{k:>8} | {bs} | {hs}")
    print("\n(진행률 = 영역 평균 높이 변화 / 목표 변화, 주변 = 영역 주변 고리 평균 |변화|, 최저점 = 영역 안 가장 많이 낮아진 곳)")
    for d in a.hint:                      # 계산 비용: 과거 힌트 방식은 후보를 Genesis 로 실행해 보고 되돌려서 고른다(시뮬레이터에서만 가능한 검증). 실제로 누른 횟수와 별개로 보고한다.
        ev = [len(json.loads(p.read_text())) for p in sorted(d.glob("eval_round_*.json"))]
        sm = json.loads((d / "summary.json").read_text())
        print(f"계산 비용 [{d.parent.name}]: 라운드 {len(ev)}개에서 후보 {sum(ev)}개를 Genesis 로 실행해 보고 되돌림(라운드별 {ev}), 확정 {sm['commits']}회, 벽시계 {sm['minutes']:.0f}분 (기준선은 학습 모델 예측으로 계획, Genesis 되돌림 없음)")
    print("깔끔함(영역 안에서 변화가 목표 변화와 얼마나 다른가 = 고르지 않음, 표준편차 mm / 영역 밖 최대 변화 mm):")
    for k in ks:
        b, h = res["base"].get(k), res["hint"].get(k)
        print(f"  {k:>2}회  기준선 " + (f"{np.mean([x['std_mm'] for x in b]):.2f} / {np.mean([x['out_max_mm'] for x in b]):.2f}" if b else "-") + "   과거 힌트 " + (f"{h[0]['std_mm']:.2f} / {h[0]['out_max_mm']:.2f}" if h else "-"))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        kh = a.fig_pushes or (max(res["hint"]) if res["hint"] else None)
        kb = min([k for k in res["base"] if k >= kh], default=max(res["base"], default=None)) if (kh and res["base"]) else (max(res["base"]) if res["base"] else None)
        stages = [("start (state %d)" % K, h_K)]
        if kb is not None:
            stages.append((f"baseline planner, {kb} pushes (seed 1)", maps["base"][kb][0]))
        if kh is not None and kh in maps["hint"]:
            stages.append((f"past-hint compensation, {kh} push(es)", maps["hint"][kh][0]))
        rr, cc_ = np.nonzero(ndimage.binary_dilation(union, iterations=int(round(1.6 * R))))
        sl = (slice(max(rr.min() - 5, 0), rr.max() + 5), slice(max(cc_.min() - 5, 0), cc_.max() + 5))
        tgt = rs.h_star_all(h_K)
        px = g["px_mm"]
        fig, ax = plt.subplots(3, len(stages), figsize=(4.6 * len(stages), 12), squeeze=False)
        ls = LightSource(azdeg=315, altdeg=40)
        bgm = float(np.median(np.concatenate([h_K[0], h_K[-1], h_K[:, 0], h_K[:, -1]])))
        for j, (nm, hh) in enumerate(stages):
            ax[0, j].imshow(((hh - bgm) * 1000)[sl], cmap="turbo", vmin=-10, vmax=14)
            ax[0, j].set_title(nm, fontsize=9)
            ax[1, j].imshow(ls.hillshade(ndimage.gaussian_filter(hh * 1000, 1.0), vert_exag=3.0, dx=px, dy=px)[sl], cmap="gray")
            ax[2, j].imshow(((hh - tgt) * 1000)[sl], cmap="RdBu", vmin=-6, vmax=6)
            for r_ in range(3):
                ax[r_, j].contour(m[sl], levels=[0.5], colors="k", linewidths=0.8)
                ax[r_, j].set_xticks([]); ax[r_, j].set_yticks([])
        ax[0, 0].set_ylabel("height above background (mm)"); ax[1, 0].set_ylabel("shaded relief"); ax[2, 0].set_ylabel("stage - NEW target (mm)\nred=too high, blue=too low")
        fig.suptitle(f"{root.parent.name} @ {K}: request '{n0} {goal_move * 1000:+.1f} mm' (outline = region)")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=110)
        print("그림:", a.fig)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
