"""총 행동 수에 따른 요청 영역 점수: 보상 없음(원래 이력, 이어 조형 A1/A2) 대 보상 있음(보상 확정 행동 + 이어 조형 B1/B2, v11).
같은 총 행동 수에서 비교한다. 점수 e = 5개 영역(형태+기울기+테두리 오차를 80번째 값으로 나눈 비율)의 평균, 80번째 = 3.000, 낮을수록 좋음.
보상 확정 행동은 1개당 1행동, 2-step(PAIR)은 2행동으로 센다.

사용: python plot_compare.py [--out results/case_A/compare_actions.png]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cf_common as cc
import cf_compensate as cfc
from region_set import RegionSet, border_median

ROOT = Path("results/A_history200_20261114_060837/history")
CASE = Path("results/case_A")
BLUE, ORANGE = "#2a78d6", "#eb6834"      # dataviz 기준 팔레트의 1번(파랑), 2번(주황)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CASE / "compare_actions.png")
    ap.add_argument("--orig-upto", type=int, default=120)
    a = ap.parse_args()
    R = cc.geometry(ROOT)["tool_r_px"]
    sigma = cc.band_sigma(R, 0.4)
    S = lambda k: np.load(ROOT / f"action_{k:06d}" / "transition.npz")["after_m"].astype(float)
    h80 = S(80)
    rs = RegionSet.load(CASE / "cf_edit_A80_v4_mdl/prompt", ref_default=h80)
    cfg0 = cfc.build_cfg(rs, h80, 1.0, int(round(0.94 * R)), int(round(1.17 * R)), R)
    norm, names, bm80 = cfg0["norm"], rs.names, border_median(h80)

    def score(h):
        sh = border_median(h) - bm80
        pre = cc.appearance_pre(rs.masks, {n: rs.targets[n] + sh for n in names}, sigma)
        v = cc.appearance(h, pre, R, sigma)
        per = {n: v[n][2] / norm[n][0] + v[n][0] / norm[n][1] + v[n][1] / norm[n][2] for n in names}
        return float(np.mean(list(per.values()))), per

    series = {}      # 이름 -> (행동 수 리스트, 점수 리스트, 영역별 점수 리스트)

    def add(name, ks, hs):
        sc = [score(h) for h in hs]
        series[name] = (list(ks), [s[0] for s in sc], [s[1] for s in sc])

    ks = list(range(80, a.orig_upto + 1))
    add("orig", ks, [S(k) for k in ks])
    for arm in ("A1", "A2"):
        d = CASE / f"ab_{arm}"
        add(arm, [80 + j for j in range(21)], [np.load(d / "start_m.npy").astype(float)] + [np.load(d / "states" / f"step_{j:03d}.npy").astype(float) for j in range(1, 21)])
    # 보상 확정 경로: v8 3개(1행동씩) -> v10 PAIR(2행동) -> v11 (있으면)
    v8 = CASE / "cf_edit_A80_v8/cf"
    comp_k, comp_h = [80], [h80]
    for i, f in ((81, v8 / "states/iter_001.npy"), (82, v8 / "states/iter_002.npy"), (83, v8 / "final_m.npy")):
        if f.exists():
            comp_k.append(i)
            comp_h.append(np.load(f).astype(float))
    v10 = CASE / "cf_edit_A80_v10_cont/cf/states/iter_001.npy"
    comp_k.append(85)
    comp_h.append(np.load(v10).astype(float))
    k_now = 85
    v11 = CASE / "cf_edit_A80_v11_prompt"
    log = CASE / "cf_edit_A80_v11_prompt.log"
    n11 = 0
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            m = re.match(r"\s+확정 (\d+):", line)
            if m:
                n11 = int(m.group(1))
                k_now += 2 if " + x=" in line else 1
                f = v11 / "cf/states" / f"iter_{n11:03d}.npy"
                if f.exists():
                    comp_k.append(k_now)
                    comp_h.append(np.load(f).astype(float))
    add("comp", comp_k, comp_h)
    for arm in ("B1", "B2"):
        d = CASE / f"ab_{arm}"
        add(arm, [85 + j for j in range(21)], [np.load(d / "start_m.npy").astype(float)] + [np.load(d / "states" / f"step_{j:03d}.npy").astype(float) for j in range(1, 21)])

    # ---- 그림: 위 전체 점수, 아래 영역별
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(11.5, 7.8), facecolor=SURF)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1], hspace=0.5, wspace=0.28)
    panels = [("Region score e (mean of 5 requested regions)", None, fig.add_subplot(gs[0, :]))]
    reg_names = ["groove_top", "groove_bottom", "left_leg", "left_foot", "right_foot"]
    for i, n in enumerate(reg_names):
        panels.append((n, n, fig.add_subplot(gs[1 + i // 3, i % 3])))

    def style(ax, title):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.tick_params(labelsize=8)

    for title, key, ax in panels:
        style(ax, title)
        big = key is None
        def y_of(name):
            ks_, e_, per_ = series[name]
            return e_ if big else [p[key] for p in per_]
        # 보상 없음: 원래 이력(가는 선), A1/A2
        ax.plot(series["orig"][0], y_of("orig"), color=BLUE, lw=1.0, alpha=0.45, zorder=2)
        for arm in ("A1", "A2"):
            ax.plot(series[arm][0], y_of(arm), color=BLUE, lw=1.6, zorder=3)
        # 보상 있음: 확정 경로(마커)와 이어 조형 B1/B2
        for arm in ("B1", "B2"):
            ax.plot(series[arm][0], y_of(arm), color=ORANGE, lw=1.6, zorder=3)
        ax.plot(series["comp"][0], y_of("comp"), color=ORANGE, lw=1.6, marker="o", ms=4.5, mec=SURF, mew=1.0, zorder=4)
        ax.axhline(3.0 if big else 3.0, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(78, max(a.orig_upto, 106) + 1)
        if big:
            ax.set_ylabel("score (80 = 3.00, lower is better)", fontsize=8.5)
            ax.set_xlabel("total number of actions", fontsize=8.5)
            ax.text(a.orig_upto + 0.8, series["orig"][1][-1], "no compensation\n(original history)", color=BLUE, fontsize=8.5, va="center")
            yb = series["B2"][1][-1]
            ax.text(106.3, yb, "compensation\n+ continued", color=ORANGE, fontsize=8.5, va="center")
            ya = np.mean([series["A1"][1][-1], series["A2"][1][-1]])
            ax.text(100.8, ya + 0.12, "no comp. + continued", color=BLUE, fontsize=8, va="bottom")
        ax.set_xlim(78, a.orig_upto + 12 if big else 108)
    fig.suptitle("Requested-region score vs total actions: with vs without local compensation (A, state 80)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Orange dots: compensation actions (1 per push, 2-step = 2); orange lines after 85: continued sculpting from the compensated state (2 runs). "
                          "Blue: no compensation (thin = original recorded history, bold = 2 re-runs from state 80). The dashed line is the start (3.00).",
             ha="left", va="bottom", fontsize=7.8, color=MUTED, wrap=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, facecolor=SURF, bbox_inches="tight")
    print("saved", a.out, "| 보상 경로 행동 수", series["comp"][0], "점수", [round(x, 3) for x in series["comp"][1]])
    for n in ("orig", "A1", "A2", "B1", "B2"):
        ks_, e_, _ = series[n]
        pick = [(k, round(e, 3)) for k, e in zip(ks_, e_) if k in (80, 85, 90, 95, 100, 105)]
        print(n, pick)


if __name__ == "__main__":
    main()
