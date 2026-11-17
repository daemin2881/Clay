"""설계 파이프라인 결과 비교 그림 (dataviz 스킬 규칙: 팔레트 슬롯 순서 고정, 발산 색 = 파랑<->회색<->빨강, 직접 라벨).

위: 홈 주변 확대 높이 변화 (H_T 대비, 깊어짐=파랑 / 솟음=빨강). 아래: 항별 막대 (작은 다중 패널, 한 패널에 척도 하나).
사용: python make_design_figure.py --spec-dir S --arm NAME=DIR [--arm ...] --out fig.png   (첫 번째 팔이 baseline)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from edit_spec import load_spec

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # 팔레트 슬롯 1~4 (고정 순서)
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])   # 음수=솟음(빨강) 양수=깊어짐(파랑)
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])


def visual_metrics(spec, h, h_t):
    """눈에 보이는 지표: 홈 주변 창(45화소)에서의 높이오차 RMS, 목표 기울기 대비 기울기오차 RMS, 홈이 깊어진 평균, 창 밖 최대 변화."""
    from scipy import ndimage
    from visual_score import _grad
    me = spec.m_edit[0, 0].numpy()
    win = ndimage.binary_dilation(me, iterations=45)
    hs = spec.h_star[0, 0].numpy().astype(np.float64) * 1000
    h = h.astype(np.float64) * 1000
    ht = h_t.astype(np.float64) * 1000
    gx, gy = _grad(hs)
    g2 = _grad(h)
    wi = win[1:-1, 1:-1]
    gerr = np.sqrt(((g2[0] - gx) ** 2 + (g2[1] - gy) ** 2)[wi].mean())
    gref = np.sqrt((gx ** 2 + gy ** 2)[wi].mean())
    return dict(rms=float(np.sqrt(((h - hs) ** 2)[win].mean())), grad=float(100 * gerr / gref),
                deep=float((ht - h)[me].mean()), far=float(np.abs(h - ht)[~win].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vlim", type=float, default=3.0)
    a = ap.parse_args()
    spec = load_spec(a.spec_dir)
    h_t = spec.h_t[0, 0].numpy()
    h_star = spec.h_star[0, 0].numpy()
    me = spec.m_edit[0, 0].numpy()
    ys, xs = np.where(me)
    r0, r1, c0, c1 = max(ys.min() - 55, 0), min(ys.max() + 55, 512), max(xs.min() - 55, 0), min(xs.max() + 55, 512)
    arms = []
    for s in a.arm:
        n, d = s.split("=", 1)
        arms.append((n, np.load(Path(d) / "final_m.npy")))
    bg = float(np.median(np.concatenate([h_t[0], h_t[-1], h_t[:, 0], h_t[:, -1]])))

    n_img = 2 + len(arms)
    fig = plt.figure(figsize=(3.3 * n_img, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, max(n_img, 4), height_ratios=[1.0, 0.9], hspace=0.22, wspace=0.42)
    panels = [("H_T before edit (mm above bg)", (h_t - bg) * 1000, SEQ, None),
              (f"Target change H* - H_T (deepen)", (h_t - h_star) * 1000, DIV, a.vlim)]
    for n, h in arms:
        panels.append((f"{n}\n(deepened = blue, raised = red)", (h_t - h) * 1000, DIV, a.vlim))
    for i, (title, img, cmap, vlim) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        kw = dict(vmin=-vlim, vmax=vlim) if vlim else {}
        im = ax.imshow(img[r0:r1, c0:c1], cmap=cmap, **kw)
        ax.contour(me[r0:r1, c0:c1], levels=[0.5], colors=INK, linewidths=0.9)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7, colors=INK2)
    base = visual_metrics(spec, h_t, h_t)
    names = ["start (no action)"] + [n for n, _ in arms]
    vals = [base] + [visual_metrics(spec, h, h_t) for _, h in arms]
    metrics = [("rms", "Shape error, RMS (mm)"), ("grad", "Appearance error (% of target slope)"),
               ("deep", "Groove deepened (mm); goal 2.35"), ("far", "Max change far away (mm)")]
    for j, (k, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(SURFACE)
        y = np.arange(len(names))[::-1]
        colors = [GRID] + [SERIES[i % len(SERIES)] for i in range(len(arms))]
        v = [x[k] for x in vals]
        ax.barh(y, v, color=colors, height=0.62)
        for yy, vv in zip(y, v):
            ax.text(vv, yy, f" {vv:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names if j == 0 else [""] * len(names), fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlim(0, max(max(v) * 1.35, 1e-6))
        ax.tick_params(axis="x", labelsize=7, colors=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_axisbelow(True)
    fig.suptitle("Groove deepening on letter A (delta = 3 mm). Bottom: lower is better, except 'deepened' (target 2.35)", fontsize=11, color=INK, y=0.995)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
