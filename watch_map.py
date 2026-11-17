"""보정이 진행되는 동안 높이맵이 어떻게 바뀌는지 창으로 본다 (다른 터미널에서 실행하거나 live_edit.py 가 띄운다).

왼쪽부터: 목표 H*(음영) | 지금 상태(음영) | 지금 - 수정 전 (파랑 = 더 깊어짐, 빨강 = 솟음) | 목표와의 차이.
run-dir/states/iter_XX.npy 가 새로 생길 때마다 갱신하고, 위에 반복 번호와 눈에 보이는 오차(L3D, Lviz: 시작 = 1.00)를 적는다.
화면(DISPLAY)이 있어야 한다. 창을 닫으면 종료한다.
사용: python watch_map.py --spec-dir S --run-dir OUT
"""
import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import numpy as np

from visual_score import VisualScorer

PIX_MM = 360.0 / 511
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--pad", type=int, default=70)
    a = ap.parse_args()
    h_t = np.load(a.spec_dir / "h_t.npy")
    h_star = np.load(a.spec_dir / "h_star.npy")
    me = np.load(a.spec_dir / "m_edit.npy") > 0
    sc = VisualScorer(h_star, h_t, me)
    ys, xs = np.where(me)
    r0, r1 = max(ys.min() - a.pad, 0), min(ys.max() + a.pad, 512)
    c0, c1 = max(xs.min() - a.pad, 0), min(xs.max() + a.pad, 512)
    ls = LightSource(azdeg=315, altdeg=40)
    shade = lambda h: ls.hillshade(h * 1000.0, vert_exag=2.0, dx=PIX_MM, dy=PIX_MM)[r0:r1, c0:c1]

    plt.ion()
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6))
    titles = ["Target H*", "Current state", "Current - before\n(blue = deeper, red = raised, mm)", "Current - Target (mm)"]
    imgs = [
        axs[0].imshow(shade(h_star), cmap="gray", vmin=0, vmax=1),
        axs[1].imshow(shade(h_t), cmap="gray", vmin=0, vmax=1),
        axs[2].imshow(np.zeros((r1 - r0, c1 - c0)), cmap=DIV, vmin=-4, vmax=4),
        axs[3].imshow(np.zeros((r1 - r0, c1 - c0)), cmap=DIV, vmin=-4, vmax=4),
    ]
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.contour(me[r0:r1, c0:c1], levels=[0.5], colors="#e34948" if ax in axs[:2] else "k", linewidths=0.8, linestyles="--")
    for im, ax in ((imgs[2], axs[2]), (imgs[3], axs[3])):
        fig.colorbar(im, ax=ax, fraction=0.046)
    sup = fig.suptitle("waiting for the first action...", fontsize=11)
    fig.tight_layout()
    shown = None
    while plt.fignum_exists(fig.number):
        files = sorted((a.run_dir / "states").glob("iter_*.npy"))
        if files and files[-1] != shown:
            shown = files[-1]
            try:
                h = np.load(shown)
            except Exception:
                plt.pause(0.5)
                continue
            r = sc.ratios(h)
            imgs[1].set_data(shade(h))
            imgs[2].set_data(((h_t - h) * 1000.0)[r0:r1, c0:c1])
            imgs[3].set_data(((h - h_star) * 1000.0)[r0:r1, c0:c1])
            sup.set_text(f"{shown.stem}   shape L3D x{r['l3d_win']:.2f}   appearance Lviz x{r['lviz_win']:.2f}   edge x{r['edge_win']:.2f}   (1.00 = no change, lower is better)")
        plt.pause(1.0)


if __name__ == "__main__":
    main()
