"""목표 height map 과 각 결과를 음영(hillshade) 이미지로 나란히 놓는다 — '눈으로 같은가'를 직접 보기 위한 그림.

같은 광원(왼쪽 위), 같은 수직 과장, 같은 확대 영역(홈 + 주변)을 쓴다. 첫 두 칸은 수정 전(H_T)과 목표(H*).
사용: python make_shaded_compare.py --spec-dir S --arm NAME=DIR [--arm ...] --out fig.png   (DIR 에 final_m.npy)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

from edit_spec import load_spec

PIX_MM = 360.0 / 511


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pad", type=int, default=70)
    ap.add_argument("--exag", type=float, default=2.0)
    ap.add_argument("--target-title", default="Target H*")
    a = ap.parse_args()
    spec = load_spec(a.spec_dir)
    me = spec.m_edit[0, 0].numpy()
    ys, xs = np.where(me)
    r0, r1 = max(ys.min() - a.pad, 0), min(ys.max() + a.pad, 512)
    c0, c1 = max(xs.min() - a.pad, 0), min(xs.max() + a.pad, 512)
    panels = [("Before (H_T)", spec.h_t[0, 0].numpy()), (a.target_title, spec.h_star[0, 0].numpy())]
    for s in a.arm:
        n, d = s.split("=", 1)
        panels.append((n.replace("\\n", "\n"), np.load(Path(d) / "final_m.npy")))
    ls = LightSource(azdeg=315, altdeg=40)
    fig, axs = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.9), facecolor="#fcfcfb")
    for ax, (title, h) in zip(axs, panels):
        shade = ls.hillshade(h * 1000.0, vert_exag=a.exag, dx=PIX_MM, dy=PIX_MM)
        ax.imshow(shade[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1)
        ax.contour(me[r0:r1, c0:c1], levels=[0.5], colors="#e34948", linewidths=0.9, linestyles="--")
        ax.set_title(title, fontsize=9, color="#0b0b0b")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Shaded relief, same light and zoom. Dashed red = region to edit. Judge by eye: does it look like the Target?",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor="#fcfcfb")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
