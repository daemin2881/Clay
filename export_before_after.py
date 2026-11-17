"""반사실 과거 보상 적용 전/후를 이미지 파일로 모두 내보낸다.
  before_heightmap.png / after_heightmap.png / diff_heightmap.png : 높이맵(같은 색 범위)
  particles_before_after.png : Genesis 저장 상태의 MPM 입자 위에서 본 모습(높이로 색칠). 높이맵이 아니라 시뮬레이터 입자 자체이다.
사용: python export_before_after.py --before-state S.pt --after-state S2.pt --before-h H.npy --after-h H2.npy --region-mask M.npy --out-dir DIR
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vs_history import load_snapshot

X0, Y0, PS = 0.44 - 0.18, -0.18, 0.36 / 511


def particles(state):
    p = next(x for x in load_snapshot(state)["physics"] if x is not None and x["is_mpm"])["tensors"]
    pos = p["_pos"][0].numpy()
    act = p["_active"][0].numpy() > 0.5
    return pos[act]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-state", required=True, type=Path)
    ap.add_argument("--after-state", required=True, type=Path)
    ap.add_argument("--before-h", required=True, type=Path)
    ap.add_argument("--after-h", required=True, type=Path)
    ap.add_argument("--region-mask", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pad", type=int, default=70)
    ap.add_argument("--label-after", default="After: past action #186 replaced by a no-op, replayed")
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    hb, ha = (np.load(p).astype(np.float64) * 1000 for p in (a.before_h, a.after_h))
    m = np.load(a.region_mask) > 0
    ys, xs = np.where(m)
    r0, r1, c0, c1 = max(ys.min() - a.pad, 0), min(ys.max() + a.pad, 512), max(xs.min() - a.pad, 0), min(xs.max() + a.pad, 512)
    bg = float(np.median(np.concatenate([hb[0], hb[-1], hb[:, 0], hb[:, -1]])))
    kw = dict(cmap="turbo", vmin=-10, vmax=14)
    for name, h, title in (("before_heightmap", hb, "Before: recorded A after 190 actions"), ("after_heightmap", ha, a.label_after)):
        fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor="#fcfcfb")
        im = ax.imshow((h - bg)[r0:r1, c0:c1], **kw)
        ax.contour(m[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=1.0, linestyles="--")
        ax.set_title(title + "\nheight above background (mm)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(out / f"{name}.png", dpi=120, facecolor="#fcfcfb", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor="#fcfcfb")
    im = ax.imshow((ha - hb)[r0:r1, c0:c1], cmap="RdBu", vmin=-4, vmax=4)
    ax.contour(m[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=1.0, linestyles="--")
    ax.set_title("After - Before (mm): blue = higher after, red = lower after", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out / "diff_heightmap.png", dpi=120, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    # 입자 (시뮬레이터 상태 자체): 영역 중심을 지나는 단면(x-z, y-z)에 있는 입자를 수정 전/후로 겹쳐 그린다
    cy, cx = ys.mean(), xs.mean()
    xc, yc = (X0 + cx * PS) * 1000, (Y0 + cy * PS) * 1000
    pb, pa = particles(a.before_state) * 1000, particles(a.after_state) * 1000
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), facecolor="#fcfcfb")
    band = 4.0
    for ax, ia, ib, c0_, ttl in ((axs[0], 0, 1, yc, f"x-z section: particles with |y - {yc:.0f}| < {band:.0f} mm"), (axs[1], 1, 0, xc, f"y-z section: particles with |x - {xc:.0f}| < {band:.0f} mm")):
        for P, col, lab, sz in ((pb, "#8a8f98", "Before (recorded 190)", 10), (pa, "#e34948", "After (revised past)", 5)):
            k = np.abs(P[:, ib] - c0_) < band
            ax.scatter(P[k, ia], P[k, 2], s=sz, c=col, label=lab, linewidths=0, alpha=0.8)
        ax.set_xlim((xc if ia == 0 else yc) - 45, (xc if ia == 0 else yc) + 45); ax.set_ylim(35, 75)
        ax.set_xlabel("world " + ("x" if ia == 0 else "y") + " (mm)"); ax.set_ylabel("z (mm)"); ax.set_title(ttl, fontsize=10); ax.legend(loc="lower right")
    fig.suptitle("Simulator particles (Genesis MPM state) before and after, through the broken-leg region", fontsize=11)
    fig.savefig(out / "particles_before_after.png", dpi=110, facecolor="#fcfcfb", bbox_inches="tight")
    print("저장:", [p.name for p in sorted(out.glob("*.png"))])


if __name__ == "__main__":
    main()
