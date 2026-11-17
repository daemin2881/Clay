"""보상 실행들의 높이맵을 사진(PNG)으로 저장한다. 파일 이름과 제목에 '총 행동 수'가 들어간다.
각 PNG: 왼쪽 높이맵(배경 위 mm, turbo, 모든 그림이 같은 색 범위), 오른쪽 음영(같은 광원). 점선 = 요청 영역.
입력은 compensation_manifest.py 가 만든 JSON 이다. 이력(보상 없음)의 원하는 시점도 같은 형식으로 뽑는다.

사용: python export_pngs.py --manifest results/case_A/compensation_manifest.json --history-dir H --regions DIR --out results/case_A/png \
        [--orig 80,85,90,100,105] [--arm-steps 5,10,15,20]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

import cf_common as cc
from region_set import RegionSet, border_median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--orig", default="80,85,90,100,105", help="보상 없는 원래 이력에서 그릴 총 행동 수들")
    ap.add_argument("--arm-steps", default="5,10,15,20")
    ap.add_argument("--pad", type=int, default=50)
    a = ap.parse_args()
    root = a.history_dir.resolve()
    g = cc.geometry(root)
    S = lambda k: np.load(root / f"action_{k:06d}" / "transition.npz")["after_m"].astype(np.float64)
    h80 = S(80)
    rs = RegionSet.load(a.regions, ref_default=h80)
    union = rs.union()
    tgt = cc.load_target(root).astype(np.float64)
    fp = cc.relief_focus(tgt, g["tool_r_px"], base=cc.plate_level(root))
    ys, xs = np.where(fp["focus"] | union)
    r0, r1 = max(ys.min() - a.pad, 0), min(ys.max() + a.pad, 512)
    c0, c1 = max(xs.min() - a.pad, 0), min(xs.max() + a.pad, 512)
    ls = LightSource(azdeg=315, altdeg=40)
    a.out.mkdir(parents=True, exist_ok=True)
    written = []

    def draw(h, title, path):
        bg = border_median(h)
        fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.4), facecolor="#fcfcfb")
        axs[0].imshow(((h - bg) * 1000)[r0:r1, c0:c1], cmap="turbo", vmin=-10, vmax=14)
        axs[1].imshow(ls.hillshade(h * 1000.0, vert_exag=2.0, dx=g["px_mm"], dy=g["px_mm"])[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1)
        axs[0].set_title("height above plate (mm)", fontsize=9)
        axs[1].set_title("shaded relief (same light)", fontsize=9)
        for ax in axs:
            ax.contour(union[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=10.5)
        fig.savefig(path, dpi=100, facecolor="#fcfcfb", bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    draw(tgt, "Target A (requested regions dashed)", a.out / "target.png")
    for k in [int(x) for x in a.orig.split(",") if x]:
        draw(S(k), f"No compensation (original history): total {k} actions", a.out / f"original_{k:03d}actions.png")
    man = json.loads(a.manifest.read_text())
    steps = [int(x) for x in a.arm_steps.split(",") if x]
    for name, v in man.items():
        if "stages" in v:                         # 보상 실행: 확정 단계마다
            for st in v["stages"]:
                p = Path(st["heightmap"])
                if p.exists():
                    tag = st["stage"].split(" ")[0] + (st["stage"].split(" ")[1] if st["stage"].startswith("commit") else "")
                    draw(np.load(p).astype(np.float64), f"Compensation {name} - {st['stage']}: total {st['total_actions']} actions", a.out / f"{name}_{st['total_actions']:03d}actions_{tag}.png")
            fh = Path(v["final_heightmap"])
            if fh.exists():
                draw(np.load(fh).astype(np.float64), f"Compensation {name} FINAL: total {v['final_total_actions']} actions", a.out / f"{name}_FINAL_{v['final_total_actions']:03d}actions.png")
        else:                                      # 이어 조형 팔
            base = v["start_total_actions"]
            d = Path(v["final_heightmap"]).parent
            for s in steps:
                p = d / "states" / f"step_{s:03d}.npy"
                if p.exists():
                    draw(np.load(p).astype(np.float64), f"{name} (continued sculpting): total {base + s} actions", a.out / f"{name}_{base + s:03d}actions.png")
            fh = Path(v["final_heightmap"])
            if fh.exists():
                draw(np.load(fh).astype(np.float64), f"{name} FINAL: total {v['final_total_actions']} actions", a.out / f"{name}_FINAL_{v['final_total_actions']:03d}actions.png")
    print(f"PNG {len(written)}개 저장: {a.out}")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
