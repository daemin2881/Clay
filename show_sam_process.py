"""SAM2 마스크가 어떻게 만들어졌는지 단계별로 한 장에 보여 준다 (입력 높이맵 -> VLM 프롬프트 -> SAM2 후보 3개 -> 선택·정리 -> 마스크 3종).
vlm_sam_roi.py 와 같은 입력·같은 선택 규칙을 쓰고, 후보 마스크 3개를 모두 그린다.
사용: python show_sam_process.py --image state_A_color.png --prompt-json prompt_groove.json --spec-dir S --out fig.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

SAM2_ROOT = Path("/home/airlab/research/clay_genesis/sam2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--prompt-json", required=True, type=Path)
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    S = 512
    p = json.loads(a.prompt_json.read_text())
    rgb = np.asarray(Image.open(a.image).convert("RGB").resize((S, S), Image.Resampling.LANCZOS), dtype=np.uint8)
    box = np.asarray([v * (S - 1) for v in p["box_norm"]], dtype=np.float32)
    pos, neg = p["positive_points_norm"], p["negative_points_norm"]
    pts = np.asarray([[x * (S - 1), y * (S - 1)] for x, y in pos + neg], dtype=np.float32)
    lab = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(SAM2_ROOT / "checkpoints/sam2.1_hiera_small.pt"), device="cuda")
    pred = SAM2ImagePredictor(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(rgb)
        masks, scores, _ = pred.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
    masks = masks.astype(bool)
    x1, y1, x2, y2 = box
    allowed = np.zeros((S, S), bool)
    allowed[int(max(0, y1 - 2)):int(min(S, y2 + 3)), int(max(0, x1 - 2)):int(min(S, x2 + 3))] = True
    info = []
    for i, (mk, sc) in enumerate(zip(masks, scores)):
        inside = (mk & allowed).sum()
        leak = 1.0 - inside / max(1, mk.sum())
        ok = all(mk[int(round(y)), int(round(x))] for x, y in pts[lab == 1]) and not any(mk[int(round(y)), int(round(x))] for x, y in pts[lab == 0])
        info.append(dict(i=i, score=float(sc), area=int(mk.sum()), leak=float(leak), ok=bool(ok), rank=float(sc) - 0.5 * leak))
    pool = [c for c in info if c["ok"]] or info
    best = max(pool, key=lambda c: c["rank"])
    sel = masks[best["i"]] & allowed
    sel = ndi.binary_fill_holes(ndi.binary_closing(sel, iterations=2)) & allowed
    lbl, _ = ndi.label(sel)
    keep = {int(lbl[int(round(y)), int(round(x))]) for x, y in pts[lab == 1]} - {0}
    sel = np.isin(lbl, list(keep)) if keep else sel

    r0, r1 = int(max(y1 - 70, 0)), int(min(y2 + 70, S))
    c0, c1 = int(max(x1 - 70, 0)), int(min(x2 + 70, S))
    fig = plt.figure(figsize=(17, 8.6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.12)

    def draw_prompt(ax, zoom):
        for (x, y), l in zip(pts, lab):
            ax.plot(x, y, "o", mfc="none", mec="#1fcc1f" if l else "#2a9df4", mew=2, ms=9)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#f0d800", lw=1.8))
        if zoom:
            ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(rgb); ax.set_title("(1) Input: final A after 190 actions\nheight map, turbo colormap (red = high)", fontsize=10)
    ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="white", lw=1.2, ls="--"))
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(rgb); draw_prompt(ax, True)
    ax.set_title("(2) VLM prompt (stand-in): box (yellow),\ninside points (green), outside points (blue)", fontsize=10)
    for k, c in enumerate(info):
        ax = fig.add_subplot(gs[0, 2] if k == 0 else gs[1, k - 1]); ax.imshow(rgb)
        ov = np.zeros((S, S, 4)); ov[masks[k]] = (0.9, 0.15, 0.6, 0.55); ax.imshow(ov)
        draw_prompt(ax, True)
        tag = "  <- SELECTED" if k == best["i"] else ""
        ax.set_title(f"(3) SAM2 candidate #{k}{tag}\nscore {c['score']:.3f}, area {c['area']} px, hits inside points: {c['ok']}", fontsize=10,
                     color="#b00020" if k == best["i"] else "#0b0b0b")
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(rgb)
    ov = np.zeros((S, S, 4)); ov[sel] = (0.9, 0.15, 0.6, 0.6); ax.imshow(ov); draw_prompt(ax, True)
    ax.set_title(f"(4) Cleaned: clip to box, close + fill holes,\nkeep component with inside points = {int(sel.sum())} px", fontsize=10)
    ax = fig.add_subplot(gs[1, 3])
    me, mb, mp = (np.load(a.spec_dir / n) > 0 for n in ("m_edit.npy", "m_boundary.npy", "m_preserve.npy"))
    img = np.zeros((S, S, 3)); img[..., 2] = mp; img[..., 1] = mb; img[..., 0] = me
    ax.imshow(img[r0:r1, c0:c1]); ax.set_title("(5) Three masks used by the method\nM_edit (red), M_boundary (green), M_preserve (blue)", fontsize=10)
    for ax in fig.axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("How the groove mask was made: height map -> VLM prompt -> SAM2 -> cleanup -> masks", fontsize=12, y=0.98)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100, facecolor="#fcfcfb", bbox_inches="tight")
    print("저장:", a.out, "| 선택 #%d, 정리 후 %d px" % (best["i"], int(sel.sum())))


if __name__ == "__main__":
    main()
