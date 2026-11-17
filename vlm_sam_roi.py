"""VLM prompt JSON(vlm_roi_prompt.py 스키마) + 이미지 -> SAM 2 ROI 마스크(512x512).

기존 sam2_cat_roi.py 는 고양이 전용이라, 같은 prompt 스키마(box_norm / positive / negative 점)를 임의 이미지에 쓰는 얇은 실행기다.
출력: roi_mask_512.npy(bool 0/1), roi_mask_512.png, roi_overlay.png, roi_sam_report.json
사용: python vlm_sam_roi.py --image state_A_view.png --prompt-json prompt_groove.json --out-dir DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

SAM2_ROOT = Path("/home/airlab/research/clay_genesis/sam2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--prompt-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=SAM2_ROOT / "checkpoints/sam2.1_hiera_small.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--box-margin-px", type=float, default=2.0, help="마스크를 VLM 박스(+여유)로 잘라 SAM이 번지는 것을 막는다")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    p = json.loads(a.prompt_json.read_text())
    S = a.size
    img = Image.open(a.image).convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
    rgb = np.asarray(img, dtype=np.uint8)
    box = np.asarray([v * (S - 1) for v in p["box_norm"]], dtype=np.float32)
    pos, neg = p["positive_points_norm"], p["negative_points_norm"]
    pts = np.asarray([[x * (S - 1), y * (S - 1)] for x, y in pos + neg], dtype=np.float32)
    lab = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    model = build_sam2(a.config, str(a.checkpoint), device=a.device)
    pred = SAM2ImagePredictor(model)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if a.device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), ctx:
        pred.set_image(rgb)
        masks, scores, _ = pred.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
    masks = masks.astype(bool)

    x1, y1, x2, y2 = box
    m = a.box_margin_px
    allowed = np.zeros((S, S), bool)
    allowed[int(max(0, y1 - m)):int(min(S, y2 + m + 1)), int(max(0, x1 - m)):int(min(S, x2 + m + 1))] = True
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    cand = []
    for i, (mk, sc) in enumerate(zip(masks, scores)):
        inside = (mk & allowed).sum()
        leak = 1.0 - inside / max(1, mk.sum())               # 박스 밖으로 번진 비율
        hit_pos = all(mk[int(round(y)), int(round(x))] for x, y in [(q[0], q[1]) for q in pts[lab == 1]])
        hit_neg = any(mk[int(round(y)), int(round(x))] for x, y in [(q[0], q[1]) for q in pts[lab == 0]])
        cand.append(dict(index=i, sam_score=float(sc), area_px=int(mk.sum()), inside_box_area_px=int(inside),
                         leak_outside_box=float(leak), area_over_box=float(inside / box_area), pos_ok=bool(hit_pos), neg_hit=bool(hit_neg)))
    ok = [c for c in cand if c["pos_ok"] and not c["neg_hit"]]
    pool = ok or cand
    best = max(pool, key=lambda c: c["sam_score"] - 0.5 * c["leak_outside_box"])
    roi = masks[best["index"]] & allowed
    # SAM 마스크의 작은 구멍/가장자리 톱니를 정리: 닫기 + 구멍 채우기, 양성점이 속한 연결 성분만 유지
    from scipy import ndimage as ndi
    raw_px = int(roi.sum())
    roi = ndi.binary_fill_holes(ndi.binary_closing(roi, iterations=2)) & allowed
    lbl, _ = ndi.label(roi)
    keep = {int(lbl[int(round(y)), int(round(x))]) for x, y in pts[lab == 1]} - {0}
    if keep:
        roi = np.isin(lbl, list(keep))
    best["raw_area_px"], best["cleaned_area_px"] = raw_px, int(roi.sum())

    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(a.out_dir / "roi_mask_512.npy", roi.astype(np.uint8))
    Image.fromarray(roi.astype(np.uint8) * 255, mode="L").save(a.out_dir / "roi_mask_512.png")
    ov = np.asarray(img).astype(np.float32)
    ov[roi] = 0.45 * ov[roi] + 0.55 * np.array([255, 60, 60])
    ovi = Image.fromarray(ov.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(ovi)
    d.rectangle([x1, y1, x2, y2], outline=(255, 255, 0))
    for (x, y), l in zip(pts, lab):
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 255, 0) if l else (0, 160, 255), width=2)
    ovi.resize((S * 2, S * 2), Image.Resampling.NEAREST).save(a.out_dir / "roi_overlay.png")
    (a.out_dir / "roi_sam_report.json").write_text(json.dumps(dict(
        prompt=str(a.prompt_json), image=str(a.image), selected=best, candidates=cand, roi_area_px=int(roi.sum()),
        box_px=[float(v) for v in box], used_fallback_all_candidates=not ok), ensure_ascii=False, indent=2))
    print(f"후보 {len(cand)}개:")
    for c in cand:
        print(f"  #{c['index']} score={c['sam_score']:.3f} 면적={c['area_px']} 박스밖번짐={c['leak_outside_box']:.2f} 양성점OK={c['pos_ok']} 음성점침범={c['neg_hit']}")
    print(f"=> 선택 #{best['index']}, ROI {int(roi.sum())}px ({roi.sum() * 0.703**2:.0f} mm²) | 저장: {a.out_dir}")


if __name__ == "__main__":
    main()
