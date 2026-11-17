"""A/하트 같은 단순 타깃으로 빠르게 First-Beam을 검증하기 위한 케이스 생성기.

1) prepare : H_base(.npy)와 수정 대상 ROI 마스크를 만든다. 이 둘로 baseline 조형을 돌린다.
2) edit    : 조형이 끝난 뒤 '실제로 달성된 최종 높이맵 H_T'에서 ROI만 수정한 H_edit를 만든다.
             (이상적 목표가 아니라 H_T 기준이므로 baseline이 완벽하지 않아도 비교가 공정하다.)

사용 예 (baseline 폴더에 이 파일을 함께 두고 실행)
  python edit_case.py prepare --target heart --out case_heart
  python visual_sculpting.py --checkpoint CKPT --target-heightmap case_heart/H_base_512_m.npy \
        --roi-mask case_heart/roi_mask_512.npy --history-dir runs/heart_hist --output-dir runs/heart_out
  python edit_case.py edit --history-dir runs/heart_hist --op lower --factor 0.5 --out case_heart/edit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

SIZE = 512


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def border_median(a):
    return float(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))


def footprint_of(h, thresh_m=5e-4):
    return h > border_median(h) + thresh_m


def segments_in_row(mask_row):
    return int(ndimage.label(mask_row[None, :])[1])


def auto_roi(h_base, name, box=None):
    fp = footprint_of(h_base)
    rows, cols = np.where(fp)
    if rows.size == 0:
        raise ValueError("타깃 footprint가 비었습니다.")
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    H, W = r1 - r0, c1 - c0
    if box is not None:
        fr0, fr1, fc0, fc1 = box
        sel_r, sel_c = (fr0, fr1), (fc0, fc1)
        note = "사용자 지정 박스"
    elif name.lower() == "heart":
        # 로브 끝은 맨 가장자리 근처에서만 두 덩어리로 갈라진다(V자 홈이 얕음). 여러 깊이에서 최대값을 쓴다.
        probes = [max(int(f * H), 1) for f in (0.01, 0.02, 0.04, 0.06)]
        top = max(segments_in_row(fp[r0 + k]) for k in probes)
        bot = max(segments_in_row(fp[r1 - 1 - k]) for k in probes)
        if top >= bot:   # 로브(두 봉우리)가 있는 쪽
            sel_r, note = (0.0, 0.45), "하트: 배열 위쪽이 로브 쪽, 왼쪽 로브"
        else:
            sel_r, note = (0.55, 1.0), "하트: 배열 아래쪽이 로브 쪽, 왼쪽 로브"
        sel_c = (0.0, 0.5)
    else:
        probe = max(int(0.08 * H), 1)
        wtop = int(fp[r0:r0 + probe].sum(axis=1).mean())
        wbot = int(fp[r1 - probe:r1].sum(axis=1).mean())
        if wtop <= wbot:
            sel_r, note = (0.0, 0.30), "글자: 배열 위쪽 꼭대기(좁은 쪽)"
        else:
            sel_r, note = (0.70, 1.0), "글자: 배열 아래쪽 꼭대기(좁은 쪽)"
        sel_c = (0.0, 1.0)
    roi = np.zeros((SIZE, SIZE), bool)
    rr0, rr1 = r0 + int(sel_r[0] * H), r0 + int(np.ceil(sel_r[1] * H))
    cc0, cc1 = c0 + int(sel_c[0] * W), c0 + int(np.ceil(sel_c[1] * W))
    roi[rr0:rr1, cc0:cc1] = True
    roi &= ndimage.binary_dilation(fp, iterations=6)
    if roi.sum() == 0 or roi.all():
        raise ValueError("ROI가 비었거나 전체입니다. --roi-box로 직접 지정하세요.")
    return roi, note


def preview(path, panels, titles, roi=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    ax = np.atleast_1d(ax)
    for a, p, t in zip(ax, panels, titles):
        sym = t.startswith("D")
        v = float(np.abs(p).max()) or 1e-9
        a.imshow(p, cmap="RdBu_r" if sym else "viridis", vmin=-v if sym else None, vmax=v if sym else None)
        if roi is not None:
            a.contour(roi, levels=[0.5], colors="k", linewidths=0.9)
        a.set_title(t)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_prepare(args):
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.h_base:
        h = np.load(args.h_base.expanduser(), allow_pickle=False).astype(np.float32)
        name = args.target or "custom"
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        from vs_targets import make_target       # baseline 폴더에 이 파일을 두고 실행
        h = make_target(args.target, torch.device("cpu")).float().numpy()
        name = args.target
    if h.shape != (SIZE, SIZE) or not np.isfinite(h).all() or np.abs(h).max() > 1.0:
        raise ValueError("H_base는 단위 m의 512x512 배열이어야 합니다.")
    box = tuple(float(x) for x in args.roi_box.split(",")) if args.roi_box else None
    roi, note = auto_roi(h, name, box)
    np.save(out / "H_base_512_m.npy", h)
    np.save(out / "roi_mask_512.npy", roi)
    relief = (h - border_median(h)) * 1000
    preview(out / "prepare_preview.png", [relief, roi.astype(float)], ["H_base relief (mm)", "ROI"], roi)
    print(f"저장: {out}/H_base_512_m.npy, roi_mask_512.npy  ({note}, ROI 픽셀 {int(roi.sum())})")
    print(f"prepare_preview.png 에서 ROI 위치가 의도와 맞는지 먼저 확인하세요. 아니면 --roi-box r0,r1,c0,c1 (footprint bbox 비율)")


def cmd_edit(args):
    root = args.history_dir.expanduser().resolve()
    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    if not dirs:
        raise FileNotFoundError("이력 폴더에 action_XXXXXX가 없습니다.")
    h_t = np.load(dirs[-1] / "transition.npz")["after_m"].astype(np.float32)
    roi_file = args.roi.expanduser() if args.roi else root / "roi_mask_512.npy"
    roi = np.load(roi_file).astype(bool)
    base = args.base_level_mm / 1000 if args.base_level_mm is not None else border_median(h_t)
    w = ndimage.gaussian_filter(roi.astype(np.float32), args.feather_px)
    w = np.clip(w / max(w.max(), 1e-9), 0, 1) if args.feather_px > 0 else roi.astype(np.float32)
    w = np.where(roi, np.maximum(w, 0.0), w * 0.0)  # 페더링이 ROI 밖으로 번지지 않게 제한
    if args.op == "lower":
        edit = h_t - w * (1.0 - args.factor) * np.maximum(h_t - base, 0.0)
    elif args.op == "raise":
        edit = h_t + w * (args.factor - 1.0) * np.maximum(h_t - base, 0.0)
    elif args.op == "smooth":
        edit = h_t + w * (ndimage.gaussian_filter(h_t, args.sigma_px) - h_t)
    else:
        raise ValueError(args.op)
    edit = edit.astype(np.float32)
    D = edit - h_t
    outside_max = float(np.abs(D[~roi]).max()) * 1000
    out = (args.out or root.parent / (root.name + "_edit")).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "H_T_512_m.npy", h_t)
    np.save(out / "h_edit_512_m.npy", edit)
    np.save(out / "desired_delta_512_m.npy", D)
    meta = dict(op=args.op, factor=args.factor, sigma_px=args.sigma_px, feather_px=args.feather_px,
                base_level_m=base, n_actions=len(dirs), roi_pixels=int(roi.sum()),
                max_abs_delta_mm=float(np.abs(D).max()) * 1000,
                roi_mean_abs_delta_mm=float(np.abs(D[roi]).mean()) * 1000,
                outside_roi_max_abs_delta_mm=outside_max,
                files={n: sha256(out / n) for n in ("H_T_512_m.npy", "h_edit_512_m.npy", "desired_delta_512_m.npy")},
                note="H_edit는 baseline이 실제로 만든 최종 상태 H_T에서 ROI만 수정한 목표. 물리 조형 결과가 아님.")
    (out / "edit_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    preview(out / "edit_preview.png", [h_t * 1000, edit * 1000, D * 1000],
            ["H_T (mm)", "H_edit (mm)", "D = H_edit - H_T (mm)"], roi)
    print(f"행동 {len(dirs)}개 이력의 최종 상태에서 H_edit 생성: ROI 평균 |D|={meta['roi_mean_abs_delta_mm']:.3f} mm, "
          f"ROI 밖 최대 |D|={outside_max:.6f} mm")
    print(f"저장: {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--target", help="heart, A, B ... (vs_targets.make_target 이름)")
    a.add_argument("--h-base", type=Path, help="make_target 대신 직접 만든 512x512 .npy")
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--roi-box", help="r0,r1,c0,c1 : footprint bbox 대비 비율(0~1). 배열의 위/왼쪽이 0")
    a.set_defaults(fn=cmd_prepare)
    b = sub.add_parser("edit")
    b.add_argument("--history-dir", required=True, type=Path)
    b.add_argument("--op", choices=["lower", "raise", "smooth"], default="lower")
    b.add_argument("--factor", type=float, default=0.5, help="lower/raise: ROI 릴리프 배율")
    b.add_argument("--sigma-px", type=float, default=8.0, help="smooth: 가우시안 sigma")
    b.add_argument("--feather-px", type=float, default=3.0)
    b.add_argument("--base-level-mm", type=float, help="기준 면 높이(m 단위가 아니라 mm). 기본: 테두리 중앙값")
    b.add_argument("--roi", type=Path, help="기본값: history-dir/roi_mask_512.npy (이력에 ROI가 없을 때 지정)")
    b.add_argument("--out", type=Path)
    b.set_defaults(fn=cmd_edit)
    args = p.parse_args()
    if args.cmd == "prepare" and not (args.target or args.h_base):
        p.error("--target 또는 --h-base가 필요합니다.")
    args.fn(args)


if __name__ == "__main__":
    main()
