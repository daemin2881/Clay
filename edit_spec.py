"""설계 §2~3: SAM 마스크 -> (M_edit, M_boundary, M_preserve), 수정 목표 H*_edit, 손실 L.

  H*_edit = H_T - delta * soft(M_edit)           (수정 영역 밖은 정확히 H_T)
  L(H)    = w_r * |M_e (H-H*)|_1                 홈 깊이
          + w_g * |M_e (grad H - grad H*)|_1     홈 안쪽 기울기
          + w_b * |M_b (|grad H| - |grad H*|)|_1 홈 가장자리 선명도
          + w_p * |M_p (H-H_T)|_1                다른 획 보존
  각 항은 해당 마스크 화소 수로 나눈 평균이고, 높이는 mm, 기울기는 mm/화소 단위다.
  delta는 VLM이 정하지 않는다. 사람이 고정하는 값(0.5, 1.0, 3.0 mm 등)이다.

사용: python3 edit_spec.py --h-t H_T.npy --edit-mask M.npy --target-used T.npy --delta-mm 3 --out DIR
다른 스크립트에서: from edit_spec import load_spec; spec = load_spec(DIR, device); spec.loss(H)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

SIZE = 512
DEFAULT_WEIGHTS = dict(w_r=1.0, w_g=2.0, w_b=2.0, w_p=1.0)


def border_median(a):
    return float(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))


def build_masks(m_edit, target_used, *, band_px=3, preserve_margin_px=4, stroke_dilate_px=4):
    m_edit = m_edit.astype(bool)
    footprint = target_used > border_median(target_used) + 5e-4
    band = ndimage.binary_dilation(m_edit, iterations=band_px) & ~ndimage.binary_erosion(m_edit, iterations=band_px)
    keep_out = ndimage.binary_dilation(m_edit, iterations=band_px + preserve_margin_px)
    strokes = ndimage.binary_dilation(footprint, iterations=stroke_dilate_px) & ~keep_out
    return m_edit, band, strokes


def soft_mask(m_edit, feather_px):
    """마스크 안쪽에서만 0->1로 올라가는 부드러운 가중치(경계 거리 변환). 마스크 밖은 정확히 0."""
    if feather_px <= 0:
        return m_edit.astype(np.float32)
    inside = ndimage.distance_transform_edt(m_edit)
    return np.clip(inside / feather_px, 0.0, 1.0).astype(np.float32) * m_edit


def build_spec(h_t, m_edit, target_used, *, delta_mm, feather_px=4.0, band_px=3,
               preserve_margin_px=4, weights=None):
    for name, a in (("H_T", h_t), ("M_edit", m_edit), ("target_used", target_used)):
        if a.shape != (SIZE, SIZE):
            raise ValueError(f"{name}는 512x512여야 합니다: {a.shape}")
    me, mb, mp = build_masks(m_edit, target_used, band_px=band_px, preserve_margin_px=preserve_margin_px)
    if not me.any() or me.all():
        raise ValueError("M_edit이 비었거나 전체입니다.")
    soft = soft_mask(me, feather_px)
    h_star = (h_t.astype(np.float64) - delta_mm / 1000.0 * soft).astype(np.float32)
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    return dict(m_edit=me, m_boundary=mb, m_preserve=mp, soft=soft, h_star=h_star,
                h_t=h_t.astype(np.float32), weights=w,
                meta=dict(delta_mm=delta_mm, feather_px=feather_px, band_px=band_px,
                          preserve_margin_px=preserve_margin_px, n_edit=int(me.sum()),
                          n_boundary=int(mb.sum()), n_preserve=int(mp.sum()),
                          outside_edit_max_abs_delta_mm=float(np.abs(h_star - h_t)[~me].max()) * 1000))


def build_spec_from_target(h_t, h_star, m_edit, target_used, *, band_px=3, preserve_margin_px=4, weights=None, meta_extra=None):
    """이미 만든 목표 H*(예: 여러 영역을 합친 것)로 spec 을 만든다. m_edit 은 모든 편집 영역의 합집합."""
    for name, a in (("H_T", h_t), ("H*", h_star), ("M_edit", m_edit), ("target_used", target_used)):
        if a.shape != (SIZE, SIZE):
            raise ValueError(f"{name}는 512x512여야 합니다: {a.shape}")
    me, mb, mp = build_masks(m_edit, target_used, band_px=band_px, preserve_margin_px=preserve_margin_px)
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    meta = dict(delta_mm=None, feather_px=None, band_px=band_px, preserve_margin_px=preserve_margin_px, n_edit=int(me.sum()),
                n_boundary=int(mb.sum()), n_preserve=int(mp.sum()),
                outside_edit_max_abs_delta_mm=float(np.abs(h_star - h_t)[~me].max()) * 1000, target_given=True)
    meta.update(meta_extra or {})
    return dict(m_edit=me, m_boundary=mb, m_preserve=mp, soft=me.astype(np.float32), h_star=h_star.astype(np.float32),
                h_t=h_t.astype(np.float32), weights=w, meta=meta)


def save_spec(out: Path, s):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "m_edit.npy", s["m_edit"].astype(np.uint8))
    np.save(out / "m_boundary.npy", s["m_boundary"].astype(np.uint8))
    np.save(out / "m_preserve.npy", s["m_preserve"].astype(np.uint8))
    np.save(out / "soft_edit.npy", s["soft"])
    np.save(out / "h_star.npy", s["h_star"])
    np.save(out / "h_t.npy", s["h_t"])
    (out / "spec.json").write_text(json.dumps(dict(weights=s["weights"], **s["meta"]), indent=2))


def preview(out: Path, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys, xs = np.where(s["m_edit"] | s["m_boundary"])
    r0, r1, c0, c1 = max(ys.min() - 60, 0), min(ys.max() + 60, SIZE), max(xs.min() - 60, 0), min(xs.max() + 60, SIZE)
    base = border_median(s["h_t"])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    for a, img, t, cm in ((ax[0], (s["h_t"] - base) * 1000, "H_T (mm, bg=0)", "turbo"),
                          (ax[1], (s["h_star"] - base) * 1000, "H*_edit (mm)", "turbo"),
                          (ax[2], (s["h_star"] - s["h_t"]) * 1000, "H* - H_T (mm)", "RdBu_r")):
        im = a.imshow(img[r0:r1, c0:c1], cmap=cm)
        a.set_title(t)
        plt.colorbar(im, ax=a, fraction=0.046)
    rgb = np.zeros((r1 - r0, c1 - c0, 3))
    rgb[..., 0] = s["m_edit"][r0:r1, c0:c1]
    rgb[..., 1] = s["m_boundary"][r0:r1, c0:c1]
    rgb[..., 2] = s["m_preserve"][r0:r1, c0:c1]
    ax[3].imshow(rgb)
    ax[3].set_title("M_edit(R) M_boundary(G) M_preserve(B)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "spec_preview.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# torch 손실 (GPU 후보 배치 지원). H: (B,1,512,512) 또는 (1,1,512,512), 단위 m
# ----------------------------------------------------------------------------
class Spec:
    def __init__(self, d: Path, device, dtype):
        import torch
        self.torch = torch
        self.dir = d
        cfg = json.loads((d / "spec.json").read_text())
        self.weights = cfg["weights"]
        self.meta = cfg
        ld = lambda n: torch.from_numpy(np.load(d / n)).to(device=device, dtype=dtype)[None, None]
        self.m_edit, self.m_boundary, self.m_preserve = (ld("m_edit.npy") > 0.5, ld("m_boundary.npy") > 0.5,
                                                         ld("m_preserve.npy") > 0.5)
        self.h_star, self.h_t = ld("h_star.npy"), ld("h_t.npy")
        self.n_e, self.n_b, self.n_p = (float(m.sum()) for m in (self.m_edit, self.m_boundary, self.m_preserve))
        self.gx_star, self.gy_star = self.grad(self.h_star)
        self.gmag_star = torch.sqrt(self.gx_star ** 2 + self.gy_star ** 2 + 1e-12)

    @staticmethod
    def grad(h):
        """중앙차분 기울기 (mm/화소). 가장자리는 복제 패딩."""
        import torch.nn.functional as F
        p = F.pad(h * 1000.0, (1, 1, 1, 1), mode="replicate")
        return (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) * 0.5, (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) * 0.5

    def terms(self, h):
        torch = self.torch
        gx, gy = self.grad(h)
        gmag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        me, mb, mp = self.m_edit, self.m_boundary, self.m_preserve
        red = lambda x, m, n: (x * m).sum(dim=(1, 2, 3)) / n
        r = red((h - self.h_star).abs() * 1000.0, me, self.n_e)
        g = red((gx - self.gx_star).abs() + (gy - self.gy_star).abs(), me, self.n_e)
        b = red((gmag - self.gmag_star).abs(), mb, self.n_b)
        p = red((h - self.h_t).abs() * 1000.0, mp, self.n_p)
        return dict(r=r, g=g, b=b, p=p)

    def loss(self, h, return_terms=False):
        t = self.terms(h)
        w = self.weights
        total = w["w_r"] * t["r"] + w["w_g"] * t["g"] + w["w_b"] * t["b"] + w["w_p"] * t["p"]
        return (total, t) if return_terms else total


def load_spec(d, device="cpu", dtype=None):
    import torch
    return Spec(Path(d), device, dtype or torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-t", required=True, type=Path, help="수정 직전 완성 상태 H_T (512x512, m)")
    ap.add_argument("--edit-mask", required=True, type=Path, help="SAM 2 가 만든 M_edit (0/1 .npy)")
    ap.add_argument("--target-used", required=True, type=Path, help="조형에 쓴 목표(A 획의 footprint 용)")
    ap.add_argument("--delta-mm", type=float, help="사람이 정한 수정량. VLM이 정하지 않는다. (--h-star 를 주면 생략)")
    ap.add_argument("--h-star", type=Path, help="이미 만든 목표(512x512, m). 주면 --delta-mm 대신 이것을 쓴다(여러 영역 편집용).")
    ap.add_argument("--feather-px", type=float, default=4.0)
    ap.add_argument("--band-px", type=int, default=3)
    ap.add_argument("--preserve-margin-px", type=int, default=4)
    for k, v in DEFAULT_WEIGHTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    weights = {k: getattr(a, k) for k in DEFAULT_WEIGHTS}
    if a.h_star is not None:
        s = build_spec_from_target(np.load(a.h_t).astype(np.float32), np.load(a.h_star).astype(np.float32), np.load(a.edit_mask) > 0,
                                   np.load(a.target_used).astype(np.float32), band_px=a.band_px,
                                   preserve_margin_px=a.preserve_margin_px, weights=weights)
    else:
        if a.delta_mm is None:
            ap.error("--delta-mm 또는 --h-star 가 필요합니다.")
        s = build_spec(np.load(a.h_t).astype(np.float32), np.load(a.edit_mask) > 0, np.load(a.target_used).astype(np.float32),
                       delta_mm=a.delta_mm, feather_px=a.feather_px, band_px=a.band_px,
                       preserve_margin_px=a.preserve_margin_px, weights=weights)
    save_spec(a.out, s)
    preview(a.out, s)
    m = s["meta"]
    print(f"M_edit {m['n_edit']}px, M_boundary {m['n_boundary']}px, M_preserve {m['n_preserve']}px | "
          f"delta={a.delta_mm if a.h_star is None else '목표 직접 지정'} | 수정 영역 밖 |H*-H_T| 최대 {m['outside_edit_max_abs_delta_mm']:.6f}mm")
    spec = load_spec(a.out)
    import torch
    h_t = torch.from_numpy(s["h_t"])[None, None]
    L, t = spec.loss(h_t, return_terms=True)
    print("L(H_T) =", f"{float(L):.4f}", {k: round(float(v), 4) for k, v in t.items()}, "(H=H_T 이면 w_p 항은 0)")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
