"""영역 집합: 보상할 영역들의 (마스크, 목표) 쌍. 대상(A, 고양이, 하트, ...)이나 영역의 의미(홈, 다리, ...)를 코드가 알지 못한다.

저장 형식(폴더 하나):
  regions.json          영역 목록 {name, label, kind, px, mean_need_mm, ...}과 만든 방법
  <name>_mask.npy       uint8 512x512  보상할 영역
  <name>_target.npy     float32 512x512 (m)  영역 안의 목표 높이. 영역 밖은 기준 상태 그대로
  h_ref.npy             영역을 정의한 시점의 높이맵(m). 영역 밖 변화를 재는 기준
  h_star_all.npy        모든 영역의 목표를 합친 높이맵(그림용)
읽을 수 있는 다른 형식:
  - 예전 형식: <name>_mask.npy + <name>_target.npy 짝만 있는 폴더 (regions.json 없음; h_ref 는 ref_h 로 따로 준다)
  - live_edit.py 출력 폴더: regions/r*/mask.npy, target.npy, prompt.json (VLM + SAM2 로 만든 영역), h_t.npy

영역을 만드는 방법
  auto_regions   목표와 현재의 차이에서 직접 찾는다. 목표보다 낮은 덩어리(채울 곳), 높은 덩어리(깎을 곳)를 연결 성분으로 나눈다.
  masks_regions  이미 있는 마스크(SAM2, 손으로 그린 것)에 편집 종류(match_target/deepen/trim/connect)별 목표를 붙인다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

SIZE = 512


def border_median(a):
    return float(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))


class RegionSet:
    def __init__(self, names, masks, targets, labels=None, h_ref=None, meta=None):
        self.names = list(names)
        self.masks = {n: np.asarray(masks[n]) > 0 for n in self.names}
        self.targets = {n: np.asarray(targets[n], dtype=np.float64) for n in self.names}
        self.labels = {n: (labels or {}).get(n, n) for n in self.names}
        self.h_ref = None if h_ref is None else np.asarray(h_ref, dtype=np.float64)
        self.meta = meta or {}

    def union(self):
        u = np.zeros((SIZE, SIZE), bool)
        for m in self.masks.values():
            u |= m
        return u

    def h_star_all(self, h_base=None):
        """영역 안은 각 영역의 목표, 밖은 기준 상태."""
        base = self.h_ref if h_base is None else np.asarray(h_base, dtype=np.float64)
        out = base.copy()
        for n in self.names:
            out[self.masks[n]] = self.targets[n][self.masks[n]]
        return out

    def subset(self, names):
        missing = [n for n in names if n not in self.names]
        if missing:
            raise KeyError(f"없는 영역: {missing} (있는 영역: {self.names})")
        return RegionSet(names, self.masks, self.targets, self.labels, self.h_ref, self.meta)

    # ------------------------------------------------------------ 저장 / 읽기
    def save(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in self.names:
            np.save(out / f"{n}_mask.npy", self.masks[n].astype(np.uint8))
            np.save(out / f"{n}_target.npy", self.targets[n].astype(np.float32))
            row = dict(name=n, label=self.labels[n], px=int(self.masks[n].sum()))
            row.update(self.meta.get("regions", {}).get(n, {}))
            rows.append(row)
        if self.h_ref is not None:
            np.save(out / "h_ref.npy", self.h_ref.astype(np.float32))
            np.save(out / "h_star_all.npy", self.h_star_all().astype(np.float32))
        np.save(out / "m_union.npy", self.union().astype(np.uint8))
        (out / "regions.json").write_text(json.dumps(dict(version=1, source=self.meta.get("source"), params=self.meta.get("params"),
                                                          regions=rows), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, d, names=None, ref_h=None, ref_default=None):
        """ref_h: 기준 높이맵 파일(있으면 폴더 안의 것보다 우선). ref_default: 폴더에도 없을 때 쓸 높이맵(예: 보상 시작 상태)."""
        d = Path(d)
        if (d / "regions.json").exists():
            info = json.loads((d / "regions.json").read_text())
            order = [r["name"] for r in info["regions"]]
            labels = {r["name"]: r.get("label", r["name"]) for r in info["regions"]}
            meta = dict(source=info.get("source"), params=info.get("params"), regions={r["name"]: r for r in info["regions"]})
            masks = {n: np.load(d / f"{n}_mask.npy") for n in order}
            targets = {n: np.load(d / f"{n}_target.npy") for n in order}
            h_ref = np.load(d / "h_ref.npy") if (d / "h_ref.npy").exists() else None
        elif (d / "regions").is_dir() and any((d / "regions").glob("r*/mask.npy")):          # live_edit.py 출력
            order, masks, targets, labels = [], {}, {}, {}
            for rd in sorted((d / "regions").glob("r*"), key=lambda p: int(p.name[1:])):
                if not (rd / "mask.npy").exists():
                    continue
                pj = json.loads((rd / "prompt.json").read_text()) if (rd / "prompt.json").exists() else {}
                n = pj.get("name", rd.name)
                order.append(n)
                masks[n], targets[n] = np.load(rd / "mask.npy"), np.load(rd / "target.npy")
                labels[n] = f"{n} ({pj.get('edit_type', '')}) {pj.get('request_part', '')}".strip()
            h_ref = np.load(d / "h_t.npy") if (d / "h_t.npy").exists() else None
            meta = dict(source="live_edit", regions={})
        else:                                                                                  # 예전 형식
            order = sorted(p.name[:-len("_mask.npy")] for p in d.glob("*_mask.npy") if (d / (p.name[:-len("_mask.npy")] + "_target.npy")).exists())
            if not order:
                raise FileNotFoundError(f"영역 폴더 형식을 알 수 없습니다: {d}")
            masks = {n: np.load(d / f"{n}_mask.npy") for n in order}
            targets = {n: np.load(d / f"{n}_target.npy") for n in order}
            labels = {}
            h_ref = np.load(d / "h_ref.npy") if (d / "h_ref.npy").exists() else None
            meta = dict(source="legacy", regions={})
        if ref_h is not None:
            h_ref = np.load(ref_h)
        if h_ref is None:
            h_ref = ref_default
        if h_ref is None:
            raise ValueError(f"기준 높이맵(h_ref)이 없습니다: {d} — --ref-h 로 주세요(영역을 정의한 시점의 높이맵).")
        rs = cls(order, masks, targets, labels, h_ref, meta)
        return rs.subset(names) if names else rs


# ------------------------------------------------------------------ 만드는 방법
def auto_regions(h_cur, target, g, *, thr_mm=2.0, sigma_mm=2.0, min_area_mm2=100.0, max_regions=6, border_px=30, pad_px=2,
                 max_area_mm2=None, detrend_mm=0.0, cap_mm=8.0, within=None):
    """목표와 현재의 차이에서 영역을 직접 찾는다. 대상에 대한 규칙이 없다.

    목표 goal:  detrend_mm = 0 이면 목표(배경 높이를 현재에 맞춘 것) 그대로.
                detrend_mm > 0 이면 '현재 - 목표'의 완만한 성분(가우시안 sigma=detrend_mm)을 목표에 더해, 판 전체의 높이 차이처럼
                도구보다 훨씬 큰 규모의 차이는 오차로 보지 않는다(조형이 아직 끝나지 않은 상태에서 모양의 국소 오차만 보려는 용도).
    r = goal - 현재.  r 을 sigma_mm 로 부드럽게 한 뒤
      r >  thr_mm 인 연결 덩어리 = '채울 곳'(점토가 모자람),  r < -thr_mm 인 덩어리 = '깎을 곳'(점토가 남음).
    작은 덩어리(min_area_mm2 미만)와 맵 가장자리(border_px)는 버리고, 오차 부피(|r| 적분)가 큰 순서로 max_regions 개를 고른다.
    max_area_mm2 를 주면 그보다 큰 덩어리는 좌표 k-평균으로 쪼갠다(영역 하나가 도구 몇 개보다 커지지 않게).
    cap_mm: |r| 이 이보다 큰 곳은 국소 편집으로 만들 수 없는 차이(목표의 깊은 도랑, 아직 시작도 못 한 큰 작업)라서 후보에서 뺀다.
    within: 불리언 마스크를 주면 그 안에서만 찾는다(관심 영역).
    """
    from region_edit import region_target
    px_mm = g["px_mm"]
    h = np.asarray(h_cur, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    goal = tgt - border_median(tgt) + border_median(h)
    if detrend_mm > 0:
        goal = goal + local_offset(h - goal, detrend_mm / px_mm)
    r = ndimage.gaussian_filter((goal - h) * 1000.0, sigma_mm / px_mm)
    valid = np.zeros((SIZE, SIZE), bool)
    valid[border_px:SIZE - border_px, border_px:SIZE - border_px] = True
    valid &= np.abs(r) <= cap_mm
    if within is not None:
        valid &= np.asarray(within) > 0
    comps = []
    for kind, sign in (("fill", +1), ("cut", -1)):
        m = ndimage.binary_opening((sign * r > thr_mm) & valid, iterations=1)
        lab, n = ndimage.label(m)
        for k in range(1, n + 1):
            c = lab == k
            parts = [c]
            if max_area_mm2 and c.sum() * px_mm ** 2 > max_area_mm2:
                parts = _split(c, int(np.ceil(c.sum() * px_mm ** 2 / max_area_mm2)))
            for p in parts:
                area = float(p.sum() * px_mm ** 2)
                if area >= min_area_mm2:
                    comps.append(dict(kind=kind, core=p, area=area, volume=float(np.abs(r[p]).sum() * px_mm ** 2), mean=float(r[p].mean())))
    comps.sort(key=lambda c: -c["volume"])
    comps = comps[:max_regions]
    names, masks, targets, labels, rows, count = [], {}, {}, {}, {}, {"fill": 0, "cut": 0}
    for i, c in enumerate(comps):
        others = np.zeros((SIZE, SIZE), bool)
        for j, o in enumerate(comps):
            if j != i:
                others |= o["core"]
        m = (ndimage.binary_dilation(c["core"], iterations=pad_px) if pad_px else c["core"]) & ~others
        count[c["kind"]] += 1
        n = f"{c['kind']}{count[c['kind']]}"
        ys, xs = np.where(m)
        x, y = g["x0"] + (xs.mean() + 0.5) / g["px_per_m_x"], g["y0"] + (ys.mean() + 0.5) / g["px_per_m_y"]
        tgt_r, _ = region_target(h.astype(np.float32), m, "match_target", reference=goal.astype(np.float32))
        names.append(n)
        masks[n], targets[n] = m, tgt_r
        kind_ko = "채울 곳" if c["kind"] == "fill" else "깎을 곳"
        labels[n] = f"{kind_ko} {count[c['kind']]} (평균 {c['mean']:+.1f}mm, {c['area']:.0f}mm², x={x * 1000:.0f} y={y * 1000:.0f}mm)"
        rows[n] = dict(kind=c["kind"], mean_need_mm=c["mean"], area_mm2=c["area"], error_volume=c["volume"], center_xy_mm=[x * 1000, y * 1000])
    return RegionSet(names, masks, targets, labels, h, dict(source="auto", regions=rows, params=dict(
        thr_mm=thr_mm, sigma_mm=sigma_mm, min_area_mm2=min_area_mm2, max_regions=max_regions, border_px=border_px, pad_px=pad_px,
        max_area_mm2=max_area_mm2, detrend_mm=detrend_mm, cap_mm=cap_mm, within=None if within is None else "mask"), goal=goal))


def local_offset(d, window_px, factor=4):
    """d 의 '국소 기준면'(window_px 폭 윈도우 중앙값). 평균/가우시안과 달리 목표의 단차(판 가장자리 등)를 번지게 하지 않는다.
    계산량을 줄이려고 factor 배 줄여서 중앙값을 구하고 다시 키운다."""
    f = int(factor)
    small = d.reshape(SIZE // f, f, SIZE // f, f).mean((1, 3))
    w = max(int(round(window_px / f)) | 1, 3)
    med = ndimage.median_filter(small, size=w, mode="nearest")
    return ndimage.zoom(med, f, order=1)


def _split(comp, k):
    ys, xs = np.where(comp)
    pts = np.stack([ys, xs], 1).astype(np.float64)
    rng = np.random.default_rng(0)
    cen = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(30):
        lab = np.argmin(((pts[:, None] - cen[None]) ** 2).sum(-1), 1)
        cen = np.stack([pts[lab == i].mean(0) if (lab == i).any() else cen[i] for i in range(k)])
    out = []
    for i in range(k):
        m = np.zeros(comp.shape, bool)
        m[ys[lab == i], xs[lab == i]] = True
        out.append(m)
    return out


def masks_regions(h_cur, specs, g, *, target=None, delta_mm=3.0):
    """마스크를 이미 가진 영역들에 편집 종류별 목표를 붙인다.

    specs: [dict(name, mask(512x512 bool), edit_type, [label], [delta_mm])]
    edit_type: match_target(목표를 따름; target 필요) | deepen | deepen_widen | trim | connect | thin | sharpen  (region_edit.py 와 같은 규칙)
    """
    from region_edit import region_target
    h = np.asarray(h_cur, dtype=np.float32)
    names, masks, targets, labels, rows = [], {}, {}, {}, {}
    for s in specs:
        n = s["name"]
        ref = None if target is None else np.asarray(target, dtype=np.float32)
        t, m = region_target(h, np.asarray(s["mask"]) > 0, s.get("edit_type", "match_target"), delta_mm=s.get("delta_mm", delta_mm), reference=ref, px_mm=g["px_mm"])
        names.append(n)
        masks[n], targets[n] = m, t
        labels[n] = s.get("label", n)
        rows[n] = dict(edit_type=s.get("edit_type", "match_target"), mean_need_mm=float((t.astype(np.float64) - h)[m].mean() * 1000), area_mm2=float(m.sum() * g["px_mm"] ** 2))
    return RegionSet(names, masks, targets, labels, h, dict(source="masks", regions=rows))


# ------------------------------------------------------------------ 그림
def preview(path, rs, h_cur, target, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.asarray(h_cur, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    bg = border_median(h)
    goal = rs.meta["goal"] if rs.meta.get("goal") is not None else tgt - border_median(tgt) + bg
    ys, xs = np.where(rs.union())
    pad = 50
    r0, r1, c0, c1 = max(ys.min() - pad, 0), min(ys.max() + pad, SIZE), max(xs.min() - pad, 0), min(xs.max() + pad, SIZE)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4), facecolor="#fcfcfb")
    kw = dict(cmap="turbo", vmin=-10, vmax=14)
    ax[0].imshow(((h - bg) * 1000)[r0:r1, c0:c1], **kw); ax[0].set_title("Current (mm above background)", fontsize=9)
    ax[1].imshow(((goal - bg) * 1000)[r0:r1, c0:c1], **kw); ax[1].set_title("Target (aligned to current background)", fontsize=9)
    im = ax[2].imshow(((h - goal) * 1000)[r0:r1, c0:c1], cmap="RdBu_r", vmin=-6, vmax=6); ax[2].set_title("Current - target (mm): red = too high, blue = too low", fontsize=9)
    for a in ax:
        for n in rs.names:
            m = rs.masks[n][r0:r1, c0:c1]
            a.contour(m, levels=[0.5], colors="k", linewidths=0.8, linestyles="--")
            yy, xx = np.where(m)
            if len(yy):
                a.text(xx.mean(), yy.mean(), n, ha="center", va="center", fontsize=8, color="k", bbox=dict(fc="w", alpha=0.6, lw=0, pad=1))
        a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=95, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
