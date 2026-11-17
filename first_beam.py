"""First-Beam: 저장된 조형 이력에서 '목표 변화량 D'에 영향을 준 과거 구간을 찾는다.

numpy/scipy/PIL만 사용한다(torch, Genesis 불필요). 입력은 ActionHistory가 만든 폴더의
action_XXXXXX/transition.npz, record.json 과 manifest.json, roi_mask_512.npy 이다.

핵심 아이디어
  D        = H_edit - H_T                     (현재 상태에서 바꾸고 싶은 양)
  dH       = 과거 행동(또는 5행동 구간)이 만든 실제 높이 변화
  대리 모델 : 그 행동을 s배로 바꾸면 최종 높이가 H_T + (s-1)*dH 로 바뀐다고 1차 근사
  gain(s)  = MAE(H_T, H_edit) - MAE(H_T + (s-1)dH, H_edit)   (전체 맵 기준, ROI 밖 훼손 자동 반영)

대리 모델은 후보 순위를 매기는 용도일 뿐이며, 실제 효과는 counterfactual_replay.py 의 Genesis 재생으로 확인한다.

사용 예
  python3 -I first_beam.py --history-dir runs/heart_hist --h-edit case/h_edit_512_m.npy \
      --lookback 40 --seq-len 5 --top-k 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

SIZE = 512
FACTORS = (0.7, 1.3)  # 1.0 은 원래 행동

# 점수 가중치 (합 1.0). 필요하면 여기만 고치면 된다.
WEIGHTS = dict(gain=0.40, roi_def=0.10, selectivity=0.10, overlap=0.10,
               align=0.10, dist=0.10, recent=0.10)


# ----------------------------------------------------------------------------
# 입력 로딩
# ----------------------------------------------------------------------------
def load_history(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"action_XXXXXX 폴더가 없습니다: {root}")
    actions, before, after = [], [], []
    for expected, d in enumerate(dirs, start=1):
        number = int(d.name.split("_")[1])
        if number != expected:
            raise RuntimeError(f"행동 번호가 연속적이지 않습니다: {d.name}")
        z = np.load(d / "transition.npz")
        rec = json.loads((d / "record.json").read_text())
        actions.append(np.asarray(rec["action"], dtype=np.float64))
        before.append(z["before_m"].astype(np.float32))
        after.append(z["after_m"].astype(np.float32))
    return manifest, np.stack(actions), np.stack(before), np.stack(after)


def map_geometry(manifest):
    ax = manifest["map_axes"]
    x0, x1 = ax["x_bounds"]
    y0, y1 = ax["y_bounds"]
    settings = manifest["signature"]["details"]["settings"]
    tool_r = float(settings.get("TOOL_RADIUS", 0.015))
    return dict(x0=x0, x1=x1, y0=y0, y1=y1, tool_r=tool_r,
                px_per_m_x=SIZE / (x1 - x0), px_per_m_y=SIZE / (y1 - y0))


def world_to_pixel(g, x, y):
    col = (np.asarray(x) - g["x0"]) * g["px_per_m_x"] - 0.5
    row = (np.asarray(y) - g["y0"]) * g["px_per_m_y"] - 0.5
    return row, col


def path_points(g, action, n=24):
    x, y, th, length, _ = action
    t = np.linspace(0.0, 1.0, n)
    xs = x + t * length * math.cos(th)
    ys = y + t * length * math.sin(th)
    return world_to_pixel(g, xs, ys)


def footprint(g, action):
    """도구 반경을 가진 캡슐 모양 접촉 영역(불리언 512x512)."""
    rows, cols = path_points(g, action)
    radius = g["tool_r"] * g["px_per_m_x"]
    rr, cc = np.mgrid[0:SIZE, 0:SIZE]
    best = np.full((SIZE, SIZE), np.inf, dtype=np.float32)
    for r, c in zip(rows, cols):
        best = np.minimum(best, np.hypot(rr - r, cc - c).astype(np.float32))
    return best <= radius


# ----------------------------------------------------------------------------
# 특징량
# ----------------------------------------------------------------------------
class Scorer:
    def __init__(self, g, roi, h_final, h_edit):
        self.g = g
        self.roi = roi.astype(bool)
        self.outside = ~self.roi
        self.h_final = h_final
        self.h_edit = h_edit
        self.D = (h_edit - h_final).astype(np.float32)
        self.D_roi_norm = float(np.linalg.norm(self.D[self.roi])) + 1e-12
        # ROI까지의 거리(px). ROI 안은 0.
        self.dist_to_roi = ndimage.distance_transform_edt(~self.roi).astype(np.float32)
        self._fp_cache = {}
        self.err0 = float(np.abs(self.D).mean())
        self.roi_err0 = float(np.abs(self.D[self.roi]).mean())

    def surrogate(self, dH):
        """factor별 전체/ROI MAE 개선량(mm)과 최적 factor."""
        best = dict(factor=1.0, gain_mm=0.0, roi_gain_mm=0.0)
        table = {}
        for s in FACTORS:
            resid = self.D - (s - 1.0) * dH
            err = float(np.abs(resid).mean())
            roi_err = float(np.abs(resid[self.roi]).mean())
            gain = (self.err0 - err) * 1000.0
            roi_gain = (self.roi_err0 - roi_err) * 1000.0
            table[f"{s:g}"] = dict(gain_mm=gain, roi_gain_mm=roi_gain)
            if gain > best["gain_mm"]:
                best = dict(factor=s, gain_mm=gain, roi_gain_mm=roi_gain)
        best["table"] = table
        return best

    def features(self, dH, actions_in_item, age_from_end, lookback):
        roi, out = self.roi, self.outside
        roi_def = float(np.abs(dH[roi]).sum())
        out_def = float(np.abs(dH[out]).sum())
        selectivity = roi_def / (roi_def + out_def + 1e-12)
        fp = np.zeros((SIZE, SIZE), dtype=bool)
        dists = []
        for a in actions_in_item:
            key = tuple(np.round(a, 9))
            if key not in self._fp_cache:
                self._fp_cache[key] = footprint(self.g, a)
            fp |= self._fp_cache[key]
            rows, cols = path_points(self.g, a)
            rr = np.clip(np.round(rows).astype(int), 0, SIZE - 1)
            cc = np.clip(np.round(cols).astype(int), 0, SIZE - 1)
            dists.append(float(self.dist_to_roi[rr, cc].min()))
        overlap = float((fp & roi).sum() / max(fp.sum(), 1))
        dist_px = float(np.mean(dists))
        n_roi = float(np.linalg.norm(dH[roi])) + 1e-12
        align = float((dH[roi] * self.D[roi]).sum() / (n_roi * self.D_roi_norm))
        sur = self.surrogate(dH)
        recent = math.exp(-age_from_end / max(lookback / 2.0, 1.0))
        return dict(roi_def_m_px=roi_def, outside_def_m_px=out_def, selectivity=selectivity,
                    overlap=overlap, dist_px=dist_px, align=align, recent=recent,
                    surrogate=sur)


def normalize_scores(items):
    """각 항목을 구간 내 최대값으로 나눠 0~1로 만들고 가중합을 계산한다."""
    gains = np.array([max(it["f"]["surrogate"]["gain_mm"], 0.0) for it in items])
    roi_def = np.array([it["f"]["roi_def_m_px"] for it in items])
    gmax = gains.max() if gains.max() > 0 else 1.0
    rmax = roi_def.max() if roi_def.max() > 0 else 1.0
    for it, g, r in zip(items, gains, roi_def):
        f = it["f"]
        parts = dict(gain=g / gmax, roi_def=r / rmax, selectivity=f["selectivity"],
                     overlap=f["overlap"], align=abs(f["align"]),
                     dist=math.exp(-f["dist_px"] / 25.0), recent=f["recent"])
        it["parts"] = parts
        it["score"] = float(sum(WEIGHTS[k] * parts[k] for k in WEIGHTS))


# ----------------------------------------------------------------------------
# 빔 탐색
# ----------------------------------------------------------------------------
def first_beam(root, h_edit, *, lookback, seq_len, top_k, roi_path=None):
    manifest, actions, before, after = load_history(root)
    T = len(actions)
    g = map_geometry(manifest)
    roi = np.load(roi_path if roi_path else root / "roi_mask_512.npy").astype(bool)
    h_final = after[-1]
    sc = Scorer(g, roi, h_final, h_edit)

    first = max(0, T - lookback)  # 0-based 인덱스, 포함
    # 1) 행동 단위 특징
    singles = []
    for t in range(first, T):
        dH = after[t] - before[t]
        f = sc.features(dH, [actions[t]], T - 1 - t, lookback)
        singles.append(dict(t=t, number=t + 1, action=actions[t].tolist(), f=f))
    normalize_scores(singles)
    by_t = {s["t"]: s for s in singles}

    # 2) seq_len 행동 구간 특징 (겹치는 슬라이딩 윈도우)
    seqs = []
    for s in range(first, T - seq_len + 1):
        e = s + seq_len - 1
        dH = after[e] - before[s]
        f = sc.features(dH, [actions[k] for k in range(s, e + 1)], T - 1 - e, lookback)
        seqs.append(dict(start=s + 1, end=e + 1, f=f))
    if not seqs:
        raise RuntimeError("lookback 안에 seq_len 만큼의 연속 행동이 없습니다.")
    normalize_scores(seqs)

    # 3) 겹치지 않는 Top-K (비최대 억제)
    chosen = []
    for it in sorted(seqs, key=lambda x: -x["score"]):
        if all(it["end"] < c["start"] or it["start"] > c["end"] for c in chosen):
            chosen.append(it)
        if len(chosen) == top_k:
            break

    beams = []
    for rank, it in enumerate(chosen):
        members = [by_t[k - 1] for k in range(it["start"], it["end"] + 1) if (k - 1) in by_t]
        members = sorted(members, key=lambda m: -m["score"])
        sur = it["f"]["surrogate"]
        beams.append(dict(
            rank=rank + 1, start_action=it["start"], end_action=it["end"],
            score=it["score"], score_parts=it["parts"],
            overlap=it["f"]["overlap"], dist_px=it["f"]["dist_px"],
            align=it["f"]["align"], selectivity=it["f"]["selectivity"],
            surrogate_best_factor=sur["factor"],
            surrogate_gain_mm=sur["gain_mm"], surrogate_roi_gain_mm=sur["roi_gain_mm"],
            surrogate_table=sur["table"],
            actions_ranked=[dict(action_number=m["number"], score=m["score"],
                                 action=m["action"],
                                 surrogate_best_factor=m["f"]["surrogate"]["factor"],
                                 surrogate_gain_mm=m["f"]["surrogate"]["gain_mm"])
                            for m in members]))
    return dict(schema=1, n_actions=T, lookback=lookback, seq_len=seq_len, top_k=top_k,
                weights=WEIGHTS, factors=list(FACTORS),
                baseline=dict(global_mae_mm=sc.err0 * 1000, roi_mae_mm=sc.roi_err0 * 1000),
                beams=beams, all_sequences=[dict(start=s["start"], end=s["end"], score=s["score"])
                                            for s in seqs]), (sc, actions, after, chosen)


def save_preview(path, sc, actions, chosen, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(sc.h_final * 1000, cmap="viridis")
    ax[0].set_title("H_T (mm)")
    v = max(float(np.abs(sc.D).max()) * 1000, 1e-6)
    ax[1].imshow(sc.D * 1000, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[1].contour(sc.roi, levels=[0.5], colors="k", linewidths=0.8)
    ax[1].set_title("D = H_edit - H_T (mm)")
    ax[2].imshow(sc.h_final * 1000, cmap="gray")
    ax[2].contour(sc.roi, levels=[0.5], colors="y", linewidths=0.8)
    colors = ["red", "orange", "cyan", "magenta", "lime"]
    for i, it in enumerate(chosen):
        for k in range(it["start"] - 1, it["end"]):
            r, c = path_points(sc.g, actions[k])
            ax[2].plot(c, r, color=colors[i % len(colors)], lw=2)
        ax[2].plot([], [], color=colors[i % len(colors)],
                   label=f"beam{i+1}: a{it['start']}-{it['end']}")
    ax[2].legend(loc="lower right", fontsize=8)
    ax[2].set_title("First-Beam 후보 경로")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", required=True, type=Path)
    p.add_argument("--h-edit", required=True, type=Path, help="512x512 .npy, 단위 m")
    p.add_argument("--roi", type=Path, help="기본값: history-dir/roi_mask_512.npy")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", type=Path, help="기본값: history-dir/first_beam.json")
    args = p.parse_args()
    root = args.history_dir.expanduser().resolve()
    h_edit = np.load(args.h_edit.expanduser(), allow_pickle=False).astype(np.float32)
    if h_edit.shape != (SIZE, SIZE) or not np.isfinite(h_edit).all():
        raise ValueError("H_edit는 유효한 512x512 배열이어야 합니다.")
    result, (sc, actions, after, chosen) = first_beam(
        root, h_edit, lookback=args.lookback, seq_len=args.seq_len, top_k=args.top_k,
        roi_path=args.roi.expanduser() if args.roi else None)
    out = args.out or (root / "first_beam.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    save_preview(out.with_suffix(".png"), sc, actions, chosen, result["n_actions"])
    print(f"행동 {result['n_actions']}개, 현재 오차: 전체 {result['baseline']['global_mae_mm']:.4f} mm, "
          f"ROI {result['baseline']['roi_mae_mm']:.4f} mm")
    for b in result["beams"]:
        print(f"[beam{b['rank']}] 행동 {b['start_action']}~{b['end_action']} 점수={b['score']:.3f} "
              f"겹침={b['overlap']:.2f} 거리={b['dist_px']:.0f}px 정렬={b['align']:+.2f} "
              f"대리 최적 배율={b['surrogate_best_factor']} "
              f"예상 ROI 개선={b['surrogate_roi_gain_mm']:+.4f} mm")
    print(f"저장: {out}, {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
