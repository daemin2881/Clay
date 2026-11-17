"""80번째 점토 상태에서 과거 행동을 하나씩 바꾼 것 / 한꺼번에 바꾼 것 / 하나씩의 효과를 단순히 더한 것을 같은 기준으로 비교한다.

모두 '현재 = 80번째 상태 S_80' 에 대해, 수정한 과거를 끝(80)까지 재생한 결과의 변화 C = H_cf - H_control 을 S_80 에 더해 비교한다(재생 드리프트 상쇄).
  영역 안 목표 쪽 이동 t : sum[(final-S80)(H*-S80)] / sum[(H*-S80)^2]   (세 영역을 합친 마스크 안, 1.0 = 목표 도달)
  MAE(원래 목표 대비)    : 전체 지도, A 글자 영역, 세 영역 각각 (원래 A 목표를 배경 높이에 정렬)
  형태 L3D / 외형 Lviz / 테두리 : S_80 = 1.00 인 비율 (합친 영역 주변 창)
  상호작용 : '하나씩 더한 것'과 '한꺼번에' 의 영역 안 평균 변화 차이
사용: python compare_s80.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import numpy as np

from region_edit import _bg
from visual_score import VisualScorer

D = Path("results/case_A/design")
H80 = np.load(D / "s80_h_t.npy").astype(np.float64)
HALL = np.load(D / "s80/h_star_all.npy").astype(np.float64)
MU = np.load(D / "s80/m_union.npy") > 0
REF = np.load("results/A_history200_20261114_060837/history/target_used_512_m.npy").astype(np.float64)
REF_AL = REF - _bg(REF) + _bg(H80)
FP = REF > _bg(REF) + 5e-4
MASKS = [np.load(D / f"s80/r{i}_mask.npy") > 0 for i in range(3)]
NAMES = ["튀어나온 부분", "끊어진 다리", "가운데 홈"]
SC = VisualScorer(HALL.astype(np.float32), H80.astype(np.float32), MU)


def load_single(start):
    d = D / f"s80_cf/single{start}"
    c, f = d / f"s{start}_control.npy", d / f"s{start}_skip.npy"
    if c.exists() and f.exists():
        return np.load(f).astype(np.float64) - np.load(c).astype(np.float64)
    return None


def load_combined():
    d = D / "s80_cf/combined"
    if (d / "combined.npy").exists():
        return np.load(d / "combined.npy").astype(np.float64) - np.load(d / "combined_control.npy").astype(np.float64)
    return None


def metrics(C):
    final = H80 + C
    mae = lambda t, m=None: float(np.abs(final - t)[m if m is not None else slice(None)].mean()) * 1000
    den = float(((HALL - H80)[MU] ** 2).sum()) + 1e-12
    r = SC.ratios(final.astype(np.float32))
    return dict(toward=float(((final - H80)[MU] * (HALL - H80)[MU]).sum() / den),
                mae_all=mae(REF_AL), mae_A=mae(REF_AL, FP), mae_regions=[mae(REF_AL, m) for m in MASKS],
                mae_union=mae(REF_AL, MU), l3d=r["l3d_win"], lviz=r["lviz_win"], edge=r["edge_win"], lviz_full=r["lviz_full"],
                far_max=float(np.abs(C).max()) * 1000, n_changed=int((np.abs(C) * 1000 > 0.5).sum()),
                mean_in_union=float(C[MU].mean()) * 1000)


def main():
    rows = {"S80 (아무것도 안 함)": metrics(np.zeros_like(H80))}
    singles = {}
    for s in (59, 62, 71, 75):
        C = load_single(s)
        if C is not None:
            singles[s] = C
            rows[f"하나씩: #{s} 건너뛰기"] = metrics(C)
    if singles:
        rows[f"하나씩 효과의 합 ({'+'.join('#' + str(k) for k in singles)})"] = metrics(sum(singles.values()))
    Cc = load_combined()
    if Cc is not None:
        rows["한꺼번에 (#71,#75,#62,#59)"] = metrics(Cc)
    print(f"{'':<34}{'영역 목표쪽':>10}{'MAE 전체':>9}{'MAE A영역':>10}{'MAE 합친영역':>12}{'  튀어나온/다리/홈':>22}{'  L3D':>7}{'Lviz':>7}{'테두리':>7}{'변화화소':>9}{'최대mm':>8}")
    for n, m in rows.items():
        print(f"{n:<34}{m['toward']:>+10.2f}{m['mae_all']:>9.3f}{m['mae_A']:>10.3f}{m['mae_union']:>12.3f}   "
              f"{m['mae_regions'][0]:>5.2f}/{m['mae_regions'][1]:.2f}/{m['mae_regions'][2]:.2f}{m['l3d']:>7.2f}{m['lviz']:>7.2f}{m['edge']:>7.2f}{m['n_changed']:>9d}{m['far_max']:>8.1f}")
    if singles and Cc is not None:
        Cs = sum(singles.values())
        diff = (Cc - Cs)
        inter = (f"합친 영역 안 평균 변화: 하나씩의 합 {Cs[MU].mean() * 1000:+.2f}mm, 한꺼번에 {Cc[MU].mean() * 1000:+.2f}mm "
                 f"(차이 {diff[MU].mean() * 1000:+.2f}mm). 전체 지도에서 |한꺼번에 - 하나씩의 합| 평균 {np.abs(diff).mean() * 1000:.3f}mm, 최대 {np.abs(diff).max() * 1000:.1f}mm")
        print("\n상호작용:", inter)
        rows["_interaction"] = inter
    (D / "s80_compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    # 그림
    ls = LightSource(azdeg=315, altdeg=40)
    sh = lambda h: ls.hillshade(h * 1000, vert_exag=2, dx=360 / 511, dy=360 / 511)
    ys, xs = np.where(FP)
    r0, r1, c0, c1 = max(ys.min() - 20, 0), min(ys.max() + 20, 512), max(xs.min() - 20, 0), min(xs.max() + 20, 512)
    pan = [("Original target A", REF_AL), ("State after 80 actions", H80)]
    best_single = max(singles, key=lambda k: metrics(singles[k])["toward"]) if singles else None
    if best_single:
        pan.append((f"One at a time: best (#{best_single} skipped)", H80 + singles[best_single]))
    if singles:
        pan.append(("Sum of the one-at-a-time effects", H80 + sum(singles.values())))
    if Cc is not None:
        pan.append(("All at once (#71,#75,#62,#59 skipped)", H80 + Cc))
    fig, axs = plt.subplots(1, len(pan), figsize=(3.6 * len(pan), 4.6), facecolor="#fcfcfb")
    for ax, (t, h) in zip(np.atleast_1d(axs), pan):
        ax.imshow(sh(h)[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1)
        for m in MASKS:
            ax.contour(m[r0:r1, c0:c1], levels=[0.5], colors="#e34948", linewidths=0.8, linestyles="--")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("State after 80 actions: change past actions one at a time vs all at once (dashed = bump, gap, groove)", fontsize=11)
    fig.tight_layout()
    fig.savefig(D / "figs/s80_compare.png", dpi=105, facecolor="#fcfcfb")
    print("그림:", D / "figs/s80_compare.png")


if __name__ == "__main__":
    main()
