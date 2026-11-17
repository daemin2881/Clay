"""80번째 점토 상태에서 요청 사항(홈 파기, 왼쪽 다리 얇고 뚜렷하게, 오른쪽 다리 끝과 받침 살리기)을 원래 목표 A 의 모양에서 직접 영역으로 나누고,
영역별 목표(match_target)와, 이미 재생해 둔 과거 수정(건너뛰기)들이 각 영역에 준 효과를 계산한다. SAM2/VLM 이 아니라 목표 형상에서 규칙으로 영역을 만든다.

  groove     : 각 행에서 A 의 왼쪽 끝~오른쪽 끝 사이를 채운 뒤 A 자체를 뺀 빈 공간 (위쪽 삼각 홈, 가로획 아래 열린 홈)
  left_leg   : 받침 행을 뺀, 2개 이상의 덩어리가 있는 행의 가장 왼쪽 덩어리 (A 의 왼쪽 획)
  right_leg  : 같은 행의 가장 오른쪽 덩어리 (오른쪽 획)
  left_foot / right_foot : 맨 아래 14% 행의 왼쪽/오른쪽 덩어리 (다리 끝 받침 막대)
사용: python s80_targets.py [--rank]   -> results/case_A/design/s80b/ 에 마스크·목표 저장, 표 출력
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from region_edit import _bg, region_target
from visual_score import VisualScorer

D = Path("results/case_A/design")
H = Path("results/A_history200_20261114_060837/history")
OUT = D / "s80b"
NAMES = ["groove", "left_leg", "right_leg", "left_foot", "right_foot"]
KO = {"groove": "A 사이의 홈", "left_leg": "왼쪽 다리", "right_leg": "오른쪽 다리", "left_foot": "왼쪽 다리 끝 받침", "right_foot": "오른쪽 다리 끝 받침"}


def build_masks(ref):
    F = ref > _bg(ref) + 5e-4
    rows = np.where(F.any(1))[0]
    r0, r1 = rows.min(), rows.max()
    foot_row0 = int(r1 - 0.14 * (r1 - r0))
    M = {n: np.zeros(F.shape, bool) for n in NAMES}
    for r in range(r0, r1 + 1):
        lab, n = ndimage.label(F[r][None, :])
        if n >= 2:
            cols = np.where(F[r])[0]
            runs = [np.where(lab[0] == k)[0] for k in range(1, n + 1)]
            span = np.zeros(F.shape[1], bool)
            span[cols.min():cols.max() + 1] = True
            M["groove"][r] = span & ~F[r]
            if r >= foot_row0:
                M["left_foot"][r, runs[0]] = True
                M["right_foot"][r, runs[-1]] = True
            else:
                M["left_leg"][r, runs[0]] = True
                M["right_leg"][r, runs[-1]] = True
    for n in NAMES:
        M[n] = ndimage.binary_dilation(M[n], iterations=2) if n != "groove" else M[n]
    M["groove"] = ndimage.binary_opening(M["groove"], iterations=2)
    return M, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h80 = np.load(D / "s80_h_t.npy").astype(np.float64)
    ref = np.load(H / "target_used_512_m.npy").astype(np.float64)
    ref_al = ref - _bg(ref) + _bg(h80)
    M, F = build_masks(ref)
    tg, allm = {}, np.zeros(F.shape, bool)
    for n in NAMES:
        tg[n], applied = region_target(h80.astype(np.float32), M[n], "match_target", reference=ref.astype(np.float32))
        np.save(OUT / f"{n}_mask.npy", M[n].astype(np.uint8))
        np.save(OUT / f"{n}_target.npy", tg[n])
        allm |= M[n]
    hall = h80 + sum(tg[n].astype(np.float64) - h80 for n in NAMES)
    np.save(OUT / "h_star_all.npy", hall.astype(np.float32))
    np.save(OUT / "m_union.npy", allm.astype(np.uint8))
    print("80번째 상태에서 원래 목표(배경 정렬) 대비 영역별 MAE(mm), 필요한 평균 변화(mm):")
    for n in NAMES:
        m = M[n]
        print(f"  {KO[n]:<14} 화소 {int(m.sum()):>5}  MAE {np.abs(ref_al - h80)[m].mean() * 1000:5.2f}  필요 변화 {((ref_al - h80)[m].mean() * 1000):+6.2f}")
    # 이미 재생해 둔 과거 수정들의 영역별 효과
    arms = {}
    for s in (59, 62, 71, 75):
        d = D / f"s80_cf/single{s}"
        if (d / f"s{s}_skip.npy").exists():
            arms[f"#{s} 건너뛰기"] = np.load(d / f"s{s}_skip.npy").astype(np.float64) - np.load(d / f"s{s}_control.npy").astype(np.float64)
    for s_ in (66, 70, 76, 78, 73, 68, 57):
        d = D / f"s80b_cf/a{s_}"
        if (d / f"s{s_}_skip.npy").exists():
            arms[f"#{s_} 건너뛰기"] = np.load(d / f"s{s_}_skip.npy").astype(np.float64) - np.load(d / f"s{s_}_control.npy").astype(np.float64)
    dc = D / "s80_cf/combined"
    if (dc / "combined.npy").exists():
        arms["한꺼번에(#59,#62,#71,#75)"] = np.load(dc / "combined.npy").astype(np.float64) - np.load(dc / "combined_control.npy").astype(np.float64)
    print("\n과거 수정별 영역 MAE(mm, 원래 목표 대비). 80번째 기준값 -> 수정 후 (부호: 줄면 개선)")
    hdr = f"{'':<26}" + "".join(f"{KO[n][:7]:>12}" for n in NAMES)
    print(hdr)
    print(f"{'80번째(기준)':<24}" + "".join(f"{np.abs(ref_al - h80)[M[n]].mean() * 1000:>12.2f}" for n in NAMES))
    res = {}
    for k, C in arms.items():
        fin = h80 + C
        row = {n: float(np.abs(ref_al - fin)[M[n]].mean()) * 1000 for n in NAMES}
        res[k] = row
        base = {n: float(np.abs(ref_al - h80)[M[n]].mean()) * 1000 for n in NAMES}
        print(f"{k:<24}" + "".join(f"{row[n]:>7.2f}({row[n] - base[n]:+.2f})" for n in NAMES))
    res["_base"] = {n: float(np.abs(ref_al - h80)[M[n]].mean()) * 1000 for n in NAMES}
    res["_need"] = {n: float((ref_al - h80)[M[n]].mean()) * 1000 for n in NAMES}
    (OUT / "past_effects.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    base_all = {n: float(np.abs(ref_al - h80)[M[n]].mean()) * 1000 for n in NAMES}
    best = {}
    for n in NAMES:
        cand = [(k, v[n] - base_all[n]) for k, v in res.items() if k.startswith("#")]
        if cand:
            k, d_ = min(cand, key=lambda x: x[1])
            best[n] = (k, d_)
    print("\n영역별로 가장 개선되는 과거 수정 (MAE 변화 mm, 음수 = 개선):")
    for n, (k, d_) in best.items():
        print(f"  {KO[n]:<14}: {k} ({d_:+.2f}mm){'  <- 개선' if d_ < -0.05 else '  (뚜렷한 개선 없음)'}")
    (OUT / "best_per_region.json").write_text(json.dumps({n: dict(action=k, delta_mae=d_) for n, (k, d_) in best.items()}, ensure_ascii=False, indent=1))
    if a.rank:
        import first_beam as fb
        man, actions, before, after = fb.load_history(H)
        g = fb.map_geometry(man)
        print("\n영역별 후보 행동 (56~80번째 중, 그 영역 근처(10화소)를 지나며 영역 안 높이를 크게 바꾼 순):")
        for n in NAMES:
            near = ndimage.binary_dilation(M[n], iterations=10)
            rows = []
            for t in range(56, 81):
                fp = fb.footprint(g, actions[t - 1])
                if (fp & near).any():
                    rows.append((t, float(np.abs((after[t - 1] - before[t - 1]) * 1000)[M[n]].mean()), 80 - t + 1))
            rows.sort(key=lambda r: -r[1])
            print(f"  {KO[n]:<14}: " + ", ".join(f"#{r[0]}({r[1]:.2f}mm, 재생{r[2]})" for r in rows[:4]))


if __name__ == "__main__":
    main()
