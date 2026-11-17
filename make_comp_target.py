"""⑧ 인과 효과 C* 계산과 ⑩ 보상 목표 생성 (CPU, numpy만 사용).

입력: counterfactual_replay.py 결과 폴더들(--cf DIR ...). 각 폴더에는 heightmaps.npz(h_x1=factual 재생, h_x{배율}=CF 재생)와
cf_report.json 이 있어야 한다. 재생 잡음/valid 판정이 있으면 함께 쓴다.

  C*  = H_cf - H_f          (같은 엔진에서 다시 돌린 factual 대비, 선택된 후보 하나)
  목표 = H_now + M_ROI * C* (현재 상태에 C*의 ROI 부분만 더한 목표 -> '현재 변형이 C*와 닮도록' 계획기에 전달)

후보 선택: valid 이면서 H*(수정 목표) 대비 ROI MAE가 가장 많이 줄어드는 (구간, 배율).
유효한 후보가 없으면 abstain 으로 기록하고 보상 목표를 만들지 않는다.

사용: python3 -I make_comp_target.py --cf D1 --cf D2 --h-star H.npy --h-now NOW.npy --roi ROI.npy --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", action="append", required=True, type=Path)
    ap.add_argument("--h-star", required=True, type=Path, help="수정 목표 H* (512x512 m)")
    ap.add_argument("--h-now", required=True, type=Path, help="현재(factual 최종) 높이맵")
    ap.add_argument("--roi", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-gain-mm", type=float, default=0.0, help="ROI 개선이 이 값 이하면 abstain")
    a = ap.parse_args()
    h_star = np.load(a.h_star).astype(np.float64)
    h_now = np.load(a.h_now).astype(np.float64)
    roi = np.load(a.roi).astype(bool)
    roi_now = float(np.abs(h_now - h_star)[roi].mean()) * 1000

    cands = []
    for d in a.cf:
        d = d.expanduser()
        z = np.load(d / "heightmaps.npz")
        rep = json.loads((d / "cf_report.json").read_text())
        h_f = z["h_x1"].astype(np.float64)
        for r in rep["runs"]:
            key = f"h_x{r['factor']:g}"
            h_cf = z[key].astype(np.float64)
            c = h_cf - h_f
            after = float(np.abs((h_now + roi * c) - h_star)[roi].mean()) * 1000   # C*를 현재에 더했을 때 ROI 오차
            cands.append(dict(dir=str(d), segment=f"{rep['start']}-{rep['end']}", factor=r["factor"],
                              valid=r.get("valid", None), snr=r.get("snr"),
                              roi_effect_mm=float(np.abs(c)[roi].mean()) * 1000,
                              roi_mae_if_applied_mm=after, roi_gain_mm=roi_now - after, key=key))
    ok = [c for c in cands if c["valid"] is not False and c["roi_gain_mm"] > a.min_gain_mm]
    a.out.mkdir(parents=True, exist_ok=True)
    summary = dict(roi_mae_now_vs_star_mm=roi_now, candidates=cands, abstain=not ok)
    if ok:
        best = max(ok, key=lambda c: c["roi_gain_mm"])
        z = np.load(Path(best["dir"]) / "heightmaps.npz")
        c = (z[best["key"]] - z["h_x1"]).astype(np.float32)
        target = (h_now + roi * c).astype(np.float32)
        np.save(a.out / "c_star_512_m.npy", c)
        np.save(a.out / "comp_target_512_m.npy", target)
        summary["best"] = best
    (a.out / "comp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"현재 ROI MAE (H* 대비): {roi_now:.4f}mm | 후보 {len(cands)}개")
    for c in sorted(cands, key=lambda c: -c["roi_gain_mm"]):
        print(f"  구간 {c['segment']} x{c['factor']:g}: 효과 {c['roi_effect_mm']:.4f}mm, 적용 시 ROI 오차 {c['roi_mae_if_applied_mm']:.4f}mm "
              f"(개선 {c['roi_gain_mm']:+.4f}), valid={c['valid']}, SNR={c['snr']}")
    print("=> abstain (유효한 후보 없음)" if not ok else f"=> 선택: 구간 {summary['best']['segment']} x{summary['best']['factor']:g}, 보상 목표 저장")


if __name__ == "__main__":
    main()
