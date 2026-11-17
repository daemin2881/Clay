"""VLM 이 만든 '조건(라벨)'을 과거 기록의 물리량으로 보정해 숫자로 바꾼다. 대상(A, 하트, ...)에 의존하지 않는다.

역할 분담
  VLM  : 프롬프트 -> 조건 JSON (숫자 없이 라벨만): 영역, 편집 종류, 정도(slight/moderate/strong = 조금/보통/많이), 보존할 이웃과 강도(strong/normal/loose)
  SAM2 : 영역 마스크 (cf_edit.py --regions DIR 로 이미 들어옴)
  물리 : 라벨 -> mm.  숫자는 사람이 고른 상수가 아니라 이 대상의 과거 기록에서 측정한다.
           unit_mm   = 그 영역에 영향을 준 과거 행동들이 영역 평균 높이를 바꾼 크기(중앙값, mm)   -> "한 번 누를 때의 영역 변화 척도"
           eff       = 영역 변화 / 영역 주변 고리 변화 (중앙값)                                       -> "영역을 바꿀 때 주변이 같이 바뀌는 정도"
           delta_mm  = 정도 배율 x unit_mm                                                            (깊게 편집의 깊이)
           ring_budget_mm = delta_mm / eff x 보존 배율(strong 0.5, normal 1, loose 2)                 (cf_compensate --ring-budget-mm)
  그 뒤 검증과 확정은 cf_compensate.py (후보 -> Genesis 실행 -> 점수, 과거 국소 보상) 가 한다.

조건 JSON 형식
  {"request": "...", "items": [
     {"region": "groove_top", "edit": "deepen", "magnitude": "moderate",
      "preserve": [{"region": "left_leg", "strength": "strong"}, {"region": "left_foot", "strength": "normal"}]},
     {"region": "left_leg", "edit": "match_target"} ]}
  region 은 --regions 폴더의 영역 이름. edit: deepen | match_target | trim | connect | thin | sharpen. magnitude 는 deepen·thin 에만(thin 은 영역 두께의 15/25/35%). 한국어 라벨(조금/보통/많이, 강하게/보통/약하게)도 허용.

사용: python conditions.py --history-dir H --upto 80 --regions DIR --conditions cond.json --out spec.json [--top 3]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import cf_common as cc
import first_beam as fb
from region_set import RegionSet
from region_edit import thickness_mm as cc_thickness

MAG = {"slight": 0.5, "moderate": 1.0, "strong": 1.5}
PRES = {"strong": 0.5, "normal": 1.0, "loose": 2.0}
KO = {"조금": "slight", "보통": "moderate", "많이": "strong", "강하게": "strong", "약하게": "loose"}
EDITS = ("deepen", "match_target", "trim", "connect", "thin", "sharpen")
THIN_FRAC = {"slight": 0.15, "moderate": 0.25, "strong": 0.35}      # thin 의 정도 = 영역 두께의 몇 %를 줄일지 (이 비율은 제가 정한 값)


def norm_label(v, table, what):
    if v is None:
        return None
    v = KO.get(v, v)
    if v == "normal" and table is MAG:
        v = "moderate"
    if v == "moderate" and table is PRES:
        v = "normal"
    if v not in table:
        raise ValueError(f"{what} 라벨은 {sorted(table)} 중 하나여야 합니다: {v!r}")
    return v


def load_conditions(path, names):
    d = json.loads(Path(path).read_text())
    items = []
    for it in d["items"]:
        if it["region"] not in names:
            raise ValueError(f"영역 '{it['region']}' 이 영역 폴더에 없습니다 (있는 것: {names})")
        if it.get("edit", "match_target") not in EDITS:
            raise ValueError(f"편집 종류는 {EDITS} 중 하나: {it.get('edit')}")
        pres = [dict(region=p["region"], strength=norm_label(p.get("strength", "normal"), PRES, "보존 강도")) for p in it.get("preserve", [])]
        items.append(dict(region=it["region"], edit=it.get("edit", "match_target"), magnitude=norm_label(it.get("magnitude"), MAG, "정도"), preserve=pres))
    return d.get("request", ""), items


def calibrate(history_dir, upto, masks, top=3, near_tools=0.5, thr_mm=0.15, ring_tools=0.94):
    """영역마다 (unit_mm, eff, 근거 행동들). 과거 행동 중 영역 근처를 지났거나 영역 높이를 thr 이상 바꾼 것들 중 영역 변화가 큰 top 개를 쓴다."""
    root = Path(history_dir).resolve()
    view = cc.make_view(root, upto, Path("/tmp") / f"cond_view_{root.name}") if upto else root
    g = cc.geometry(root)
    r_tool = g["tool_r_px"]
    man, actions, before, after = fb.load_history(view)
    fps = [fb.footprint(g, actions[t]) for t in range(len(actions))]
    out = {}
    for n, m in masks.items():
        near = ndimage.binary_dilation(m, iterations=int(round(near_tools * r_tool)))
        ring = ndimage.binary_dilation(m, iterations=int(round(ring_tools * r_tool))) & ~m
        rows = []
        for t in range(len(actions)):
            d = (after[t] - before[t]).astype(np.float64) * 1000.0
            reg = float(d[m].mean())
            if (fps[t] & near).any() or abs(reg) >= thr_mm:
                rows.append((t + 1, reg, float(np.abs(d[ring]).mean())))
        rows.sort(key=lambda r: -abs(r[1]))
        use = rows[:top]
        if not use:
            out[n] = dict(unit_mm=None, eff=None, evidence=[], thickness_mm=cc_thickness(m, g["px_mm"]))
            continue
        unit = float(np.median([abs(r[1]) for r in use]))
        thick = cc_thickness(m, g["px_mm"])
        eff = float(np.median([abs(r[1]) / max(r[2], 1e-3) for r in use]))
        out[n] = dict(unit_mm=unit, eff=eff, thickness_mm=thick, evidence=[dict(action=r[0], region_change_mm=r[1], ring_change_mm=r[2]) for r in use])
    return out


def to_spec(items, cal):
    spec = []
    for it in items:
        c = cal[it["region"]]
        strengths = [p["strength"] for p in it["preserve"]] or ["normal"]
        pres_factor = min(PRES[s] for s in strengths)          # 가장 강한 보존 요구를 따른다
        if c["unit_mm"] is None:
            spec.append(dict(region=it["region"], edit=it["edit"], note="영역에 영향 준 과거 행동이 없어 보정 불가"))
            continue
        if it["edit"] == "thin" and it["magnitude"]:
            delta = THIN_FRAC[it["magnitude"]] * c["thickness_mm"]          # 총 폭 감소(mm)
        else:
            delta = MAG[it["magnitude"]] * c["unit_mm"] if it["edit"] in ("deepen",) and it["magnitude"] else None
        scale = delta if delta is not None else c["unit_mm"]
        budget = scale / c["eff"] * pres_factor
        spec.append(dict(region=it["region"], edit=it["edit"], magnitude=it["magnitude"], delta_mm=delta, ring_budget_mm=budget, unit_mm=c["unit_mm"], eff=c["eff"],
                         preserve=it["preserve"], preserve_factor=pres_factor, evidence=c["evidence"]))
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--conditions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    rs = RegionSet.load(a.regions, ref_default=np.zeros((512, 512)))
    request, items = load_conditions(a.conditions, rs.names)
    cal = calibrate(a.history_dir, a.upto, {n: rs.masks[n] for n in {it["region"] for it in items}}, top=a.top)
    spec = to_spec(items, cal)
    a.out.write_text(json.dumps(dict(request=request, spec=spec), ensure_ascii=False, indent=1))
    print(f"요청: {request}")
    print(f"{'영역':<14}{'편집':<13}{'정도':<10}{'unit(mm)':>9}{'eff':>6}{'delta(mm)':>11}{'주변 허용(mm)':>14}  보존")
    for s in spec:
        if s.get("unit_mm") is None:
            print(f"{s['region']:<14}{s['edit']:<13} {s['note']}")
            continue
        pv = ", ".join(f"{p['region']}:{p['strength']}" for p in s["preserve"]) or "-"
        print(f"{s['region']:<14}{s['edit']:<13}{str(s['magnitude'] or '-'):<10}{s['unit_mm']:>9.2f}{s['eff']:>6.1f}{(('%.2f' % s['delta_mm']) if s['delta_mm'] else '-'):>11}{s['ring_budget_mm']:>14.2f}  {pv}")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
