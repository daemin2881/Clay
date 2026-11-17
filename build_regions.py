"""조형 이력의 K번째 상태에서 '고칠 영역 집합'을 만든다. 어떤 대상, 어떤 시점이든 같은 방식이다 (GPU/Genesis 불필요).

  auto   목표와 현재의 차이에서 영역을 직접 찾는다 (채울 곳 fill1.., 깎을 곳 cut1..).
  masks  이미 있는 마스크(SAM2 출력, 손으로 그린 것)에 편집 종류별 목표를 붙인다.
         --mask 이름=경로.npy[:편집종류[:delta_mm]]   편집종류: match_target(기본, 목표를 따름) deepen deepen_widen trim connect thin(폭 delta_mm 줄이기) sharpen(윤곽 선명하게)
  (VLM + SAM2 로 영역을 만드는 길은 live_edit.py --stop-after spec 의 출력 폴더를 cf_edit.py --regions 에 그대로 주면 된다.)

사용:
  python build_regions.py auto  --history-dir H [--upto 80] [--target-npy T.npy] --out DIR [--thr-mm 2 --max-regions 6 ...]
  python build_regions.py masks --history-dir H [--upto 80] --out DIR --mask groove=m.npy:deepen:3 --mask bump=b.npy:trim
출력: DIR/regions.json, <name>_mask.npy, <name>_target.npy, h_ref.npy, h_star_all.npy, regions_preview.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import cf_common as cc
from region_set import RegionSet, auto_regions, masks_regions, preview


def common(a):
    hist = cc.make_view(a.history_dir, a.upto, a.out / "history_view")
    h_cur = cc.state_height(hist)
    target = cc.load_target(hist, a.target_npy)
    g = cc.geometry(hist)
    return hist, h_cur, target, g


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("auto", "masks"):
        p = sub.add_parser(name)
        p.add_argument("--history-dir", required=True, type=Path)
        p.add_argument("--upto", type=int, help="조형 K번째 상태를 현재로 삼는다(없으면 마지막)")
        p.add_argument("--target-npy", type=Path, help="새 목표(512x512, m). 없으면 조형 때의 목표")
        p.add_argument("--out", required=True, type=Path)
    pa = sub.choices["auto"]
    pa.add_argument("--thr-mm", type=float, default=2.0, help="이 이상 차이 나는 곳만 영역 후보")
    pa.add_argument("--sigma-mm", type=float, default=2.0, help="차이 지도를 부드럽게 하는 정도")
    pa.add_argument("--min-area-mm2", type=float, default=100.0)
    pa.add_argument("--max-area-tools", type=float, default=4.0, help="영역 하나의 최대 면적(도구 원판 면적의 배수). 이보다 큰 덩어리는 쪼갠다. 0 이면 제한 없음")
    pa.add_argument("--detrend-mm", type=float, default=60.0,
                    help="'현재-목표'의 이 규모(mm)보다 큰 완만한 차이(판 전체 높이 차이 등)는 오차로 보지 않는다. 0 이면 목표를 절대 높이로 따른다")
    pa.add_argument("--max-regions", type=int, default=6)
    pa.add_argument("--border-px", type=int, default=30)
    pa.add_argument("--cap-mm", type=float, default=8.0, help="|차이|가 이보다 큰 곳은 국소 편집으로 만들 수 없어 영역에서 뺀다")
    pa.add_argument("--within", type=Path, help="이 마스크(.npy/.png) 안에서만 영역을 찾는다(관심 영역)")
    pa.add_argument("--within-roi", action="store_true", help="이력에 저장된 ROI(roi_mask_512.npy) 안에서만 찾는다")
    pa.add_argument("--no-reach-limit", action="store_true",
                    help="기본은 도구가 닿을 수 있는 곳(행동 범위 상자 + 도구 반경)에서만 영역을 찾는다. 이 옵션이면 판 전체에서 찾는다")
    pm = sub.choices["masks"]
    pm.add_argument("--mask", action="append", required=True, help="이름=경로.npy[:편집종류[:delta_mm]]")
    pm.add_argument("--delta-mm", type=float, default=3.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    hist, h_cur, target, g = common(a)
    if a.cmd == "auto":
        tool_area = np.pi * (g["tool_r"] * 1000.0) ** 2
        within = None
        if a.within is not None:
            within = np.load(a.within) if a.within.suffix == ".npy" else np.asarray(Image.open(a.within).convert("L")) > 127
        elif a.within_roi:
            roi = Path(a.history_dir) / "roi_mask_512.npy"
            if not roi.exists():
                raise SystemExit(f"이력에 ROI 가 없습니다: {roi}")
            within = np.load(roi)
        if not a.no_reach_limit:
            rmask = cc.reach_mask(g)
            if within is None:
                within = rmask
            else:
                within = (np.asarray(within) > 0) & rmask
            print(f"도구가 닿는 곳(행동 범위 + 도구 반경): 판의 {100 * rmask.mean():.0f}% 안에서만 영역을 찾는다 (--no-reach-limit 로 해제)")
        rs = auto_regions(h_cur, target, g, thr_mm=a.thr_mm, sigma_mm=a.sigma_mm, min_area_mm2=a.min_area_mm2, max_regions=a.max_regions,
                          border_px=a.border_px, max_area_mm2=(a.max_area_tools * tool_area if a.max_area_tools > 0 else None),
                          detrend_mm=a.detrend_mm, cap_mm=a.cap_mm, within=within)
    else:
        specs = []
        for s in a.mask:
            name, rest = s.split("=", 1)
            parts = rest.split(":")
            spec = dict(name=name, mask=np.load(parts[0]) > 0, edit_type=parts[1] if len(parts) > 1 else "match_target")
            if len(parts) > 2:
                spec["delta_mm"] = float(parts[2])
            specs.append(spec)
        rs = masks_regions(h_cur, specs, g, target=target, delta_mm=a.delta_mm)
        rmask = cc.reach_mask(g)
        for n in rs.names:
            f = cc.reach_fraction(rs.masks[n], g, rmask=rmask)
            if f < 0.5:
                print(f"[주의] 영역 '{n}' 은(는) 도구가 닿는 비율이 {100 * f:.0f}% 입니다. 행동 범위 밖이라 cf_compensate 는 --min-reach 기본값(0.5)에서 이 영역을 뺍니다.")
    if not rs.names:
        raise SystemExit("영역이 없습니다. 목표와 현재가 이미 비슷하거나 --thr-mm/--min-area-mm2 가 너무 큽니다.")
    rs.save(a.out)
    preview(a.out / "regions_preview.png", rs, h_cur, target, f"{rs.meta.get('source')} regions | history {Path(a.history_dir).name} | state after action {cc.n_actions(hist)}")
    print(f"기준 상태: 행동 {cc.n_actions(hist)}개 직후 | 영역 {len(rs.names)}개:")
    for n in rs.names:
        print(f"  {n:<8} {rs.labels[n]}  [{int(rs.masks[n].sum())}px]")
    print(f"저장: {a.out} (regions_preview.png 로 확인)")


if __name__ == "__main__":
    main()
