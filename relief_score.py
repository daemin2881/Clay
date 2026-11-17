"""부조(목표의 볼록한 부분)와 그 근처만 재는 점수. 배경은 점수에 넣지 않는다. 대상(A, 하트, ...)과 시점에 의존하지 않는다.

초점(focus)은 목표 높이맵 하나에서 정해진다 (cf_common.relief_focus):
  부조 = 목표에서 판 높이(최빈값)보다 thr 이상 높은 덩어리(판 테두리에 닿은 것 제외), 초점 = 부조 + 도구 반경 x near_tools 이내.
점수 (모두 초점 안에서만, 현재 점토의 국소 바닥에 목표를 맞춘 뒤 비교 = 판 전체가 가라앉아도 부조 모양만 본다):
  형태 shape / 기울기 grad / 테두리 edge   입자 질감을 뺀(흐림 blur-tools x 도구 반경) 눈에 보이는 모양 오차
  MAE(초점) / MAE(부조)                     원본 높이 오차 (mm), 참고용
  L3D / Lviz                                프로젝트 지표를 초점 안에서만 평균 (Lviz 는 입자 질감 포함 원본 픽셀)

사용:
  python relief_score.py --history-dir H [--ts 1,20,80,190]                    이력의 여러 시점 점수 + 시간에 따른 추세(스피어만)
  python relief_score.py --history-dir H --stage before=a.npy --stage after=b.npy   높이맵(m, 512x512) 파일 비교
  --target-npy 새목표.npy  --near-tools 1.5  --level none (절대 높이 그대로)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import cf_common as cc


def score_all(maps, tgt, tool_r_px, near_tools, blur_tools, level, thr_mm=None, base=None):
    fp = cc.relief_focus(tgt, tool_r_px, near_tools=near_tools, thr_mm=thr_mm, base=base)
    sigma = cc.band_sigma(tool_r_px, blur_tools)
    pre = cc.focus_pre(fp, tgt, sigma)
    rows = {}
    for k, h in maps.items():
        s = cc.focus_scores(h, pre, tool_r_px, sigma, level)
        s["lviz_focus"], s["lviz_all"] = cc.focus_lviz(h, pre, fp)
        rows[k] = s
    return fp, pre, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--ts", default="", help="점수를 낼 이력 시점들(쉼표). 기본: 1과 10개 간격과 마지막")
    ap.add_argument("--stage", action="append", default=[], help="이름=높이맵.npy (m). 주면 이력 시점 대신 이것들을 비교")
    ap.add_argument("--target-npy", type=Path)
    ap.add_argument("--near-tools", type=float, default=1.5, help="부조에서 도구 반경의 이 배수 이내를 '그 근처'로 본다")
    ap.add_argument("--thr-mm", type=float, default=None, help="부조로 보는 최소 높이(mm). 기본: max(1mm, 부조 높이의 15%)")
    ap.add_argument("--base-mm", type=float, default=None, help="목표의 판 높이(mm). 기본: 이력 manifest 의 목표 변환에서 읽고, 없으면 히스토그램 최빈값")
    ap.add_argument("--blur-tools", type=float, default=0.4)
    ap.add_argument("--level", choices=["local", "none"], default="local")
    ap.add_argument("--out", type=Path, default=None, help="초점 마스크 그림과 table.json 을 저장할 폴더")
    a = ap.parse_args()
    root = a.history_dir.resolve()
    g = cc.geometry(root)
    tgt = cc.load_target(root, a.target_npy).astype(np.float64)
    if a.stage:
        maps = {}
        for s in a.stage:
            n, p = s.split("=", 1)
            maps[n] = np.load(p).astype(np.float64)
        order = list(maps)
        timeline = False
    else:
        n = cc.n_actions(root)
        ts = [int(x) for x in a.ts.split(",")] if a.ts else sorted({1, *range(10, n + 1, max(n // 10, 1)), n})
        maps = {t: np.load(root / f"action_{t:06d}" / "transition.npz")["after_m"].astype(np.float64) for t in ts}
        order = ts
        timeline = True
    base = a.base_mm * 1e-3 if a.base_mm is not None else cc.plate_level(root)
    fp, pre, rows = score_all(maps, tgt, g["tool_r_px"], a.near_tools, a.blur_tools, a.level, a.thr_mm, base)
    px_mm = g["px_mm"]
    print(f"초점: 판 높이 {fp['base'] * 1000:.1f}mm | 모양 {fp['relief'].sum()}px ({fp['relief'].sum() * px_mm ** 2:.0f}mm^2, 판과 최대 {fp['peak_mm']:.1f}mm 차이, 모양 기준 +-{fp['thr'] * 1000:.1f}mm, 볼록 {100 * fp['raised_frac']:.0f}%/오목 {100 * (1 - fp['raised_frac']):.0f}%) | "
          f"초점(부조+근처 {fp['near_px']}px) {fp['focus'].sum()}px = 맵의 {100 * fp['focus'].mean():.0f}% | 배경은 점수에서 제외")
    print(f"점수 기준: 흐림 {cc.band_sigma(g['tool_r_px'], a.blur_tools):.1f}px, 목표 높이는 {'현재 국소 바닥에 맞춤' if a.level == 'local' else '절대 높이 그대로'}\n")
    hdr = f"{'':>10}{'형태':>7}{'기울기':>8}{'테두리':>8}{'MAE초점':>9}{'MAE부조':>9}{'바닥차':>8}{'L3D초점':>10}{'Lviz초점':>10}{'Lviz전체':>10}"
    print(hdr)
    for k in order:
        r = rows[k]
        print(f"{str(k)[:10]:>10}{r['shape']:>7.2f}{r['grad']:>8.2f}{r['edge']:>8.2f}{r['mae']:>9.2f}{r['mae_relief']:>9.2f}{r['level_mm']:>+8.2f}"
              f"{r['l3d'] * 1e6:>10.2f}{r['lviz_focus']:>10.4f}{r['lviz_all']:>10.4f}")
    print("(형태·MAE·바닥차 mm, 기울기·테두리 mm/도구반경, L3D 단위 mm^2)")
    if timeline and len(order) > 3:
        print("\n시간에 따른 추세 (스피어만, 음수일수록 조형이 진행될수록 점수가 낮아짐 = 눈의 판단과 같은 방향)")
        for key, lab in (("shape", "형태"), ("grad", "기울기"), ("edge", "테두리"), ("mae", "MAE(초점)"), ("mae_relief", "MAE(부조)"), ("lviz_focus", "Lviz(초점)"), ("lviz_all", "Lviz(전체맵)")):
            v = [rows[t][key] for t in order]
            best = order[int(np.argmin(v))]
            print(f"  {lab:<12} 스피어만 {stats.spearmanr(order, v)[0]:+.2f} | 처음 {v[0]:.3f} -> 마지막 {v[-1]:.3f} | 최저 {min(v):.3f} (t={best})")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.json").write_text(json.dumps({str(k): v for k, v in rows.items()}, ensure_ascii=False, indent=1))
        k0 = order[-1]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
        axs[0].imshow((tgt - fp["base"]) * 1000, cmap="turbo", vmin=-3, vmax=10)
        axs[0].set_title("target above plate level (mm)")
        for ax in axs[:2]:
            ax.contour(fp["relief"], levels=[0.5], colors="w", linewidths=1.0)
            ax.contour(fp["focus"], levels=[0.5], colors="k", linewidths=1.2, linestyles="--")
        h = maps[k0]
        axs[1].imshow((h - np.median(h)) * 1000, cmap="turbo", vmin=-6, vmax=10)
        axs[1].set_title(f"state {k0}: relief (white) + nearby (dashed) = scored area")
        d = np.where(fp["focus"], (h - (tgt + cc.local_level(h, pre))) * 1000, np.nan)
        im = axs[2].imshow(d, cmap="RdBu_r", vmin=-6, vmax=6)
        axs[2].set_title("state - target inside focus only (mm)\n(local level matched)")
        fig.colorbar(im, ax=axs[2], fraction=0.046)
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(a.out / "focus.png", dpi=80, bbox_inches="tight")
        print("저장:", a.out / "focus.png", a.out / "table.json")


if __name__ == "__main__":
    main()
