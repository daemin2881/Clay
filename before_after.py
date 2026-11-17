"""국소 보상 전/후를 한 장으로 비교한다 (높이맵 + 음영 + 차이 + 영역별 오차 표). 대상·시점·영역 구성에 의존하지 않는다.

단계(stage)를 여러 개 줄 수 있다. 첫 번째가 '전', 마지막이 '후'다. 모두 같은 색 범위, 같은 광원, 같은 확대 영역.
  행 1: 높이맵(배경 위 mm, turbo)  [목표 | 단계들...]
  행 2: 음영(hillshade)            [목표 | 단계들...]
  행 3: 단계 - 목표 (mm)           [        단계들...]   빨강 = 너무 높음, 파랑 = 너무 낮음
점선 = 영역 집합의 영역들. 목표는 영역 안에서만 영역의 목표이고 밖은 기준 상태 그대로다.

사용:
  python before_after.py --regions DIR [--ref-h H.npy] --stage "before=A.npy" --stage "after=B.npy" --out OUT [--title ...]
    DIR: 영역 집합 폴더(build_regions.py / live_edit.py 출력 / 예전 형식).  *.npy 는 512x512 높이맵(m)
저장: OUT/before_after.png, OUT/table.json (영역별 MAE/RMSE, 영역 밖 변화)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from scipy import ndimage

import cf_common as cc
from region_set import RegionSet, border_median

PIX_MM = 360.0 / 511


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--ref-h", type=Path, default=None, help="영역 폴더에 기준 높이맵(h_ref.npy)이 없을 때. 보통 '전' 단계 높이맵")
    ap.add_argument("--names", default="", help="영역 일부만 쓸 때 쉼표로 구분한 이름")
    ap.add_argument("--stage", action="append", required=True, help="이름=높이맵.npy (m 단위, 512x512). 첫 번째가 전, 마지막이 후. 이름에 \\n 가능")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="Local compensation: before vs after (judge by eye: does it look like the target?)")
    ap.add_argument("--history-dir", type=Path, default=None, help="주면 '목표의 모양 + 근처'(배경 제외)만 잰 점수도 표와 그림에 넣는다(relief_score.py 와 같은 정의)")
    ap.add_argument("--target-npy", type=Path, default=None)
    ap.add_argument("--near-tools", type=float, default=1.5)
    ap.add_argument("--pad", type=int, default=40)
    ap.add_argument("--exag", type=float, default=2.0)
    ap.add_argument("--stage-minus-target", action="store_true", help="3행을 예전처럼 (단계 - 목표)로 그린다(빨강 = 너무 높음). 기본은 '해야 할 일' = 목표 - 단계: 빨강 = 채워야 함, 파랑 = 깎아야 함")
    ap.add_argument("--far-px", type=int, default=25)
    ap.add_argument("--tool-r-px", type=float, default=21.3, help="도구 반경(화소). 외형 오차를 'mm / 도구 반경'으로 환산하고 흐림 크기를 정할 때 쓴다")
    ap.add_argument("--blur-tools", type=float, default=0.4, help="형태·외형 오차를 재는 흐림 크기(도구 반경의 배수). cf_compensate 와 같아야 한다")
    a = ap.parse_args()
    global PIX_MM
    if a.history_dir is not None:
        PIX_MM = cc.geometry(a.history_dir.resolve())["px_mm"]      # 음영의 화소 크기를 이력의 좌표 변환에서 읽는다
    a.out.mkdir(parents=True, exist_ok=True)

    stages = []
    for s in a.stage:
        name, path = s.split("=", 1)
        stages.append((name.replace("\\n", "\n"), np.load(path).astype(np.float64)))
    before = stages[0][1]
    rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_h=a.ref_h, ref_default=before)
    names, masks, tgts = rs.names, rs.masks, rs.targets
    hstar = rs.h_star_all(before)
    bg = border_median(before)
    union = rs.union()
    ys, xs = np.where(union)
    r0, r1 = max(ys.min() - a.pad, 0), min(ys.max() + a.pad, 512)
    c0, c1 = max(xs.min() - a.pad, 0), min(xs.max() + a.pad, 512)
    far = ~ndimage.binary_dilation(union, iterations=a.far_px)
    ring = ndimage.binary_dilation(union, iterations=20) & ~union

    # ---- 표: 영역별 MAE/RMSE (영역 목표 대비), 영역 밖 변화
    table = []
    sigma = cc.band_sigma(a.tool_r_px, a.blur_tools)
    fp = fpre = None
    if a.history_dir is not None:
        try:
            tgt_all = cc.load_target(a.history_dir.resolve(), a.target_npy).astype(np.float64)
            fp = cc.relief_focus(tgt_all, a.tool_r_px, near_tools=a.near_tools, base=cc.plate_level(a.history_dir.resolve()))
            fpre = cc.focus_pre(fp, tgt_all, sigma)
        except ValueError as e:
            print("[초점 없음]", e)
    pre = cc.appearance_pre(masks, tgts, sigma)
    for name, h in stages:
        row = dict(stage=name.replace("\n", " "), regions={})
        vis = cc.appearance(h, pre, a.tool_r_px, sigma)
        for n in names:
            d = (h - tgts[n])[masks[n]] * 1000
            row["regions"][n] = dict(mae_mm=float(np.abs(d).mean()), rmse_mm=float(np.sqrt((d ** 2).mean())), grad_err=vis[n][0], edge_err=vis[n][1], shape_err=vis[n][2])
        if fpre is not None:
            row["focus"] = cc.focus_scores(h, fpre, a.tool_r_px, sigma, "local")
        row["mean_region_mae_mm"] = float(np.mean([v["mae_mm"] for v in row["regions"].values()]))
        row["far_change_max_mm"] = float(np.abs(h - before)[far].max() * 1000)
        row["near_ring_change_mean_mm"] = float(np.abs(h - before)[ring].mean() * 1000)
        table.append(row)
    (a.out / "table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
    print("[원본 높이] 영역 MAE/RMSE (mm) — 입자 질감 포함이라 참고용")
    print(f"{'단계':<28}" + "".join(f"{n[:9]:>13}" for n in names) + f"{'평균':>8}{'밖최대':>8}")
    for r in table:
        print(f"{r['stage'][:27]:<28}" + "".join(f"{r['regions'][n]['mae_mm']:>7.2f}/{r['regions'][n]['rmse_mm']:<5.2f}" for n in names)
              + f"{r['mean_region_mae_mm']:>8.2f}{r['far_change_max_mm']:>8.1f}")
    print(f"[눈에 보이는 형태] 입자 질감을 뺀(흐림 {sigma:.1f}px) 영역별 형태 / 기울기 / 테두리 오차, 낮을수록 목표 모양에 가까움")
    for r in table:
        print(f"{r['stage'][:27]:<28}" + "".join(f"{r['regions'][n]['shape_err']:>5.2f}/{r['regions'][n]['grad_err']:.2f}/{r['regions'][n]['edge_err']:<5.2f}" for n in names))

    if fpre is not None:
        print(f"[글자+근처만] 목표의 모양 {int(fp['relief'].sum())}px + 근처 = 맵의 {100 * fp['focus'].mean():.0f}% 에서만 재고 배경은 뺌 (현재 국소 바닥에 목표를 맞춤): 형태 / 기울기 / 테두리 / MAE(모양 안)")
        for r in table:
            f = r["focus"]
            print(f"{r['stage'][:27]:<28}{f['shape']:>6.2f}/{f['grad']:.2f}/{f['edge']:.2f}/{f['mae_relief']:.2f}mm  (바닥차 {f['level_mm']:+.2f}mm)")

    # ---- 그림
    ncol = 1 + len(stages)
    fig, axs = plt.subplots(3, ncol, figsize=(3.3 * ncol, 2.5 * 3 + 0.8), facecolor="#fcfcfb")
    ls = LightSource(azdeg=315, altdeg=40)
    kw = dict(cmap="turbo", vmin=-10, vmax=14)

    def outline(ax):
        if fp is not None:
            ax.contour(fp["focus"][r0:r1, c0:c1], levels=[0.5], colors="w", linewidths=0.8, linestyles=":")
        for n in names:
            ax.contour(masks[n][r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])

    panels = [("Target (requested regions)", hstar)] + stages
    im = None
    for j, (title, h) in enumerate(panels):
        axs[0, j].imshow(((h - bg) * 1000)[r0:r1, c0:c1], **kw)
        axs[0, j].set_title(title, fontsize=9)
        outline(axs[0, j])
        shade = ls.hillshade(h * 1000.0, vert_exag=a.exag, dx=PIX_MM, dy=PIX_MM)
        axs[1, j].imshow(shade[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1)
        outline(axs[1, j])
        if j == 0:
            axs[2, 0].axis("off")
            axs[2, 0].text(0.02, 0.98, "Row 1: height above background (mm)\nRow 2: shaded relief (same light)\nRow 3: " + ("stage - target (mm)\n   red = too high (cut), blue = too low (fill)" if a.stage_minus_target else "TO-DO = target - stage (mm)\n   RED = needs FILL (raise), BLUE = needs CUT (lower)") + "\nDashed = regions:\n" +
                           "\n".join(f"  {n}" for n in names), va="top", fontsize=8.5, transform=axs[2, 0].transAxes)
        else:
            im = axs[2, j].imshow((((h - hstar) if a.stage_minus_target else (hstar - h)) * 1000)[r0:r1, c0:c1], cmap="RdBu_r", vmin=-6, vmax=6)
            outline(axs[2, j])
    if im is not None:
        fig.colorbar(im, ax=axs[2, 1:], fraction=0.02, pad=0.01).set_label("stage - target (mm)" if a.stage_minus_target else "to-do (mm): + fill / - cut")
    fig.suptitle(a.title, fontsize=11)
    fig.savefig(a.out / "before_after.png", dpi=110, facecolor="#fcfcfb", bbox_inches="tight")
    print("저장:", a.out / "before_after.png", a.out / "table.json")


if __name__ == "__main__":
    main()
