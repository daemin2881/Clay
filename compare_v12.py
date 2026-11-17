"""v12 최종 결과(요청 항목별 가지)와 보상 없이 진행한 원래 이력을 같은 총 행동 수에서 비교한다.
출력(results/case_A/v12_compare/):
  01_height_compare.png        높이맵(강한 색, 같은 색 범위) : 목표 / 원래 80, 86 / v12 홈 / v12 왼쪽 다리 / v10 최종(85)
  02_diff_maps.png             차이 맵(빨강-파랑, 색 대비 큼): v12 - 원래@86, v12 - 시작(85), 목표 - 각 상태
  03_normal_rgb_maps.png       표면 법선 RGB 맵(요철이 색으로 크게 보임)
  04_particle_maps.png         입자(particle) 맵: 위에서 본 입자 높이 색 + 홈 단면(x-z)
  05_graphs.png                총 행동 수-점수 선 그래프, 86번째 영역별 막대, 영역 평균 높이 변화, '3 mm 더 깊게' 목표까지 남은 차
  summary.txt                  행동 수와 수치 요약
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource
from scipy import ndimage

import cf_common as cc
import cf_compensate as cfc
from region_set import RegionSet, border_median

CASE = Path("results/A_history200_20261114_060837")
RES = Path("results/case_A")
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RES / "v12_compare")
    ap.add_argument("--total", type=int, default=86, help="v12 가지의 총 행동 수 (80 + v8 3 + v10 2 + v12 1)")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    root = CASE / "history"
    g = cc.geometry(root)
    R = g["tool_r_px"]
    S = lambda k: np.load(root / f"action_{k:06d}" / "transition.npz")["after_m"].astype(np.float64)
    TOT = a.total
    h80, hO = S(80), S(TOT)
    start85 = np.load(RES / "cf_edit_A80_v10_cont/cf/states/iter_001.npy").astype(np.float64)
    hG = np.load(RES / "cf_edit_A80_v12_grooves/cf/final_m.npy").astype(np.float64)
    hL = np.load(RES / "cf_edit_A80_v12_left_leg/cf/final_m.npy").astype(np.float64)
    tgt = cc.load_target(root).astype(np.float64)
    rsT = RegionSet.load(RES / "cf_edit_A80_v4_mdl/prompt", ref_default=h80)
    rsD = RegionSet.load(RES / "regions_deepen3", ref_default=start85)
    names = rsT.names
    union = rsT.union()
    fp = cc.relief_focus(tgt, R, base=cc.plate_level(root))
    ys, xs = np.where(fp["focus"] | union)
    r0, r1 = max(ys.min() - 50, 0), min(ys.max() + 50, 512)
    c0, c1 = max(xs.min() - 50, 0), min(xs.max() + 50, 512)
    sl = (slice(r0, r1), slice(c0, c1))
    states = {f"original {TOT} actions (no compensation)": hO, f"v12 grooves branch ({TOT} actions)": hG, f"v12 left-leg branch ({TOT} actions)": hL}

    def contour(ax):
        for n in names:
            ax.contour(rsT.masks[n][sl], levels=[0.5], colors="k", linewidths=0.8, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])

    kw = dict(cmap="turbo", vmin=-10, vmax=14)
    # ---- 01 높이맵
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), facecolor=SURF)
    items = [("Target A (requested regions dashed)", tgt), ("original, 80 actions", h80), (f"original, {TOT} actions\n(no compensation)", hO),
             (f"v12 grooves branch, {TOT} actions\n(+{TOT - 80} from 80)", hG), (f"v12 left-leg branch, {TOT} actions\n(+{TOT - 80} from 80)", hL), ("v10 final (85 actions)\nstart of v12", start85)]
    for ax, (t, h) in zip(axs.ravel(), items):
        im = ax.imshow(((h - border_median(h)) * 1000)[sl], **kw)
        ax.set_title(t, fontsize=10); contour(ax)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01, label="height above plate (mm)")
    fig.suptitle(f"Height maps: v12 branches vs original history at the same total action count ({TOT})", fontsize=12)
    fig.savefig(out / "01_height_compare.png", dpi=105, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---- 02 차이 맵
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), facecolor=SURF)
    dk = dict(cmap="RdBu_r", vmin=-6, vmax=6)
    diffs = [(f"v12 grooves - original@{TOT}", hG - hO), (f"v12 left leg - original@{TOT}", hL - hO), ("v12 grooves - its start (85)\n= the single push", hG - start85),
             (f"original@{TOT} - target", hO - (tgt - border_median(tgt) + border_median(hO))), (f"v12 grooves - target", hG - (tgt - border_median(tgt) + border_median(hG))),
             (f"v12 left leg - target", hL - (tgt - border_median(tgt) + border_median(hL)))]
    for ax, (t, d) in zip(axs.ravel(), diffs):
        im = ax.imshow((d * 1000)[sl], **dk)
        ax.set_title(t, fontsize=10); contour(ax)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01, label="mm (red = higher / too high, blue = lower / too low)")
    fig.suptitle("Difference maps (strong diverging colors)", fontsize=12)
    fig.savefig(out / "02_diff_maps.png", dpi=105, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---- 03 법선 RGB
    def normal_rgb(h, exag=10.0):
        hs = ndimage.gaussian_filter(h, 1.0)
        dzdy, dzdx = np.gradient(hs, g["px_mm"] / 1000.0)
        n = np.stack([-exag * dzdx, -exag * dzdy, np.ones_like(hs)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return np.clip(n * 0.5 + 0.5, 0, 1)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), facecolor=SURF)
    for ax, (t, h) in zip(axs, [("target", tgt), (f"original@{TOT}", hO), ("v12 grooves", hG), ("v12 left leg", hL)]):
        ax.imshow(normal_rgb(h)[sl]); ax.set_title(t, fontsize=10); contour(ax)
    fig.suptitle("Surface-normal RGB maps (R = x slope, G = y slope, B = flat): bumps and pits show as color", fontsize=11)
    fig.savefig(out / "03_normal_rgb_maps.png", dpi=105, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---- 04 입자 맵 (표면 입자만: 높이맵 표면에서 6.5 mm 이내(입자 중심은 표면보다 약 2.3~3 mm 아래))
    def particles(path):
        z = torch.load(path, map_location="cpu", weights_only=False)
        return z["physics"][3]["tensors"]["_pos"][0].numpy().astype(np.float64)
    hmaps = {f"original@{TOT}": hO, "v12 grooves": hG, "v12 left leg": hL}
    pp = {f"original@{TOT}": particles(root / f"action_{TOT:06d}" / "post_state.pt"), "v12 grooves": particles(RES / "cf_edit_A80_v12_grooves/cf/states/iter_001.pt"),
          "v12 left leg": particles(RES / "cf_edit_A80_v12_left_leg/cf/states/iter_001.pt")}
    wx = lambda col: g["x0"] + (col + 0.5) / g["px_per_m_x"]
    wy = lambda row: g["y0"] + (row + 0.5) / g["px_per_m_y"]
    xlo, xhi, ylo, yhi = wx(c0), wx(c1), wy(r0), wy(r1)

    def surface(P, h):
        col = np.clip(np.round((P[:, 0] - g["x0"]) * g["px_per_m_x"] - 0.5).astype(int), 0, 511)
        row = np.clip(np.round((P[:, 1] - g["y0"]) * g["px_per_m_y"] - 0.5).astype(int), 0, 511)
        return P[:, 2] >= h[row, col] - 0.0065
    fig = plt.figure(figsize=(16, 9.8), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    for j, (t, P) in enumerate(pp.items()):
        ax = fig.add_subplot(gs[0, j])
        bg = border_median(hmaps[t])
        sel = (P[:, 0] >= xlo) & (P[:, 0] <= xhi) & (P[:, 1] >= ylo) & (P[:, 1] <= yhi) & surface(P, hmaps[t])
        Q = P[sel]; o = np.argsort(Q[:, 2])
        sc = ax.scatter(Q[o, 0], Q[o, 1], c=(Q[o, 2] - bg) * 1000, s=9, cmap="turbo", vmin=-10, vmax=14, linewidths=0)
        ax.set_xlim(xlo, xhi); ax.set_ylim(yhi, ylo); ax.set_aspect("equal")
        ax.set_title(f"{t}: surface particles seen from above ({int(sel.sum())})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=fig.axes[:3], fraction=0.02, pad=0.01, label="particle height above plate (mm)")
    cols = {f"original@{TOT}": GRAY, "v12 grooves": ORANGE, "v12 left leg": GREEN}
    for j, n in enumerate(("groove_bottom", "groove_top", "left_leg")):
        ax = fig.add_subplot(gs[1, j])
        yy, xx = np.where(rsT.masks[n]); yc = wy(yy.mean()); xl, xr = wx(xx.min() - 25), wx(xx.max() + 25)
        zs = []
        for t, P in pp.items():
            bg = border_median(hmaps[t])
            sel = (np.abs(P[:, 1] - yc) < 0.006) & (P[:, 0] >= xl) & (P[:, 0] <= xr) & surface(P, hmaps[t])
            ax.scatter(P[sel, 0] * 1000, (P[sel, 2] - bg) * 1000, s=16, color=cols[t], alpha=0.6, linewidths=0, label=t)
        ax.axvspan(wx(xx.min()) * 1000, wx(xx.max()) * 1000, color="#999", alpha=0.12)
        ax.set_title(f"surface cross-section (x-z) through {n} (|y - y_c| < 6 mm)", fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("surface particle height above plate (mm)"); ax.grid(True, color=GRID, lw=0.6)
        if j == 0: ax.legend(fontsize=8, markerscale=1.6)
    fig.suptitle("Particle maps (Genesis MPM clay surface particles): original history vs v12 branches", fontsize=12)
    fig.savefig(out / "04_particle_maps.png", dpi=100, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---- 점수 계산
    sigma = cc.band_sigma(R, 0.4)
    cfg0 = cfc.build_cfg(rsT, h80, 1.0, int(round(0.94 * R)), int(round(1.17 * R)), R)
    norm, bm80 = cfg0["norm"], border_median(h80)

    def score(h):
        sh = border_median(h) - bm80
        pre = cc.appearance_pre(rsT.masks, {n: rsT.targets[n] + sh for n in names}, sigma)
        v = cc.appearance(h, pre, R, sigma)
        per = {n: v[n][2] / norm[n][0] + v[n][0] / norm[n][1] + v[n][1] / norm[n][2] for n in names}
        return float(np.mean(list(per.values()))), per, v
    ks = list(range(80, 106))
    e_orig = [score(S(k))[0] for k in ks]
    e_pts = {"orig": score(hO), "grooves": score(hG), "leg": score(hL), "start85": score(start85), "o80": score(h80)}

    # ---- 05 그래프
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURF)
    ax = axs[0, 0]
    ax.plot(ks, e_orig, color=BLUE, lw=1.4, label="original history (no compensation)")
    for arm, col in (("A1", BLUE), ("A2", BLUE)):
        pass
    ax.plot([80, 81, 82, 83, 85], [3.0, 2.957, 2.905, 2.868, 2.765], color=ORANGE, lw=1.6, marker="o", ms=4.5, label="compensation (v8 -> v10)")
    ax.scatter([TOT], [e_pts["grooves"][0]], color=ORANGE, marker="D", s=55, zorder=5, label=f"v12 grooves branch @{TOT}")
    ax.scatter([TOT], [e_pts["leg"][0]], color=GREEN, marker="s", s=55, zorder=5, label=f"v12 left-leg branch @{TOT}")
    ax.axhline(3.0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.set_title("Region score vs total actions (A-target metric, 80 = 3.00, lower is better)", loc="left"); ax.set_xlabel("total number of actions"); ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=8)
    ax = axs[0, 1]
    w = 0.26; x = np.arange(len(names))
    for i, (lab, key, col) in enumerate((("original", "orig", GRAY), ("v12 grooves", "grooves", ORANGE), ("v12 left leg", "leg", GREEN))):
        ax.bar(x + (i - 1) * w, [e_pts[key][1][n] for n in names], w, color=col, label=lab)
    ax.axhline(3.0, color=MUTED, lw=0.7, ls=(0, (4, 3))); ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_title(f"Per-region score at {TOT} actions (A-target metric; 3.00 = state 80)", loc="left"); ax.legend(fontsize=8); ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax = axs[1, 0]
    dh = lambda h: [(h - h80)[rsT.masks[n]].mean() * 1000 for n in names]
    for i, (lab, h, col) in enumerate((("original", hO, GRAY), ("v12 grooves", hG, ORANGE), ("v12 left leg", hL, GREEN))):
        ax.bar(x + (i - 1) * w, dh(h), w, color=col, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_title(f"Mean height change vs state 80 at {TOT} actions (mm, negative = dug)", loc="left"); ax.legend(fontsize=8); ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax = axs[1, 1]
    gl = ["groove_top", "groove_bottom"]
    rem = lambda h: [(h - rsD.targets[n])[rsD.masks[n]].mean() * 1000 for n in gl]
    xx2 = np.arange(2)
    for i, (lab, h, col) in enumerate((("start (85)", start85, MUTED), ("original", hO, GRAY), ("v12 grooves", hG, ORANGE), ("v12 left leg", hL, GREEN))):
        ax.bar(xx2 + (i - 1.5) * 0.2, rem(h), 0.2, color=col, label=lab)
    ax.set_xticks(xx2); ax.set_xticklabels(["groove_top", "groove_bottom"]); ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("Height above the '3 mm deeper' target (mm; 0 = goal reached, smaller = deeper)", loc="left"); ax.legend(fontsize=8); ax.grid(True, axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"v12 vs original history at {TOT} total actions (80 + {TOT - 80})", fontsize=12)
    fig.savefig(out / "05_graphs.png", dpi=105, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---- 요약
    lines = [f"총 행동 수: 원래 이력 {TOT} (보상 없음) | v12 홈 가지 {TOT} = 80 + 보상 5(v8 3 + v10 2) + v12 1 | v12 왼쪽 다리 가지 {TOT} = 80 + 5 + 1", "",
             f"{'영역':<14}{'원래@80':>9}{'원래@' + str(TOT):>10}{'v12 홈':>9}{'v12 다리':>9}   (A 목표 기준 영역 항 합, 80번째 = 3.00)"]
    for n in names:
        lines.append(f"{n:<14}{e_pts['o80'][1][n]:>9.2f}{e_pts['orig'][1][n]:>10.2f}{e_pts['grooves'][1][n]:>9.2f}{e_pts['leg'][1][n]:>9.2f}")
    lines.append(f"{'영역 평균 e':<14}{e_pts['o80'][0]:>9.3f}{e_pts['orig'][0]:>10.3f}{e_pts['grooves'][0]:>9.3f}{e_pts['leg'][0]:>9.3f}")
    lines.append("")
    lines.append("영역 평균 높이 변화(80번째 대비, mm): " + " | ".join(f"{n}: 원래 {dh(hO)[i]:+.2f} / v12홈 {dh(hG)[i]:+.2f} / v12다리 {dh(hL)[i]:+.2f}" for i, n in enumerate(names)))
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print("저장:", out)


if __name__ == "__main__":
    main()
