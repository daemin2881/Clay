"""회차(총 행동 수)별로 '보상 없음(원래 이력)'과 '국소 보상' 높이맵을 비교하는 PNG와 그래프를 만든다.

논문(Visual Sculpting)의 그림 형식을 따라 쓴다:
  깊이(높이) 맵, 시각 정렬 표현(기울기: dx 초록 / dy 파랑, 그림 5), 음영(ray-traced 대용, 그림 13), 실제 변화 맵(빨강 = 눌림, 파랑 = 밀려 올라옴, 그림 7/13),
  행동 수에 따른 3D 손실(파랑)과 시각 손실(빨강)(그림 11).

색 약속(이 스크립트의 모든 변화/차이 맵): 빨강 = 더 낮음(눌림, 파짐), 파랑 = 더 높음(밀려 올라옴). 차이 = (보상) - (원래).
출력: results/case_A/rounds_compare/
  individual/  회차별 개별 PNG (깊이 맵 원래/보상, 차이 맵)
  S1_depth_by_round.png            회차별 [원래 | 보상 | 차이]
  S2_cumulative_change.png         80번째 대비 누적 변화(원래 | 보상) 회차별
  S3_visual_aligned.png            목표/원래/보상의 기울기(dx 초록, dy 파랑)와 음영
  S4_fig13_comp_actions.png        보상 확정 행동마다 전/후 깊이, 실제 변화, 음영 전/후, 시각 변화(그림 13 형식)
  S5_fig13_orig_actions.png        같은 형식, 원래 이력의 81~86번째 행동
  G1_loss_vs_actions.png           그림 11 형식: 행동 수별 3D 손실(파랑)/시각 손실(빨강), 원래 대 보상
  G2_difference_graphs.png         (보상 - 원래) 차이 그래프: 영역 점수, 영역별 높이, 3D 손실, 시각 손실
  numbers.json / summary_rounds.txt 수치
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy import ndimage

import cf_common as cc
import cf_compensate as cfc
from region_set import RegionSet, border_median

CASE = Path("results/A_history200_20261114_060837")
RES = Path("results/case_A")
BLUE, ORANGE, GREEN, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#d03a3a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ROUNDS = [80, 81, 82, 83, 85, 86]
DLIM = 4.0     # 차이 맵 색 범위 (mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RES / "rounds_compare")
    a = ap.parse_args()
    out = a.out
    (out / "individual").mkdir(parents=True, exist_ok=True)
    root = CASE / "history"
    g = cc.geometry(root)
    R, PX = g["tool_r_px"], g["px_mm"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    T = lambda k: np.load(root / f"action_{k:06d}" / "transition.npz")
    S = lambda k: T(k)["after_m"].astype(np.float64)
    L = lambda p: np.load(RES / p).astype(np.float64)

    h80 = S(80)
    # 보상 경로 (행동 수 -> 높이맵). v8 3개(1행동씩) -> v10 PAIR(2행동) -> v12(1행동, 두 가지)
    C = {80: h80, 81: L("cf_edit_A80_v8/cf/states/iter_001.npy"), 82: L("cf_edit_A80_v8/cf/states/iter_002.npy"), 83: L("cf_edit_A80_v8/cf/states/iter_003.npy"),
         85: L("cf_edit_A80_v10_cont/cf/states/iter_001.npy")}
    CG, CL = dict(C), dict(C)
    CG[86] = L("cf_edit_A80_v12_grooves/cf/final_m.npy")
    CL[86] = L("cf_edit_A80_v12_left_leg/cf/final_m.npy")
    labels = {81: "v8 #1  refine -> right foot", 82: "v8 #2  G1(past #68) -> right foot", 83: "v8 #3  G1(past #68) again -> right foot",
              85: "v10  PAIR (2 pushes) -> left foot", "86G": "v12  G1(past #35) -> groove_bottom (3 mm deeper)", "86L": "v12  refine -> left leg"}
    tgt = cc.load_target(root).astype(np.float64)
    rs = RegionSet.load(RES / "cf_edit_A80_v4_mdl/prompt", ref_default=h80)
    names = rs.names
    fp = cc.relief_focus(tgt, R, base=cc.plate_level(root))
    ys, xs = np.where(fp["focus"] | rs.union())
    r0, r1 = max(ys.min() - 50, 0), min(ys.max() + 50, 512)
    c0, c1 = max(xs.min() - 50, 0), min(xs.max() + 50, 512)
    sl = (slice(r0, r1), slice(c0, c1))

    def contour(ax, sl_=sl, lw=0.8):
        for n in names:
            ax.contour(rs.masks[n][sl_], levels=[0.5], colors="k", linewidths=lw, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])

    mm = lambda h: (h - border_median(h)) * 1000.0
    depth_kw = dict(cmap="turbo", vmin=-10, vmax=14)
    diff_kw = dict(cmap="RdBu", vmin=-DLIM, vmax=DLIM)       # 낮음 = 빨강, 높음 = 파랑
    ls = LightSource(azdeg=315, altdeg=40)

    def shade(h):
        return ls.hillshade(ndimage.gaussian_filter(h * 1000.0, 1.0), vert_exag=3.0, dx=PX, dy=PX)

    def vis_rgb(h, gmax=None):
        """그림 5 형식의 시각 정렬 표현: dx = 초록, dy = 파랑."""
        hs = ndimage.gaussian_filter(h * 1000.0, 1.5)
        gy, gx = np.gradient(hs)
        gm = gmax if gmax is not None else float(np.percentile(np.hypot(gx, gy), 99))
        rgb = np.stack([np.full_like(gx, 0.12), np.clip(0.5 + gx / (2 * gm), 0, 1), np.clip(0.5 + gy / (2 * gm), 0, 1)], -1)
        return rgb, gm

    # ---------- 점수/손실 ----------
    sigma = cc.band_sigma(R, 0.4)
    cfg0 = cfc.build_cfg(rs, h80, 1.0, int(round(0.94 * R)), int(round(1.17 * R)), R)
    norm, bm80 = cfg0["norm"], border_median(h80)
    tm = (tgt - border_median(tgt)) * 1000.0

    def score(h):
        sh = border_median(h) - bm80
        pre = cc.appearance_pre(rs.masks, {n: rs.targets[n] + sh for n in names}, sigma)
        v = cc.appearance(h, pre, R, sigma)
        per = {n: v[n][2] / norm[n][0] + v[n][0] / norm[n][1] + v[n][1] / norm[n][2] for n in names}
        return float(np.mean(list(per.values()))), per

    def grad2(x):
        gy, gx = np.gradient(x)
        return gx, gy

    tb = ndimage.gaussian_filter(tm, sigma)
    tgx_b, tgy_b = grad2(tb)
    tgx_r, tgy_r = grad2(tm)

    def losses(h):
        """3D 손실 = 지도 전체의 높이 차 제곱 평균(mm^2), 시각 손실 = 기울기 차 제곱 평균((mm/화소)^2). 판 높이는 테두리 중앙값으로 맞춘다.
        blur: 도구 반경 0.4배(8.5화소)로 흐린 지도(입자 질감 제거), raw: 흐리지 않은 원본(계획기와 같은 방식)."""
        m = mm(h)
        mb = ndimage.gaussian_filter(m, sigma)
        gxb, gyb = grad2(mb)
        gxr, gyr = grad2(m)
        return {"l3d_blur": float(np.mean((mb - tb) ** 2)), "lviz_blur": float(np.mean((gxb - tgx_b) ** 2 + (gyb - tgy_b) ** 2)),
                "l3d_raw": float(np.mean((m - tm) ** 2)), "lviz_raw": float(np.mean((gxr - tgx_r) ** 2 + (gyr - tgy_r) ** 2))}

    def reg_mean(h):
        return {n: float((h - h80)[rs.masks[n]].mean() * 1000) for n in names}

    # ---------- 개별 PNG ----------
    def save_single(path, img, title, cbar_label, kw, with_contour=True):
        fig, ax = plt.subplots(figsize=(6.2, 5.4), facecolor=SURF)
        im = ax.imshow(img[sl], **kw)
        if with_contour:
            contour(ax)
        ax.set_title(title, fontsize=10.5, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, fontsize=8.5)
        fig.savefig(path, dpi=100, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    variants = []      # (회차, 라벨, 높이맵 보상, 파일 접미사)
    for k in ROUNDS:
        if k == 86:
            variants.append((86, "v12 grooves branch", CG[86], "comp_grooves"))
            variants.append((86, "v12 left-leg branch", CL[86], "comp_leftleg"))
        else:
            variants.append((k, "compensation", C[k], "comp"))
    for k, lab, hc, suf in variants:
        ho = S(k)
        save_single(out / "individual" / f"depth_original_{k:03d}actions.png", mm(ho), f"original (no compensation), {k} actions", "height above plate (mm)", depth_kw)
        save_single(out / "individual" / f"depth_{suf}_{k:03d}actions.png", mm(hc), f"{lab}, {k} actions", "height above plate (mm)", depth_kw)
        save_single(out / "individual" / f"diff_{suf}_minus_original_{k:03d}actions.png", (hc - ho) * 1000, f"{lab} - original at {k} actions",
                    f"mm  (red = compensated is lower, blue = higher; +-{DLIM:.0f} mm)", diff_kw)

    # ---------- S1: 회차별 [원래 | 보상 | 차이] ----------
    rows = [(k, "", C[k]) for k in ROUNDS if k != 86] + [(86, " (v12 grooves)", CG[86]), (86, " (v12 left leg)", CL[86])]
    fig, axs = plt.subplots(len(rows), 3, figsize=(11.2, 3.35 * len(rows)), facecolor=SURF, layout="constrained")
    for i, (k, tag, hc) in enumerate(rows):
        ho = S(k)
        im1 = axs[i, 0].imshow(mm(ho)[sl], **depth_kw)
        axs[i, 1].imshow(mm(hc)[sl], **depth_kw)
        im3 = axs[i, 2].imshow(((hc - ho) * 1000)[sl], **diff_kw)
        axs[i, 0].set_title(f"original, {k} actions", fontsize=9.5, loc="left")
        axs[i, 1].set_title(f"compensated, {k} actions{tag}", fontsize=9.5, loc="left")
        axs[i, 2].set_title(f"difference (compensated - original) @ {k}", fontsize=9.5, loc="left")
        for ax in axs[i]:
            contour(ax)
    fig.colorbar(im1, ax=axs[:, :2], fraction=0.018, pad=0.012, label="height above plate (mm)")
    fig.colorbar(im3, ax=axs[:, 2], fraction=0.04, pad=0.02, label=f"mm (red = lower, blue = higher, +-{DLIM:.0f})")
    fig.suptitle("Height maps at the same total action count: original (no compensation) vs local compensation", fontsize=12)
    fig.savefig(out / "S1_depth_by_round.png", dpi=85, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---------- S2: 80번째 대비 누적 변화 ----------
    fig, axs = plt.subplots(len(rows), 2, figsize=(8.2, 3.35 * len(rows)), facecolor=SURF, layout="constrained")
    for i, (k, tag, hc) in enumerate(rows):
        ho = S(k)
        axs[i, 0].imshow(((ho - h80) * 1000)[sl], **diff_kw)
        im = axs[i, 1].imshow(((hc - h80) * 1000)[sl], **diff_kw)
        axs[i, 0].set_title(f"original: {k} - 80", fontsize=9.5, loc="left")
        axs[i, 1].set_title(f"compensated{tag}: {k} - 80", fontsize=9.5, loc="left")
        for ax in axs[i]:
            contour(ax)
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, label=f"change vs state 80 (mm; red = lowered, blue = raised; +-{DLIM:.0f})")
    fig.suptitle("Cumulative change since state 80", fontsize=12)
    fig.savefig(out / "S2_cumulative_change.png", dpi=85, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---------- S3: 시각 정렬(기울기) + 음영 ----------
    items = [("target A", tgt), ("original, 86 actions", S(86)), ("compensated: v12 grooves, 86", CG[86]), ("compensated: v12 left leg, 86", CL[86])]
    _, gm = vis_rgb(tgt)
    fig, axs = plt.subplots(3, 4, figsize=(16, 12.6), facecolor=SURF, layout="constrained")
    for j, (t, h) in enumerate(items):
        axs[0, j].imshow(mm(h)[sl], **depth_kw); axs[0, j].set_title(f"{t}\ndepth", fontsize=9.5, loc="left")
        axs[1, j].imshow(vis_rgb(h, gm)[0][sl]); axs[1, j].set_title("visually-aligned: spatial gradient (dx green, dy blue)", fontsize=9, loc="left")
        axs[2, j].imshow(shade(h)[sl], cmap="gray", vmin=0, vmax=1); axs[2, j].set_title("shaded render (light from upper left)", fontsize=9, loc="left")
        for r in range(3):
            contour(axs[r, j], lw=0.6)
    fig.suptitle("Paper-style views: depth / visually-aligned gradient (Fig. 5) / shaded render (Fig. 13)", fontsize=12)
    fig.savefig(out / "S3_visual_aligned.png", dpi=80, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---------- S4/S5: 행동 1개의 전/후 (그림 13 형식) ----------
    def action_sheet(entries, title, path):
        fig, axs = plt.subplots(len(entries), 6, figsize=(17, 2.95 * len(entries)), facecolor=SURF, layout="constrained")
        ims = {}
        for i, (lab, hb, ha) in enumerate(entries):
            d = (ha - hb) * 1000.0
            dd = ndimage.gaussian_filter(np.abs(d), 3.0)
            cy, cx = np.unravel_index(np.argmax(dd), dd.shape)
            half = 70
            y0_, x0_ = int(np.clip(cy - half, 0, 512 - 2 * half)), int(np.clip(cx - half, 0, 512 - 2 * half))
            w = (slice(y0_, y0_ + 2 * half), slice(x0_, x0_ + 2 * half))
            sb, sa = shade(hb), shade(ha)
            cols = [(mm(hb)[w], depth_kw, "state before (depth)"), (mm(ha)[w], depth_kw, "state after (depth)"),
                    (d[w], dict(cmap="RdBu", vmin=-3, vmax=3), "real change (red = pressed, blue = raised)"),
                    (sb[w], dict(cmap="gray", vmin=0, vmax=1), "shaded before"), (sa[w], dict(cmap="gray", vmin=0, vmax=1), "shaded after"),
                    ((sa - sb)[w], dict(cmap="RdBu_r", vmin=-0.5, vmax=0.5), "visual change (shade after - before)")]
            for j, (img, kw, t) in enumerate(cols):
                im = axs[i, j].imshow(img, **kw)
                ims[j] = im
                axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                axs[i, j].set_title((lab + "\n" if j == 0 else "") + t, fontsize=8.6, loc="left")
            for n in names:
                axs[i, 2].contour(rs.masks[n][w], levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
            axs[i, 2].text(0.02, 0.03, f"min {d[w].min():+.1f} / max {d[w].max():+.1f} mm", transform=axs[i, 2].transAxes, fontsize=7.5, color=INK,
                           bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
        fig.suptitle(title, fontsize=12)
        fig.savefig(path, dpi=80, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    ent = [(f"#{k_after}: {labels[k_after]}", C[k_before], C[k_after]) for k_before, k_after in ((80, 81), (81, 82), (82, 83), (83, 85))]
    ent.append((f"#86: {labels['86G']}", C[85], CG[86]))
    ent.append((f"#86: {labels['86L']}", C[85], CL[86]))
    action_sheet(ent, "Local-compensation commits (each: state before / after, real change, shaded, visual change) — crop 140 px around the largest change", out / "S4_fig13_comp_actions.png")
    ento = [(f"original history action #{k}", S(k - 1), S(k)) for k in range(81, 87)]
    action_sheet(ento, "Original history (no compensation): actions #81 to #86 in the same format", out / "S5_fig13_orig_actions.png")

    # ---------- 손실, 점수 ----------
    ks_o = list(range(80, 107))
    Lo = {k: losses(S(k)) for k in ks_o}
    eo = {k: score(S(k)) for k in ks_o}
    Lc = {k: losses(C[k]) for k in C}
    ec = {k: score(C[k]) for k in C}
    LG, LLg = losses(CG[86]), losses(CL[86])
    eG, eL = score(CG[86]), score(CL[86])

    def arm(name, k0):
        d = RES / f"ab_{name}"
        hs = [np.load(d / "start_m.npy").astype(np.float64)] + [np.load(d / "states" / f"step_{j:03d}.npy").astype(np.float64) for j in range(1, 21)]
        return [k0 + j for j in range(21)], [losses(h) for h in hs], [score(h)[0] for h in hs]
    arms = {n: arm(n, k0) for n, k0 in (("A1", 80), ("A2", 80), ("B1", 85), ("B2", 85))}

    # ---------- G1: 그림 11 형식 ----------
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.6), facecolor=SURF, layout="constrained")
    for ax, key, ttl in ((axs[0], "blur", "Band-limited (blur 0.4 tool radius): shape-level error"), (axs[1], "raw", "Raw pixels (as in the planner): particle texture included")):
        a2 = ax.twinx()
        ax.plot(ks_o, [Lo[k][f"l3d_{key}"] for k in ks_o], "o-", color=BLUE, ms=3.2, lw=0.9, alpha=0.5, label="3D loss, original history")
        a2.plot(ks_o, [Lo[k][f"lviz_{key}"] for k in ks_o], "o-", color=RED, ms=3.2, lw=0.9, alpha=0.5, label="visual loss, original history")
        for n in ("A1", "A2"):
            kk, ll, _ = arms[n]
            ax.plot(kk, [x[f"l3d_{key}"] for x in ll], "x", color=BLUE, ms=4.5, alpha=0.7, label="3D loss, no comp. re-runs (A1/A2)" if n == "A1" else None)
            a2.plot(kk, [x[f"lviz_{key}"] for x in ll], "x", color=RED, ms=4.5, alpha=0.7, label="visual loss, no comp. re-runs (A1/A2)" if n == "A1" else None)
        kc = sorted(Lc)
        ax.plot(kc, [Lc[k][f"l3d_{key}"] for k in kc], "D-", color="#0c3d7a", ms=5.5, lw=1.8, label="3D loss, compensation")
        a2.plot(kc, [Lc[k][f"lviz_{key}"] for k in kc], "D-", color="#8c1d1d", ms=5.5, lw=1.8, label="visual loss, compensation")
        for n, mk in (("B1", "^"), ("B2", "v")):
            kk, ll, _ = arms[n]
            ax.plot(kk, [x[f"l3d_{key}"] for x in ll], mk, color="#0c3d7a", ms=4, alpha=0.65, label="3D loss, compensation + continued (B1/B2)" if n == "B1" else None)
            a2.plot(kk, [x[f"lviz_{key}"] for x in ll], mk, color="#8c1d1d", ms=4, alpha=0.65, label="visual loss, compensation + continued (B1/B2)" if n == "B1" else None)
        for LL, mk, nm in ((LG, "*", "v12 grooves @86"), (LLg, "P", "v12 left leg @86")):
            ax.plot([86], [LL[f"l3d_{key}"]], mk, color="#0c3d7a", ms=11, mec="white")
            a2.plot([86], [LL[f"lviz_{key}"]], mk, color="#8c1d1d", ms=11, mec="white")
        ax.set_xlabel("total number of actions"); ax.set_ylabel("3D loss  (mean squared height error, mm^2)", color=BLUE)
        a2.set_ylabel("visual loss  (mean squared gradient error, (mm/px)^2)", color=RED)
        ax.grid(True, color=GRID, lw=0.6); ax.set_title(ttl, loc="left", fontsize=10)
        ax.axvline(80, color=MUTED, lw=0.7, ls=(0, (4, 3)))
        if key == "blur":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, fontsize=8, loc="outside lower center", ncol=4, framealpha=0.9)
    fig.suptitle("3D loss (blue) and visual loss (red) vs total actions, same format as the paper's Fig. 11: original vs local compensation (star / plus = v12 grooves / left-leg branch at 86)", fontsize=11)
    fig.savefig(out / "G1_loss_vs_actions.png", dpi=95, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---------- G2: 차이 그래프 ----------
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURF, layout="constrained")
    kc = [k for k in ROUNDS if k != 86]
    # (a) 영역 점수 e 차이
    ax = axs[0, 0]
    ax.plot(kc, [ec[k][0] - eo[k][0] for k in kc], "o-", color=ORANGE, lw=1.8, label="compensation chain (v8 -> v10)")
    ax.plot([86], [eG[0] - eo[86][0]], "D", color=ORANGE, ms=8, label="v12 grooves branch")
    ax.plot([86], [eL[0] - eo[86][0]], "s", color=GREEN, ms=8, label="v12 left-leg branch")
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_title("Region score: compensated - original (below 0 = closer to target A)", loc="left", fontsize=9.5)
    ax.set_xlabel("total number of actions"); ax.set_ylabel("score difference (80 = 3.00)"); ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=8)
    # (b) 영역별 평균 높이 차이 (보상 - 원래)
    ax = axs[0, 1]
    cols = {"groove_top": "#6a3d9a", "groove_bottom": "#e08a00", "left_leg": GREEN, "left_foot": BLUE, "right_foot": RED}
    for n in names:
        ax.plot(kc, [reg_mean(C[k])[n] - reg_mean(S(k))[n] for k in kc], "o-", color=cols[n], lw=1.5, label=n)
        ax.plot([86], [reg_mean(CG[86])[n] - reg_mean(S(86))[n]], "D", color=cols[n], ms=7)
        ax.plot([86.35], [reg_mean(CL[86])[n] - reg_mean(S(86))[n]], "s", color=cols[n], ms=7)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("Mean height per region: compensated - original (mm)", loc="left", fontsize=9)
    ax.set_xlabel("total number of actions (at 86: diamond = grooves branch, square = left-leg branch)"); ax.set_ylabel("mm (negative = compensated is lower)"); ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=8)
    # (c),(d) 손실 차이
    for ax, key, ttl in ((axs[1, 0], "l3d_blur", "3D loss: compensated - original (band-limited, mm^2)"), (axs[1, 1], "lviz_blur", "Visual loss: compensated - original (band-limited, (mm/px)^2)")):
        ax.plot(kc, [Lc[k][key] - Lo[k][key] for k in kc], "o-", color=ORANGE, lw=1.8, label="compensation chain")
        ax.plot([86], [LG[key] - Lo[86][key]], "D", color=ORANGE, ms=8, label="v12 grooves branch")
        ax.plot([86], [LLg[key] - Lo[86][key]], "s", color=GREEN, ms=8, label="v12 left-leg branch")
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_title(ttl, loc="left", fontsize=9.5)
        ax.set_xlabel("total number of actions"); ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=8)
    fig.suptitle("Differences between compensated and original at the same total action count", fontsize=12)
    fig.savefig(out / "G2_difference_graphs.png", dpi=95, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # ---------- 수치 ----------
    num = {"rounds": ROUNDS, "original": {str(k): {"e": eo[k][0], "per": eo[k][1], **Lo[k], "reg_dh_mm": reg_mean(S(k))} for k in ROUNDS},
           "compensated_chain": {str(k): {"e": ec[k][0], "per": ec[k][1], **Lc[k], "reg_dh_mm": reg_mean(C[k])} for k in kc},
           "v12_grooves_86": {"e": eG[0], "per": eG[1], **LG, "reg_dh_mm": reg_mean(CG[86])},
           "v12_left_leg_86": {"e": eL[0], "per": eL[1], **LLg, "reg_dh_mm": reg_mean(CL[86])},
           "continued_runs_at_end": {n: {"k_end": arms[n][0][-1], "e_end": arms[n][2][-1]} for n in arms}}
    (out / "numbers.json").write_text(json.dumps(num, indent=1))
    lines = [f"{'actions':>8} | {'e orig':>7} {'e comp':>7} | {'L3D blur orig':>13} {'comp':>8} | {'Lviz blur orig':>14} {'comp':>9} | {'L3D raw orig':>12} {'comp':>8} | {'Lviz raw orig':>13} {'comp':>9}"]
    for k in kc:
        lines.append(f"{k:>8} | {eo[k][0]:7.3f} {ec[k][0]:7.3f} | {Lo[k]['l3d_blur']:13.4f} {Lc[k]['l3d_blur']:8.4f} | {Lo[k]['lviz_blur']:14.5f} {Lc[k]['lviz_blur']:9.5f} | {Lo[k]['l3d_raw']:12.3f} {Lc[k]['l3d_raw']:8.3f} | {Lo[k]['lviz_raw']:13.4f} {Lc[k]['lviz_raw']:9.4f}")
    for tag, e_, L_ in (("86 grooves", eG, LG), ("86 left leg", eL, LLg)):
        lines.append(f"{tag:>8} | {eo[86][0]:7.3f} {e_[0]:7.3f} | {Lo[86]['l3d_blur']:13.4f} {L_['l3d_blur']:8.4f} | {Lo[86]['lviz_blur']:14.5f} {L_['lviz_blur']:9.5f} | {Lo[86]['l3d_raw']:12.3f} {L_['l3d_raw']:8.3f} | {Lo[86]['lviz_raw']:13.4f} {L_['lviz_raw']:9.4f}")
    lines.append("")
    lines.append("continued runs (end): " + ", ".join(f"{n}: {arms[n][0][-1]} actions e={arms[n][2][-1]:.3f}" for n in arms))
    (out / "summary_rounds.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print("saved", out)


if __name__ == "__main__":
    main()
