"""현재 상태에서 하는 국소 보상: 과거 반사실이 알려 준 '영역이 이렇게 됐을 것'을 현재 점토에서 실제 행동으로 만든다. 과정을 실시간으로 PNG 로 저장한다.

  보상 목표 T  = 현재 + soft(영역) * C*      (C* = H_cf - H_control, 과거 수정으로 생긴 영역 안 변화)
  후보         : 영역 주변의 (x, y, theta, 길이, 깊이) 를 여러 개 뽑는다 (--candidates).
  평가         : 후보를 시뮬레이터에서 실제로 실행해 보고 매번 되돌린다. 영역 주변 창의 MAE(T 대비)를 잰다. 영역 밖 변화가 --far-limit-mm 를 넘는 후보는 제외.
  확정         : 가장 좋은 후보가 현재보다 --eps-mm 이상 좋아질 때만 실제로 실행(뷰어 창에서 보인다). 아니면 종료.
  저장         : out/frames/frame_XXX.png (확정한 행동마다 1장: 현재 높이맵 | 목표 T | 현재-T), out/states/iter_XXX.npy, 마지막 상태 스냅샷.
사용: python live_compensate.py --history-dir H --cf-npy H_cf.npy --ctrl-npy H_ctrl.npy --region-spec SPEC --out DIR --viewer
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

X0, Y0, PS = 0.44 - 0.18, -0.18, 0.36 / 511


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--cf-npy", required=True, type=Path)
    ap.add_argument("--ctrl-npy", required=True, type=Path)
    ap.add_argument("--region-spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--reach-px", type=int, default=10)
    ap.add_argument("--far-limit-mm", type=float, default=6.0)
    ap.add_argument("--eps-mm", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", choices=["cf", "target"], default="cf",
                    help="cf: 보상 목표 = 현재 + 영역*C* (과거 수정 효과). target: 보상 목표 = 원래 목표 map 의 그 영역(region-spec 의 h_star)")
    ap.add_argument("--hint-actions", default="", help="과거에 그 영역을 눌렀던 행동 번호들(쉼표). 그 행동을 얕게/반대로/옆으로 바꾼 후보를 만든다")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()
    root = a.history_dir.resolve()
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if a.viewer else "0"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature, tensor_save

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != man["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")
    last = len([d for d in root.glob("action_??????") if d.is_dir()])
    adapter.restore(load_snapshot(root / f"action_{last:06d}" / "post_state.pt"))
    h0 = adapter.observe().numpy().astype(np.float64)
    m_raw = np.load(a.region_spec / "m_edit.npy") > 0
    h_star = np.load(a.region_spec / "h_star.npy").astype(np.float64)
    M = ndimage.binary_dilation(m_raw, iterations=3)
    soft = np.clip(ndimage.distance_transform_edt(M) / 4.0, 0, 1)
    C = np.load(a.cf_npy).astype(np.float64) - np.load(a.ctrl_npy).astype(np.float64)
    T = h0 + soft * C if a.objective == "cf" else h_star.copy()
    win = ndimage.binary_dilation(M, iterations=25)
    reach = ndimage.binary_dilation(M, iterations=a.reach_px)
    ridx = np.flatnonzero(reach.ravel())
    rg = np.random.default_rng(a.seed)
    (a.out / "frames").mkdir(parents=True, exist_ok=True)
    (a.out / "states").mkdir(exist_ok=True)
    ys, xs = np.where(M)
    r0, r1, c0, c1 = max(ys.min() - 50, 0), min(ys.max() + 50, 512), max(xs.min() - 50, 0), min(xs.max() + 50, 512)
    bg = float(np.median(np.concatenate([h0[0], h0[-1], h0[:, 0], h0[:, -1]])))
    score = lambda h: float(np.abs(h - T)[win].mean()) * 1000
    mae_star = lambda h: float(np.abs(h - h_star)[m_raw].mean()) * 1000

    def frame(h, idx, note):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#fcfcfb")
        kw = dict(cmap="turbo", vmin=-10, vmax=14)
        ax[0].imshow(((h - bg) * 1000)[r0:r1, c0:c1], **kw); ax[0].set_title(f"Current height map (mm above bg) - {note}", fontsize=9)
        ax[1].imshow(((T - bg) * 1000)[r0:r1, c0:c1], **kw); ax[1].set_title("Compensation target T (from the counterfactual past)", fontsize=9)
        im = ax[2].imshow(((h - T) * 1000)[r0:r1, c0:c1], cmap="RdBu", vmin=-4, vmax=4); ax[2].set_title("Current - T (mm)", fontsize=9)
        for x_ in ax:
            x_.contour(m_raw[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=0.9, linestyles="--"); x_.set_xticks([]); x_.set_yticks([])
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        fig.suptitle(f"step {idx}: window MAE vs T = {score(h):.3f} mm | region MAE vs original target = {mae_star(h):.3f} mm", fontsize=10)
        fig.savefig(a.out / "frames" / f"frame_{idx:03d}.png", dpi=90, facecolor="#fcfcfb", bbox_inches="tight")
        plt.close(fig)

    frame(h0, 0, "start (state 80)")
    np.save(a.out / "states/iter_000.npy", h0.astype(np.float32))
    print(f"시작: 창 MAE(T 대비) {score(h0):.3f}mm, 영역 MAE(원래 목표 대비) {mae_star(h0):.3f}mm | 영역 안 보상 목표 평균 변화 {float((T - h0)[m_raw].mean() * 1000):+.2f}mm", flush=True)
    hints = []
    for n_ in [int(x) for x in a.hint_actions.split(",") if x.strip()]:
        x_, y_, th_, L_, d_ = json.loads((root / f"action_{n_:06d}" / "record.json").read_text())["action"]
        px, py = -math.sin(th_), math.cos(th_)
        wrap = lambda t: math.atan2(math.sin(t), math.cos(t))
        hints += [[x_, y_, th_, L_, d_ * 0.5], [x_, y_, th_, L_, d_],
                  [x_ + L_ * math.cos(th_), y_ + L_ * math.sin(th_), wrap(th_ + math.pi), L_, d_ * 0.5],
                  [x_ + 0.004 * px, y_ + 0.004 * py, th_, L_, d_ * 0.5], [x_ - 0.004 * px, y_ - 0.004 * py, th_, L_, d_ * 0.5]]
    log = []
    cur = h0
    for it in range(1, a.iters + 1):
        saved = adapter.capture()
        s_now = score(cur)
        cands = []
        for _ in range(a.candidates):
            k = int(rg.choice(ridx))
            r_, c_ = divmod(k, 512)
            th = float(rg.uniform(-math.pi, math.pi))
            L, d = [(0.008, 0.004), (0.008, 0.008), (0.012, 0.006), (0.006, 0.010)][int(rg.integers(0, 4))]
            cands.append([X0 + c_ * PS, Y0 + r_ * PS, th, L, d])
        if it == 1:
            cands = hints + cands
        res = []
        for c in cands:
            adapter.execute(torch.tensor(c, dtype=torch.float32), it)
            h = adapter.observe().numpy().astype(np.float64)
            res.append((score(h), float(np.abs(h - cur)[~win].max()) * 1000, mae_star(h), c))
            adapter.restore(saved)
        ok = [r for r in res if r[1] <= a.far_limit_mm]
        res.sort(key=lambda r: r[0])
        print(f"[{it}] 후보 {len(res)}개를 실제로 시도 후 되돌림. 현재 창 MAE {s_now:.3f}mm. 상위 3개: " +
              ", ".join(f"{r[0]:.3f}(밖 {r[1]:.1f}mm)" for r in res[:3]), flush=True)
        best = min(ok, key=lambda r: r[0]) if ok else None
        if best is None or best[0] > s_now - a.eps_mm:
            print(f"    개선되는 후보가 없어 종료 (최선 {res[0][0]:.3f} vs 현재 {s_now:.3f})", flush=True)
            break
        adapter.execute(torch.tensor(best[3], dtype=torch.float32), it)      # 확정(뷰어에서 보인다)
        cur = adapter.observe().numpy().astype(np.float64)
        np.save(a.out / f"states/iter_{it:03d}.npy", cur.astype(np.float32))
        frame(cur, it, f"after compensation action {it}")
        log.append(dict(iter=it, action=best[3], window_mae=best[0], region_mae_vs_target=best[2], far_max_mm=best[1]))
        print(f"    확정: x={best[3][0]:.4f} y={best[3][1]:.4f} th={best[3][2]:+.2f} L={best[3][3] * 1000:.0f}mm d={best[3][4] * 1000:.0f}mm -> 창 MAE {s_now:.3f} -> {best[0]:.3f}, "
              f"영역 MAE(원래 목표) {mae_star(cur):.3f}mm, 영역 밖 최대 {best[1]:.1f}mm | 프레임 저장", flush=True)
    np.save(a.out / "start_m.npy", h0.astype(np.float32))
    np.save(a.out / "final_m.npy", cur.astype(np.float32))
    tensor_save(a.out / "final_state.pt", adapter.capture())
    (a.out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(f"완료: 확정한 보상 행동 {len(log)}개 | 창 MAE(T) {score(h0):.3f} -> {score(cur):.3f}mm | 영역 MAE(원래 목표) {mae_star(h0):.3f} -> {mae_star(cur):.3f}mm | 저장: {a.out}")


if __name__ == "__main__":
    main()
