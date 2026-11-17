"""롤백 탐침: 완성된 A 의 마지막 상태에서 후보 행동들을 Genesis 로 실제 실행해 보고 매번 원래 상태로 되돌린다.

모델 예측이 아니라 실제 물리 결과로, 눈에 보이는 오차(L3D 형태, Lviz 외형)가 줄어드는 행동이 이 도구로 존재하는지 본다.
이력 폴더는 읽기만 하고 아무것도 쓰지 않는다(결과 JSON 만 --out 에 저장).
사용: python probe_actions.py --history-dir H --spec-dir S --out results.json [--l 5,8 --d 3,6,10 --theta 0,1.57]
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--offsets-px", default="0:0,-8:0,8:0,0:-8,0:8", help="편집영역 중심에서의 (dx:dy) 화소 오프셋")
    ap.add_argument("--l", default="5", help="길이(mm)")
    ap.add_argument("--d", default="3,6,10", help="깊이(mm)")
    ap.add_argument("--theta", default="0,1.5708")
    ap.add_argument("--random", type=int, default=0, help="N>0 이면 격자 대신 무작위 N개 (위치·방향·길이·깊이)")
    ap.add_argument("--reach-px", type=float, default=10.0, help="무작위 위치 범위: 편집영역 중심에서 이 반경(화소) 안")
    ap.add_argument("--n-theta", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = a.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import vs_simulator as sim
    import vs_config as cfg
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature
    from visual_score import VisualScorer

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")
    n = len([d for d in root.glob("action_??????") if d.is_dir()])
    adapter.restore(load_snapshot(root / f"action_{n:06d}" / "post_state.pt"))
    base = adapter.capture()
    start = adapter.observe().numpy().astype(np.float64)
    spec = a.spec_dir
    h_star = np.load(spec / "h_star.npy")
    me = np.load(spec / "m_edit.npy") > 0
    sc = VisualScorer(h_star, start, me)
    rows_, cols_ = np.where(me)
    cy, cx = rows_.mean(), cols_.mean()
    pix = cfg.CLAY_SIZE_X / (cfg.HEIGHT_MAP_RES - 1)
    x0, y0 = cfg.CLAY_CENTER_X - cfg.CLAY_SIZE_X / 2, cfg.CLAY_CENTER_Y - cfg.CLAY_SIZE_Y / 2
    offs = [tuple(float(v) for v in s.split(":")) for s in a.offsets_px.split(",")]
    Ls, Ds, Ts = ([float(v) for v in s.split(",")] for s in (a.l, a.d, a.theta))
    cands = [(ox, oy, t, l, d) for (ox, oy), t, l, d in itertools.product(offs, Ts, Ls, Ds)]
    if a.random > 0:
        rg = np.random.default_rng(a.seed)
        ths = np.linspace(-np.pi, np.pi, a.n_theta, endpoint=False)
        cands = []
        while len(cands) < a.random:
            ox, oy = rg.uniform(-a.reach_px, a.reach_px, 2)
            if np.hypot(ox, oy) <= a.reach_px:
                cands.append((float(ox), float(oy), float(rg.choice(ths)), float(rg.choice(Ls)), float(rg.choice(Ds))))
    print(f"편집영역 중심 (row,col)=({cy:.0f},{cx:.0f}), 후보 {len(cands)}개")
    res = []
    t_all = time.perf_counter()
    for i, (ox, oy, t, l, d) in enumerate(cands):
        act = torch.tensor([x0 + (cx + ox) * pix, y0 + (cy + oy) * pix, t, l / 1000, d / 1000])
        t0 = time.perf_counter()
        adapter.execute(act, 0)
        h = adapter.observe().numpy().astype(np.float64)
        r = sc.summary(h, start)
        adapter.restore(base)
        back = float(np.abs(adapter.observe().numpy() - start).max()) * 1000
        r.update(offset_px=[ox, oy], theta=t, l_mm=l, d_mm=d, seconds=time.perf_counter() - t0, restore_err_mm=back)
        res.append(r)
        print(f"[{i + 1:3d}/{len(cands)}] off=({ox:+.0f},{oy:+.0f}) th={t:.2f} l={l:.0f} d={d:.0f} | 깊이 +{r['deepen_mm']:.2f}mm | "
              f"창 L3D×{r['l3d_win']:.2f} Lviz×{r['lviz_win']:.2f} | 전체 L3D×{r['l3d_full']:.2f} Lviz×{r['lviz_full']:.2f} | "
              f"창 밖 최대 {r['outside_win_max_mm']:.2f}mm | {r['seconds']:.0f}s (복원오차 {back:.4f}mm)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    best = sorted(res, key=lambda r: r["lviz_win"] + r["l3d_win"])[:5]
    print("\n창 L3D+Lviz 기준 상위 5개 (1.0 = 아무것도 안 함):")
    for r in best:
        print(f"  off={r['offset_px']} th={r['theta']:.2f} l={r['l_mm']:.0f} d={r['d_mm']:.0f}: 창 L3D×{r['l3d_win']:.2f} Lviz×{r['lviz_win']:.2f}, 깊이 +{r['deepen_mm']:.2f}mm")
    print(f"총 {time.perf_counter() - t_all:.0f}s | 저장: {a.out}")


if __name__ == "__main__":
    main()
