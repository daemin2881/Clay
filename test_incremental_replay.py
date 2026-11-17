#!/usr/bin/env python3
"""반사실 세계 이어 붙이기가 처음부터 다시 재생하는 것과 같은 결과를 내는지 시험한다(Genesis 필요, 창 없음, 아무것도 저장하지 않음).

비교: 행동 S 직전 상태에서 K 까지 '한 번에' 재생한 결과 vs  K-1 까지 재생한 전체 물리 상태를 저장(capture)했다가 다른 상태로 갔다가 복원(restore)한 뒤 K 번째 행동만 실행한 결과.
원래대로(control)와 행동 S 를 skip 으로 바꾼 반사실, 두 경우를 본다. 차이가 재현 오차(0.001 mm 수준)면 이어 붙이기를 써도 정확도가 같다.
사용: python test_incremental_replay.py --history-dir <이력 폴더> --start 79
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--view-dir", type=Path, default=Path("results/_incr_view"))
    a = ap.parse_args()
    import cf_common as cc
    root = cc.make_view(a.history_dir, None, a.view_dir)
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, physics_signature, restore_rng
    import counterfactual_replay as cfr
    import cf_compensate as cfc
    if physics_signature(sim) != man["signature"]:
        raise SystemExit("physics signature 불일치")
    K = cc.n_actions(root)
    cfc.BOUNDS = cc.action_bounds()
    adapter = GenesisStateAdapter(sim)
    pre, records, rngs = cfr.load_plan(root, a.start)
    print(f"행동 #{a.start}~#{K} ({K - a.start + 1}개). 마지막 푸시 #{K}: {np.round(records[K]['action'], 4).tolist()}", flush=True)
    skip = {a.start: torch.tensor(cfc.variant_action(records[a.start]["action"], None, None), dtype=torch.float32)}

    def drift(x, y):
        d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)) * 1000
        return f"MAE {d.mean():.5f} mm | p99.9 {np.quantile(d, 0.999):.4f} mm | 최대 {d.max():.4f} mm | 0.5mm 초과 {int((d > 0.5).sum())}px"

    for name, edits in (("control(원래대로)", {}), (f"반사실(#{a.start} skip)", skip)):
        t0 = time.time()
        h_full, _ = cfr.run_replay(adapter, pre, records, rngs, a.start, K, edits)            # 처음부터 K 까지 한 번에
        t_full = time.time() - t0
        t0 = time.time()
        adapter.restore(pre)                                                                   # K-1 까지만 재생하고 전체 상태를 저장
        for n in range(a.start, K):
            restore_rng(rngs[n])
            adapter.execute(edits.get(n, torch.tensor(records[n]["action"])), records[n]["action_index"])
        snap = adapter.capture()
        adapter.restore(pre)                                                                   # 일부러 다른 상태로 갔다가
        adapter.restore(snap)                                                                  # 저장한 상태를 복원(캐시에서 불러온 것과 같다)
        restore_rng(rngs[K])
        adapter.execute(edits.get(K, torch.tensor(records[K]["action"])), records[K]["action_index"])   # 새 푸시만 이어서 실행
        h_inc = adapter.observe().numpy()
        t_inc = time.time() - t0
        print(f"[{name}] 처음부터 vs 이어 붙이기: {drift(h_full, h_inc)}   (소요 {t_full:.0f}초 / {t_inc:.0f}초)", flush=True)
        if not edits:
            print(f"   참고: 처음부터 재생 vs 기록된 #{K} 직후 높이맵: {drift(h_full, np.load(root / f'action_{K:06d}' / 'transition.npz')['after_m'])}", flush=True)


if __name__ == "__main__":
    main()
