#!/usr/bin/env python3
"""Genesis 창을 띄워 검증 과정을 눈으로 본다 (돌고 있는 실험과 별개의 프로세스, 아무것도 저장하지 않는다).

두 가지 모드
  push    K번째 행동 직후 상태를 복원하고 주어진 푸시(들)를 실행한다 = 확정한 보상 푸시가 실제로 어떻게 눌리는지(검증 실행).
            python watch_replay.py push --history-dir H --upto 80 --actions-from results/case_A/chain_smoke2/d0/D5_s1/cf/rounds.json
            python watch_replay.py push --history-dir H --upto 80 --actions-json '[[x,y,theta,L,d]]'
  replay  행동 S 직전 상태에서 K 까지 기록된 행동을 다시 재생한다 = 반사실 재생(control). --variant 를 주면 행동 S 만 바꿔서 재생한다(반사실).
            python watch_replay.py replay --history-dir H --upto 80 --start 73 [--variant skip|1:1.5|1.3:1]
  variant 표기는 cf_compensate 의 --cf-variants 와 같다: skip = 거의 안 누름, 길이배율:깊이배율.
끝나면 --hold 초 동안 창을 유지한다. 결과 높이맵이 기록과 얼마나 맞는지(control)나 얼마나 변했는지를 터미널에 적는다.
DISPLAY 가 필요하다(예: DISPLAY=:1). 창 렌더링 때문에 보통보다 느리다.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["push", "replay"])
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int, default=None)
    ap.add_argument("--actions-json", default=None, help="push: 실행할 행동 목록 JSON [[x,y,theta,L,d], ...]")
    ap.add_argument("--actions-from", type=Path, default=None, help="push: cf_compensate rounds.json 에서 확정한 행동을 읽는다")
    ap.add_argument("--start", type=int, default=None, help="replay: 재생을 시작할 행동 번호(1부터)")
    ap.add_argument("--variant", default=None, help="replay: 행동 --start 만 바꾼다. skip 또는 길이배율:깊이배율")
    ap.add_argument("--hold", type=float, default=40.0, help="끝난 뒤 창을 유지하는 시간(초)")
    ap.add_argument("--view-dir", type=Path, default=Path("results/_watch_view"))
    a = ap.parse_args()
    import cf_common as cc
    import first_beam as fb
    root = cc.make_view(a.history_dir, a.upto, a.view_dir)
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1"
    if not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY 가 없습니다. 예: DISPLAY=:1 python watch_replay.py ...")
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature
    import counterfactual_replay as cfr
    if physics_signature(sim) != man["signature"]:
        raise SystemExit("physics signature 불일치: 이력을 만든 코드/설정/버전과 다릅니다.")
    _, actions, before, after = fb.load_history(root)
    K = len(actions)
    adapter = GenesisStateAdapter(sim)
    if a.mode == "push":
        acts = None
        if a.actions_json:
            acts = json.loads(a.actions_json)
        elif a.actions_from:
            rounds = [r for r in json.loads(a.actions_from.read_text()) if r.get("committed")]
            acts = []
            for r in rounds:
                act = r["action"]
                acts += act if isinstance(act[0], (list, tuple)) else [act]
        if not acts:
            raise SystemExit("실행할 행동이 없습니다(--actions-json 또는 --actions-from)")
        adapter.restore(load_snapshot(root / f"action_{K:06d}" / "post_state.pt"))
        h0 = adapter.observe().numpy().astype(np.float64)
        print(f"[push] 행동 {K} 직후 상태 복원(기록 대비 MAE {np.abs(h0 - after[K - 1]).mean() * 1000:.4f} mm). 푸시 {len(acts)}개 실행:", flush=True)
        for i, act in enumerate(acts, 1):
            t0 = time.time()
            print(f"   {i}/{len(acts)}: x={act[0]:.4f} y={act[1]:.4f} th={act[2]:+.2f} L={act[3] * 1000:.1f}mm d={act[4] * 1000:.1f}mm", flush=True)
            adapter.execute(torch.tensor(act, dtype=torch.float32), K + i - 1)
            print(f"      완료 ({time.time() - t0:.0f}초)", flush=True)
        h1 = adapter.observe().numpy().astype(np.float64)
        d = (h1 - h0) * 1000
        print(f"[push] 결과: 변화 최대 +{d.max():.2f} / {d.min():.2f} mm, 평균 |변화| {np.abs(d).mean():.3f} mm", flush=True)
    else:
        if a.start is None:
            raise SystemExit("replay 모드는 --start 가 필요합니다")
        import cf_compensate as cfc
        cfc.BOUNDS = cc.action_bounds()
        pre, records, rngs = cfr.load_plan(root, a.start)
        edits = {}
        if a.variant:
            vname, lf, df = cfc.parse_cf_variants(a.variant)[0]
            edits[a.start] = torch.tensor(cfc.variant_action(actions[a.start - 1].tolist(), lf, df), dtype=torch.float32)
            print(f"[replay] 행동 #{a.start} 를 '{vname}' 로 바꿔서 #{K} 까지 재생(반사실)", flush=True)
        else:
            print(f"[replay] 행동 #{a.start}~#{K} 를 기록 그대로 재생(control)", flush=True)
        t0 = time.time()
        h, _ = cfr.run_replay(adapter, pre, records, rngs, a.start, K, edits)
        d = (h.astype(np.float64) - after[K - 1].astype(np.float64)) * 1000
        print(f"[replay] 완료 ({time.time() - t0:.0f}초). 기록된 현재 대비 변화: 평균 |차이| {np.abs(d).mean():.3f} mm, 최대 {np.abs(d).max():.2f} mm "
              + ("(control: 0 에 가까울수록 재생이 믿을 만함)" if not a.variant else "(반사실: 이 차이가 '그때 달리 했다면' 지금이 달라지는 양)"), flush=True)
    print(f"창을 {a.hold:.0f}초 유지합니다.", flush=True)
    time.sleep(a.hold)


if __name__ == "__main__":
    main()
