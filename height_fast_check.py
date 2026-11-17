#!/usr/bin/env python3
"""vs_heightmap_fast.particles_to_height_gpu_fast 가 원본(vs_simulator.particles_to_height_gpu)과 같은 결과를 내는지, 얼마나 빠른지 실제 입자 위치로 검증한다.
이력의 여러 시점 스냅샷(pre_state.pt)을 복원해서 점토 입자 위치를 꺼내 두 함수에 넣고 비교한다 (Genesis, 창 없음, 푸시는 실행하지 않음).
사용: python height_fast_check.py --history DIR --actions 1,60,120,190
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--actions", default="1,60,120,190", help="이 번호 행동 직전 상태를 복원해 입자를 꺼낸다")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    root = Path(a.history).resolve()
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot
    from vs_heightmap_fast import particles_to_height_gpu_fast
    ad = GenesisStateAdapter(sim)
    allok = True
    for n in [int(x) for x in a.actions.split(",") if x]:
        ad.restore(load_snapshot(root / f"action_{n:06d}" / "pre_state.pt"))
        P = ad.simulator.clay.get_particles_pos()
        torch.cuda.synchronize()
        res = {}
        for name, fn in (("원본", sim.particles_to_height_gpu), ("빠른 버전", particles_to_height_gpu_fast)):
            h = fn(P)
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                h = fn(P)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            res[name] = (h, float(np.median(ts)))
        h0, t0 = res["원본"]
        h1, t1 = res["빠른 버전"]
        same = bool(torch.equal(h0, h1))
        d = (h0 - h1).abs()
        allok &= same
        print(f"행동 #{n} 직전 | 입자 {P.shape[0]:,}개 | 원본 {t0:.3f}초, 빠른 버전 {t1:.3f}초 ({t0 / max(t1, 1e-9):.1f}배) | 비트 단위로 같음: {same} | 최대 차이 {float(d.max()):.3e} m, 다른 화소 {int((d > 0).sum())}개", flush=True)
    print("=> 모든 시점에서 원본과 완전히 같음" if allok else "=> 원본과 다른 곳이 있음: 이 버전은 쓰지 않는다", flush=True)


if __name__ == "__main__":
    main()
