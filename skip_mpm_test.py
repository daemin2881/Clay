#!/usr/bin/env python3
"""접촉 없는 구간에서 점토(MPM) 계산을 건너뛰어도 결과가 같은지, 얼마나 빨라지는지 시험한다 (Genesis, 창 없음, 프로세스 하나).

배경: 푸시 하나 = 약 852스텝(이동 300, 하강 200, 누르기 12, 상승 300, 유지 20, 안정화 20), 스텝마다 MPM 16 substep.
도구가 점토 위 6 cm 에서 움직이는 구간에서는 점토가 가만히 있는데도 MPM 을 계산한다.
Genesis 는 활성 solver 가 강체뿐일 때 쓰는 '강체만 계산' 경로(Simulator._rigid_only)가 있다. 그 플래그를 접촉 없는 스텝에서만 켠다.
  - move_tip(이동, 호버에서 수렴 확인)의 모든 스텝: 건너뜀
  - linear_tip 하강(steps=DOWN_STEPS): 앞쪽 (1-keep_down) 비율만 건너뜀, 마지막 keep_down 은 점토 표면 접근이라 계산
  - linear_tip 상승(steps=UP_STEPS): 처음 keep_up 비율은 점토와 닿아 있어 계산, 나머지는 건너뜀
  - 누르기, 유지, 안정화는 전부 계산
비교: 같은 시작 상태에서 (A) 전부 계산 (B) 건너뛰기, 그리고 기록된 결과와 각각 비교. 다음 행동 n개를 이어서 돌려 어긋남이 쌓이는지도 본다.

사용: python skip_mpm_test.py --history DIR --start 181 --n 10 [--keep-down 0.1 --keep-up 0.4] [--skip-all-free]
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, default=1, help="이어서 재생할 행동 수")
    ap.add_argument("--keep-down", type=float, default=0.10)
    ap.add_argument("--keep-up", type=float, default=0.40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.history).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    det = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import torch
    import vs_simulator as sim
    import vs_config as cfg
    from vs_history import GenesisStateAdapter, physics_signature
    import counterfactual_replay as cfr
    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("physics signature 불일치")
    S = a.start
    E = S + a.n - 1
    pre, records, rngs = cfr.load_plan(root, S)
    rec = np.load(root / f"action_{E:06d}" / "transition.npz")["after_m"].astype(np.float64)

    Cls = sim.VisualSculptingSimulator
    state = dict(mode=None, i=0, n=0, skipped=0, total=0)
    orig_step, orig_move, orig_lin = Cls.step, Cls.move_tip, Cls.linear_tip
    DOWN, UP = int(cfg.DOWN_STEPS), int(cfg.UP_STEPS)

    def step(self, *args, **kw):
        s = state
        skip = False
        if s["mode"] == "travel":
            skip = True
        elif s["mode"] == "down":
            skip = s["i"] < int((1.0 - a.keep_down) * s["n"])
        elif s["mode"] == "up":
            skip = s["i"] >= int(a.keep_up * s["n"])
        s["i"] += 1
        s["total"] += 1
        if skip:
            s["skipped"] += 1
            self.scene.sim._rigid_only = True
            try:
                return orig_step(self, *args, **kw)
            finally:
                self.scene.sim._rigid_only = False
        return orig_step(self, *args, **kw)

    def move_tip(self, target, steps):
        prev = dict(state)
        state.update(mode="travel", i=0, n=int(steps))
        try:
            return orig_move(self, target, steps)
        finally:
            state.update(mode=prev["mode"], i=prev["i"], n=prev["n"])

    def linear_tip(self, start, end, steps):
        prev = dict(state)
        n = max(1, int(steps))
        mode = "down" if n == DOWN else ("up" if n == UP else None)
        state.update(mode=mode, i=0, n=n)
        try:
            return orig_lin(self, start, end, steps)
        finally:
            state.update(mode=prev["mode"], i=prev["i"], n=prev["n"])

    def run(label, patched):
        if patched:
            Cls.step, Cls.move_tip, Cls.linear_tip = step, move_tip, linear_tip
        else:
            Cls.step, Cls.move_tip, Cls.linear_tip = orig_step, orig_move, orig_lin
        state.update(mode=None, i=0, n=0, skipped=0, total=0)
        t0 = time.perf_counter()
        h, _ = cfr.run_replay(adapter, pre, records, rngs, S, E, {})
        sec = time.perf_counter() - t0
        Cls.step, Cls.move_tip, Cls.linear_tip = orig_step, orig_move, orig_lin
        h = np.asarray(h, np.float64)
        d = np.abs(h - rec) * 1000
        return h, sec, dict(vs_record_mae=float(d.mean()), vs_record_p999=float(np.quantile(d, 0.999)), vs_record_max=float(d.max()),
                            skipped=state["skipped"], total=state["total"])

    print(f"[{time.strftime('%T')}] 행동 #{S}~#{E} ({a.n}개) 시험: keep_down={a.keep_down} keep_up={a.keep_up}", flush=True)
    hA, tA, mA = run("full", False)
    print(f"[{time.strftime('%T')}] (A) 전부 계산: {tA:.1f}초 ({tA / a.n:.1f}초/행동) | 기록 대비 평균 {mA['vs_record_mae']:.4f} mm, p99.9 {mA['vs_record_p999']:.3f} mm, 최대 {mA['vs_record_max']:.2f} mm", flush=True)
    hB, tB, mB = run("skip", True)
    dAB = np.abs(hA - hB) * 1000
    print(f"[{time.strftime('%T')}] (B) 접촉 없는 구간 건너뜀: {tB:.1f}초 ({tB / a.n:.1f}초/행동) | 건너뛴 스텝 {mB['skipped']}/{mB['total']} ({100 * mB['skipped'] / max(mB['total'], 1):.0f}%)", flush=True)
    print(f"      기록 대비 평균 {mB['vs_record_mae']:.4f} mm, p99.9 {mB['vs_record_p999']:.3f} mm, 최대 {mB['vs_record_max']:.2f} mm", flush=True)
    print(f"      (A)와 (B) 차이 평균 {float(dAB.mean()):.4f} mm, p99.9 {float(np.quantile(dAB, 0.999)):.3f} mm, 최대 {float(dAB.max()):.2f} mm", flush=True)
    print(f"      => 속도 {tA / tB:.2f}배 | 기록 대비 평균 어긋남 {mA['vs_record_mae']:.4f} -> {mB['vs_record_mae']:.4f} mm", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, full=hA.astype(np.float32), skip=hB.astype(np.float32), rec=rec.astype(np.float32))


if __name__ == "__main__":
    main()
