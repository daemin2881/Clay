#!/usr/bin/env python3
"""control 재생이 기록된 현재 상태를 영역 창 안에서 얼마나 긴 구간까지 재현하는지 잰다 (--ctrl-valid-len 을 정하는 근거).

control = 과거 행동을 하나도 바꾸지 않고, 시작 행동 s 직전 상태에서 현재(K)까지 다시 실행한 결과.
이게 기록된 현재와 영역 창 안에서 맞으면(local_gate: 평균 어긋남 <= 0.15 mm, p99 <= 0.5 mm) 그 길이(K-s+1 행동)까지는 새로 재생하지 않고 기록을 control 로 써도 된다.
A@80 은 s=2 (79행동) 가 통과해서 --ctrl-valid-len 79 였다.

사용:
  python ctrl_len_check.py eval --history DIR --regions DIR --cache DIR [--upto K]     # 캐시에 이미 있는 s*_control.npy 를 영역별로 채점 (CPU)
  python ctrl_len_check.py run  --history DIR --cache DIR --starts 151,131 [--upto K]  # 새 control 재생을 캐시에 저장 (Genesis, 창 없음, 프로세스 하나가 순서대로)
같은 캐시 폴더(cf_replay_cache/K<K>)를 쓰면 이후 cf_compensate 가 그 control 을 그대로 이어받는다(ctrl_evidence).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_eval(a):
    from scipy import ndimage
    import cf_common as cc
    import first_beam as fb
    from region_set import RegionSet
    from cf_compensate import local_gate
    root = Path(a.history).resolve()
    man, actions, before, after = fb.load_history(root)
    K = a.upto or len(actions)
    g = cc.geometry(root)
    win_px = int(round(a.win_tools * g["tool_r_px"]))
    rs = RegionSet.load(Path(a.regions), names=None, ref_h=None, ref_default=cc.state_height(root))
    wins = {n: ndimage.binary_dilation(rs.masks[n], iterations=win_px) for n in rs.names}
    h_rec = after[K - 1].astype(np.float64)
    cache = Path(a.cache)
    rows = []
    for fcp in sorted(cache.glob("s*_control.npy")):
        m = re.fullmatch(r"s(\d+)_control\.npy", fcp.name)
        if not m:
            continue
        s = int(m.group(1))
        hc = np.load(fcp).astype(np.float64)
        dd = np.abs(hc - h_rec) * 1000
        per = {n: local_gate(hc, h_rec, wins[n], None, a.max_mae, a.max_p99) for n in rs.names}
        rows.append((K - s + 1, s, per, float(dd.mean()), float(np.quantile(dd, 0.999)), int((dd > 0.5).sum())))
    rows.sort()
    print(f"K={K} 캐시 {cache} | 영역 {len(rs.names)}개 | 창 폭 {win_px}px | 게이트: 영역 창 안 평균 <= {a.max_mae} mm, p99 <= {a.max_p99} mm")
    if not rows:
        print("  control 재생이 캐시에 없다")
        return
    print(f"  {'길이':>5} {'시작':>5} | 전역 평균/p99.9/0.5mm초과화소 | 영역별 평균(mm) (통과=o, 탈락=X)")
    best_all, best_any = 0, 0
    for L, s, per, gm, p999, nbad in rows:
        ok_all = all(v["ok"] for v in per.values())
        ok_n = sum(v["ok"] for v in per.values())
        cells = " ".join(f"{n}:{per[n]['mae']:.3f}{'o' if per[n]['ok'] else 'X'}" for n in rs.names)
        print(f"  {L:>5} #{s:<4} | {gm:.4f} / {p999:.3f} / {nbad:>5} | {ok_n}/{len(rs.names)} 통과: {cells}")
        if ok_all:
            best_all = max(best_all, L)
        if ok_n:
            best_any = max(best_any, L)
    print(f"  => 모든 영역 통과한 가장 긴 재생: {best_all}행동 | 하나라도 통과한 가장 긴 재생: {best_any}행동")


def cmd_run(a):
    import torch
    root = Path(a.history).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    det = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import vs_simulator as sim
    try:
        from vs_heightmap_fast import apply_fast_height
        apply_fast_height(sim)      # 입자->높이맵 변환의 빠른 버전(결과 동일, 파일은 안 바꿈). VS_FAST_HEIGHT=0 이면 끔
    except Exception as _e_fh:
        print(f"[빠른 높이맵 적용 실패: {_e_fh!r}] 원본 함수를 쓴다", flush=True)
    from vs_history import GenesisStateAdapter, physics_signature
    import counterfactual_replay as cfr
    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("physics signature 불일치")
    K = a.upto or len([d for d in root.glob("action_??????") if d.is_dir()])
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.starts.split(",") if x]:
        out = cache / f"s{s}_control.npy"
        if out.exists():
            print(f"[{time.strftime('%T')}] s{s} 이미 있음 (길이 {K - s + 1})", flush=True)
            continue
        t0 = time.time()
        pre, records, rngs = cfr.load_plan(root, s)
        h, _ = cfr.run_replay(adapter, pre, records, rngs, s, K, {})
        np.save(out, h.astype(np.float32))
        print(f"[{time.strftime('%T')}] s{s} control 재생 완료: 길이 {K - s + 1}행동, {time.time() - t0:.0f}초 -> {out.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "run"):
        p = sub.add_parser(name)
        p.add_argument("--history", required=True)
        p.add_argument("--cache", required=True)
        p.add_argument("--upto", type=int, default=0)
        if name == "eval":
            p.add_argument("--regions", required=True)
            p.add_argument("--win-tools", type=float, default=1.2)
            p.add_argument("--max-mae", type=float, default=0.15)
            p.add_argument("--max-p99", type=float, default=0.5)
        else:
            p.add_argument("--starts", required=True)
    a = ap.parse_args()
    (cmd_eval if a.cmd == "eval" else cmd_run)(a)


if __name__ == "__main__":
    main()
