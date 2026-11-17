"""정답을 아는 시험(planted counterfactual)으로 First-Beam을 채점한다 (CPU, Genesis 불필요).

시험 구성: counterfactual_replay.py --segment S,E 가 만든 heightmaps.npz 의 h_x{배율} 을 H_edit 로 쓴다.
정답은 S~E 구간(바꾼 구간). First-Beam에는 기록된 이력과 H_edit, ROI만 준다.

채점
  - top1_hit   : 1위 후보가 정답 구간과 3개 이상 행동이 겹치는가
  - top3_hit   : 상위 3개(겹치지 않는 Top-K) 중 하나가 그런가
  - exact_rank : 정답 구간과 똑같은 윈도우의 전체 순위 (1이 최고)
  - chance     : 우연히 1위가 맞을 확률 = (정답과 3개 이상 겹치는 윈도우 수) / (전체 윈도우 수)
  - direction  : 대리 모델이 고른 배율 방향이 실제와 같은가 (배율 1.5 계획 → 1.3, 0.6 계획 → 0.7이 맞는 답)

사용: python eval_first_beam_planted.py --history-dir H --roi ROI --planted DIR [--planted DIR ...] --lookback 20
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import first_beam as fb


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--roi", required=True, type=Path)
    ap.add_argument("--planted", action="append", required=True, type=Path, help="seg{S}_{E} 폴더")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-overlap", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    root = a.history_dir.expanduser().resolve()
    rows = []
    for d in a.planted:
        d = d.expanduser().resolve()
        S, E = (int(v) for v in re.search(r"seg(\d+)_(\d+)", d.name).groups())
        z = np.load(d / "heightmaps.npz")
        for key in sorted(k for k in z.files if k.startswith("h_x") and k != "h_x1"):
            f = float(key[3:])
            h_edit = z[key].astype(np.float32)
            res, _ = fb.first_beam(root, h_edit, lookback=a.lookback, seq_len=a.seq_len,
                                   top_k=a.top_k, roi_path=a.roi)
            seqs = sorted(res["all_sequences"], key=lambda s: -s["score"])
            ov = [overlap(s["start"], s["end"], S, E) for s in seqs]
            n_good = sum(o >= a.min_overlap for o in ov)
            exact = next((i + 1 for i, s in enumerate(seqs) if s["start"] == S and s["end"] == E), None)
            top1 = res["beams"][0]
            top1_hit = overlap(top1["start_action"], top1["end_action"], S, E) >= a.min_overlap
            top3_hit = any(overlap(b["start_action"], b["end_action"], S, E) >= a.min_overlap for b in res["beams"])
            want = 1.3 if f > 1 else 0.7
            rows.append(dict(segment=f"{S}-{E}", planted_factor=f, n_windows=len(seqs), exact_rank=exact,
                             top1=f"{top1['start_action']}-{top1['end_action']}", top1_hit=top1_hit, top3_hit=top3_hit,
                             chance=n_good / len(seqs),
                             beams=[f"{b['start_action']}-{b['end_action']}" for b in res["beams"]],
                             surrogate_factor=top1["surrogate_best_factor"],
                             direction_ok=(top1["surrogate_best_factor"] == want),
                             D_roi_mae_mm=res["baseline"]["roi_mae_mm"], D_global_mae_mm=res["baseline"]["global_mae_mm"]))
    print(f"{'정답구간':>8} {'계획배율':>6} | 1위 | 1위적중 | Top3적중 | 정답윈도우 순위 | 우연확률 | 상위3 | 방향 | D(ROI mm)")
    for r in rows:
        print(f"{r['segment']:>8} {r['planted_factor']:>6g} | {r['top1']:>7} | {str(r['top1_hit']):>5} | {str(r['top3_hit']):>6} | "
              f"{str(r['exact_rank']):>3}/{r['n_windows']} | {r['chance']:.0%} | {','.join(r['beams'])} | "
              f"{'맞음' if r['direction_ok'] else '틀림'}({r['surrogate_factor']}) | {r['D_roi_mae_mm']:.3f}")
    n = len(rows)
    print(f"\n요약: 1위 적중 {sum(r['top1_hit'] for r in rows)}/{n}, Top3 적중 {sum(r['top3_hit'] for r in rows)}/{n}, "
          f"방향 맞음 {sum(r['direction_ok'] for r in rows)}/{n}, 우연 기대 1위 적중 {sum(r['chance'] for r in rows):.1f}/{n}")
    if a.out:
        a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
