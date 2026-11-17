"""여러 팔(baseline 재계획, 설계 파이프라인 변형들)의 start/final 높이맵을 같은 마스크·같은 항으로 채점한다.

가중치는 결과를 본 뒤 고르기 쉬워서, L 하나만 보고하지 않는다. 항별 값(r,g,b,p)과, w_g=w_b 를 바꾼 여러 L 을 모두 낸다.
  r : 홈 깊이 오차 mean|H-H*| (M_edit, mm)          g: 홈 안쪽 기울기 오차     b: 가장자리 선명도 오차
  p : 보존 변화 mean|H-H_T| (M_preserve, mm)
  depth: M_edit 평균으로 얼마나 더 깊어졌나(mm; 목표는 평균 약 2.35mm)
사용: python eval_design_arms.py --spec-dir S --arm NAME=DIR [--arm ...] [--json out.json]
각 DIR 에는 start_m.npy, final_m.npy 가 있어야 한다 (continue_sculpt.py, local_correct.py 모두 만든다).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from edit_spec import load_spec

W_GRID = (2.0, 1.0, 0.5, 0.0)


def score(spec, h):
    t = spec.terms(torch.from_numpy(h)[None, None].float())
    d = (torch.from_numpy(h)[None, None].float() - spec.h_t).abs() * 1000
    zone = spec.m_edit | spec.m_boundary
    out = {k: float(v) for k, v in t.items()}
    out["depth"] = float(((spec.h_t - torch.from_numpy(h)[None, None].float()) * 1000)[spec.m_edit].mean())
    out["pres_max"] = float(d[spec.m_preserve].max())
    out["outside_mean"] = float(d[~zone].mean())
    out["outside_max"] = float(d[~zone].max())
    for w in W_GRID:
        out[f"L_w{w:g}"] = out["r"] + w * out["g"] + w * out["b"] + out["p"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    spec = load_spec(a.spec_dir)
    rows = {}
    for s in a.arm:
        name, d = s.split("=", 1)
        d = Path(d)
        rows[name] = dict(start=score(spec, np.load(d / "start_m.npy")), final=score(spec, np.load(d / "final_m.npy")))
        if (d / "steps.csv").exists():
            n = sum(1 for _ in open(d / "steps.csv")) - 2   # 헤더와 step 0(시작) 행 제외
            rows[name]["n_actions"] = n
    first = next(iter(rows.values()))["start"]
    print(f"시작(H_T): r={first['r']:.3f} g={first['g']:.3f} b={first['b']:.3f} p={first['p']:.4f} | 목표 깊이 약 {-float(((spec.h_star - spec.h_t) * 1000)[spec.m_edit].mean()):.2f}mm")
    print(f"{'팔':<26}{'행동':>4} | {'깊이+':>6}{'r':>7}{'g':>7}{'b':>7}{'p':>8}{'보존최대':>8}{'밖평균':>8}{'밖최대':>7} | " + " ".join(f"L(w={w:g})".rjust(9) for w in W_GRID))
    for name, v in rows.items():
        f = v["final"]
        print(f"{name:<26}{v.get('n_actions', '-'):>4} | {f['depth']:>6.2f}{f['r']:>7.3f}{f['g']:>7.3f}{f['b']:>7.3f}{f['p']:>8.4f}{f['pres_max']:>8.2f}{f['outside_mean']:>8.4f}{f['outside_max']:>7.2f} | "
              + " ".join(f"{f[f'L_w{w:g}']:>9.3f}" for w in W_GRID))
    print(f"{'(시작값 = 아무것도 안 함)':<26}{0:>4} | {first['depth']:>6.2f}{first['r']:>7.3f}{first['g']:>7.3f}{first['b']:>7.3f}{first['p']:>8.4f}{first['pres_max']:>8.2f}{first['outside_mean']:>8.4f}{first['outside_max']:>7.2f} | "
          + " ".join(f"{first[f'L_w{w:g}']:>9.3f}" for w in W_GRID))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
