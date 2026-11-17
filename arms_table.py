"""모든 팔을 '아무것도 안 한 것(1.0)'과 눈에 보이는 지표로 비교한다. 1.0 미만 = 목표 height map 에 더 가까워짐.
  형태 L3D / 외형 Lviz / 테두리 edge 비율 (홈 주변 창), 같은 L3D·Lviz 의 전체 지도 비율,
  절대값: 창 높이오차 RMS(mm), 창 기울기오차(목표 기울기 대비 %), 홈이 평균 몇 mm 깊어졌나(목표 약 2.35), 창 밖 최대 변화(mm).
사용: python arms_table.py --spec-dir S --arm NAME=DIR [--arm ...] [--json out.json]   (DIR 에 start_m.npy, final_m.npy)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from visual_score import VisualScorer, _grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    h_star = np.load(a.spec_dir / "h_star.npy")
    me = np.load(a.spec_dir / "m_edit.npy") > 0
    arms = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.arm]
    start = np.load(arms[0][1] / "start_m.npy")
    sc = VisualScorer(h_star, start, me)
    hs = h_star.astype(np.float64) * 1000
    gx, gy = _grad(hs)
    gref = np.sqrt((gx ** 2 + gy ** 2)[sc.win[1:-1, 1:-1]].mean())
    rows = {}
    for name, d in [("아무것도 안 함", None)] + arms:
        h = start if d is None else np.load(d / "final_m.npy")
        r = sc.summary(h, start)
        r["edge_win"] = sc.ratios(h)["edge_win"]
        hm = h.astype(np.float64) * 1000
        r["rms_win_mm"] = float(np.sqrt(((hm - hs) ** 2)[sc.win].mean()))
        g2 = _grad(hm)
        r["grad_err_pct"] = float(100 * np.sqrt(((g2[0] - gx) ** 2 + (g2[1] - gy) ** 2)[sc.win[1:-1, 1:-1]].mean()) / gref)
        rows[name] = r
    print(f"{'':<28}| {'형태 L3D':>8}{'외형 Lviz':>10}{'테두리':>8} (창, 1.0=변화없음) | {'전체 L3D':>8}{'전체 Lviz':>10} | {'창 RMS':>7}{'기울기%':>8}{'깊어짐':>7}{'먼곳최대':>8}")
    for n, r in rows.items():
        print(f"{n:<28}| {r['l3d_win']:>8.2f}{r['lviz_win']:>10.2f}{r['edge_win']:>8.2f}{'':>22}| {r['l3d_full']:>8.2f}{r['lviz_full']:>10.2f} | "
              f"{r['rms_win_mm']:>5.2f}mm{r['grad_err_pct']:>7.0f}%{r['deepen_mm']:>6.2f}mm{r['outside_win_max_mm']:>6.1f}mm")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
