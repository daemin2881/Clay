#!/usr/bin/env python3
"""ab_cf_arms.py 로 돌린 팔(T / J)의 결과를 같은 기준으로 채점한다(CPU). 시작 상태 대비 확정 직후 상태(final_m.npy)를 저장된 영역 목표로 잰다.

[2026-11-17 변경: 팔 2개를 병렬로 돌리려고 모든 팔(J, M, V, T 시드 1~3)을 --workers 2 로 맞춰 다시 돌린다. 체인의 4단계(워커 3)는 쓰지 않고 T 시드 1 도 팔로 재실행한다. 판정 기준은 그대로.]
[방법 선택 규칙 - 2026-11-17 02:25, 결과를 보기 전에 정함. 비용은 J 약 30분 = M 약 30분 < V 약 60분 < T 약 120분 (단계당)]
  1단계  T 가 J 보다 낫다는 사전 등록 기준(위 (1)~(3))을 넘는가?
         - 넘으면 반사실이 도움이 된다고 본다 -> 2단계로.
         - 못 넘으면 반사실의 이득을 이 설정에서 보이지 못한 것이다 -> [사용자 지시 2026-11-17 02:30] 원래 방법(T)을 그대로 쓰고 연산은 결과가 같은 방법으로 줄인다. T 의 이득이 증명되지 않았다고 보고하고, J 가 T 보다 2.0%p 이상 낫다면 사용자에게 묻는다.
  2단계  M, V 가 T 와 '같은가': 주 영역 MAE 감소율 평균이 T 평균 - 1.0 %p 이상이고(사전 등록 마진 2.0 %p 의 절반), 다른 영역 최대 증가가 T 보다 0.05 mm 넘게 크지 않으면 같다고 본다.
         - 같다고 본 방법 중 가장 싼 것(M, 그다음 V)을 쓴다. 단, M 은 모델 예측만 쓰므로 Genesis 로 검증하지 않는다는 점을 함께 적는다.
         - 둘 다 못 넘으면 T 를 쓴다(시간 비용이 크다는 점을 알리고, 연산은 결과가 같은 방법으로 줄인다 - 사용자 지시 2026-11-17 02:30).
  주의   시드 3개는 거친 표본이다. 평균 차이가 시드 표준편차보다 작으면 '같다'가 아니라 '구별 못 함'이다. 이 경우에도 규칙대로 싼 쪽을 고르되 확신하지 못한다고 적는다.
[결과를 보기 전에 정한 판정 기준 - 2026-11-16 22:40]
  주 지표  주 영역(T 팔 시드 1 이 고친 영역)의 MAE 감소율(%) = (시작 MAE - 확정 후 MAE) / 시작 MAE
  T 팔이 '효과 있음'으로 인정되려면 한 대상(A, 하트) 안에서 다음 세 가지를 모두 만족해야 한다.
    (1) 시드 평균 감소율이 J 팔보다 2.0 %p 이상 크다.
    (2) 같은 시드끼리 짝을 지었을 때 T 팔이 J 팔 이상인 시드가 3개 중 2개 이상이다.
    (3) 다른 영역 MAE 의 최대 증가량(mm)이 J 팔보다 0.05 mm 넘게 크지 않다.
  둘 다 못 넘으면 '검증 안 됨'으로 보고한다. 시드 3개는 거친 표본이라 평균 차이가 시드 사이 흩어짐보다 작으면 '판정 불가'라고 같이 적는다.
  (T 가 도움이 되는지 안 되는지만 본다. 힌트가 약해서 J 팔과 같은 후보를 고르는 것도 '효과 없음'이다.)

사용: python ab_cf_eval.py --target A
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "ab_cf_vs_j"

CFG = {
    "A": dict(history="results/case_A/chain_A_rot/d2_0/child0", regions="results/case_A/chain_A_rot/regions_adj", t_main="results/case_A/chain_A_rot/d3_0/E5_s1/cf", target_npy=None),
    "heart": dict(history="results/heart_history190_20261115_203518/history", regions="results/case_heart/chain_heart_flip/regions_adj", t_main="results/case_heart/chain_heart_flip/d0/E5_s1/cf",
                  target_npy="results/case_heart/heart_target_raw_flipud_512_m.npy"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=["A", "heart"])
    a = ap.parse_args()
    c = CFG[a.target]
    import cf_chain as ch
    import cf_common as cc
    root0 = ROOT / c["history"]
    sc = ch.Scorer(ROOT / c["regions"], root0, {}, "all", (str(ROOT / c["target_npy"]) if c["target_npy"] else None))
    s0 = sc(sc.h0)
    arms = {}
    for d in sorted((BASE / a.target).glob("*_s*")) if (BASE / a.target).exists() else []:
        if d.is_dir() and not d.name.startswith("_"):
            arms[d.name] = d / "cf"
    t_chain = ROOT / c["t_main"]                  # 체인이 한 4단계(워커 3). 팔 폴더에 T_s1 이 있으면(모든 팔을 워커 2 로 맞춘 재실행) 그것을 쓰고, 없으면 체인 결과를 T_s1 로 쓴다
    if "T_s1" not in arms:
        arms["T_s1"] = t_chain
    arms = dict(sorted(arms.items()))
    main_rec = [r for r in json.loads((t_chain / "rounds.json").read_text()) if r.get("committed")]
    region = main_rec[0]["region"] if main_rec else None
    print(f"[{a.target}] 시작 상태: 이력 {c['history']} | 주 영역(T 시드 1 이 고친 영역) = {region} | 시작 MAE {s0['regions_mae'].get(region, float('nan')):.3f} mm, 평균 {np.mean(list(s0['regions_mae'].values())):.3f} mm")
    rows = {}
    for name, cf in arms.items():
        rj, fm = cf / "rounds.json", cf / "final_m.npy"
        if not (rj.exists() and fm.exists()):
            print(f"  {name:<6} 아직 결과 없음")
            continue
        rec = [r for r in json.loads(rj.read_text()) if r.get("committed")]
        s1 = sc(np.load(fm).astype(np.float64))
        if region is None or not rec:
            print(f"  {name:<6} 확정 없음")
            continue
        red = {k: 100.0 * (s0["regions_mae"][k] - s1["regions_mae"][k]) / s0["regions_mae"][k] for k in s0["regions_mae"]}
        worse = max(s1["regions_mae"][k] - s0["regions_mae"][k] for k in s0["regions_mae"] if k != region)
        acts = rec[0]["action"] if isinstance(rec[0]["action"][0], (list, tuple)) else [rec[0]["action"]]
        rows[name] = dict(red=red[region], mae0=s0["regions_mae"][region], mae1=s1["regions_mae"][region], worse=worse, mean1=float(np.mean(list(s1["regions_mae"].values()))),
                          fam=rec[0]["family"], n=len(acts), region=rec[0]["region"], J=(rec[0]["cur_J"], rec[0]["J_after"]), far=s1["far_max_mm"])
        r = rows[name]
        print(f"  {name:<6} 확정 {r['fam']:<7} 푸시 {r['n']}개 영역 {r['region']:<6} | 주 영역 MAE {r['mae0']:.3f} -> {r['mae1']:.3f} ({r['red']:+.1f}%) | 다른 영역 최대 증가 {r['worse']:+.3f} mm | 평균 영역 MAE {r['mean1']:.3f} | 영역 밖 최대 {r['far']:.1f} mm")
    T = {k: v for k, v in rows.items() if k.startswith("T_")}
    J = {k: v for k, v in rows.items() if k.startswith("J_")}
    if T and J:
        mt, mj = np.mean([v["red"] for v in T.values()]), np.mean([v["red"] for v in J.values()])
        seeds = sorted({k.split("_s")[1] for k in rows if f"T_s{k.split('_s')[1]}" in T and f"J_s{k.split('_s')[1]}" in J})
        wins = sum(T[f"T_s{s}"]["red"] >= J[f"J_s{s}"]["red"] for s in seeds)
        sd = np.std([v["red"] for v in list(T.values()) + list(J.values())])
        wt, wj = max(v["worse"] for v in T.values()), max(v["worse"] for v in J.values())
        c1, c2, c3 = (mt - mj) >= 2.0, (len(seeds) >= 3 and wins >= 2), (wt <= wj + 0.05)
        print(f"\n주 영역 MAE 감소율 평균: T {mt:+.1f}% (n={len(T)}) | J {mj:+.1f}% (n={len(J)}) | 차이 {mt - mj:+.1f} %p | 시드 사이 표준편차 {sd:.1f} %p")
        print(f"  (1) 평균 차이 >= 2.0 %p: {'통과' if c1 else '못 넘음'} | (2) 같은 시드에서 T>=J 인 시드 {wins}/{len(seeds)} (3개 중 2개 이상 필요): {'통과' if c2 else '못 넘음'} | (3) 다른 영역 증가 T {wt:+.3f} vs J {wj:+.3f} mm: {'통과' if c3 else '못 넘음'}")
        verdict = "효과 있음(기준 충족)" if (c1 and c2 and c3) else "검증 안 됨"
        if abs(mt - mj) < sd and verdict != "효과 있음(기준 충족)":
            verdict += " / 평균 차이가 시드 흩어짐보다 작아 판정 불가에 가까움"
        print(f"  => {a.target}: {verdict}")
    Cg = {k: v for k, v in rows.items() if k.startswith("C_")}
    if Cg:      # C 팔: T 와 같은 반사실, 확정은 J 로만(2026-11-17 추가). T(확정에 T 개입)와 J(반사실 없음) 사이를 가른다
        parts = [f"C {np.mean([v['red'] for v in Cg.values()]):+.1f}% (n={len(Cg)})"]
        if J:
            parts.append(f"J {np.mean([v['red'] for v in J.values()]):+.1f}% (n={len(J)})")
        if T:
            parts.append(f"T {np.mean([v['red'] for v in T.values()]):+.1f}% (n={len(T)})")
        print("\n[참고] C 팔(T 와 같은 반사실 + 확정은 J 로만): 주 영역 MAE 감소율 평균 " + " | ".join(parts))
        print("       C 가 J 보다 낫다면 반사실 힌트가 후보를 좋게 한다는 뜻이고, C 가 T 보다 낫다면 T-추적 확정 규칙이 J 를 양보시킨다는 뜻이다. 시드 3개의 거친 표본이다.")
    Vg = {k: v for k, v in rows.items() if k.startswith("V_")}
    if Vg:      # V 팔: 모델 1위 후보 하나만 Genesis 로 검증(②T 트리 없음)
        print(f"\n[참고] V 팔(모델이 거른 1위만 Genesis 재생): 주 영역 MAE 감소율 평균 {np.mean([v['red'] for v in Vg.values()]):+.1f}% (n={len(Vg)}, 단계 시간 T 의 약 절반)")
    Mg = {k: v for k, v in rows.items() if k.startswith("M_")}
    if Mg:      # 2026-11-17 추가: 모델 반사실만 쓰는 M 팔(재생 없음). 사전 등록 판정(T vs J)과 별개의 참고 비교
        parts = [f"M {np.mean([v['red'] for v in Mg.values()]):+.1f}% (n={len(Mg)}, 단계 시간 J 와 비슷)"]
        if J:
            parts.append(f"J {np.mean([v['red'] for v in J.values()]):+.1f}% (n={len(J)})")
        if T:
            parts.append(f"T {np.mean([v['red'] for v in T.values()]):+.1f}% (n={len(T)}, 단계 시간 약 3배)")
        print("\n[참고] 탐색 방법별 주 영역 MAE 감소율 평균: " + " | ".join(parts))
        print("       M 이 T 와 비슷하거나 J 보다 낫다면 Genesis 재생 없이 단계 시간을 약 1/3 로 줄일 수 있다는 뜻이다. 시드 3개의 거친 표본이라 시드 흩어짐보다 작은 차이는 해석하지 않는다.")
    return rows


def decide(rows):
    """위 [방법 선택 규칙]을 기계적으로 적용한다(팔마다 시드 3개가 다 있어야 의미가 있다)."""
    g = lambda pre: {k: v for k, v in rows.items() if k.startswith(pre)}
    G = {"J": g("J_"), "M": g("M_"), "V": g("V_"), "T": g("T_")}
    n = {k: len(v) for k, v in G.items()}
    print("\n[방법 선택 규칙 적용] 시드 수: " + ", ".join(f"{k} {n[k]}" for k in G))
    if not G["T"] or not G["J"]:
        print("  T 와 J 가 모두 있어야 판정할 수 있다")
        return
    mean = lambda d: float(np.mean([v["red"] for v in d.values()]))
    worst = lambda d: max(v["worse"] for v in d.values())
    seeds = sorted({k.split("_s")[1] for k in G["T"] if f"J_s{k.split('_s')[1]}" in G["J"]})
    wins = sum(G["T"][f"T_s{s}"]["red"] >= G["J"][f"J_s{s}"]["red"] for s in seeds)
    t_beats_j = (mean(G["T"]) - mean(G["J"]) >= 2.0) and len(seeds) >= 3 and wins >= 2 and worst(G["T"]) <= worst(G["J"]) + 0.05
    sd = float(np.std([v["red"] for d in G.values() for v in d.values()]))
    print(f"  1단계: T {mean(G['T']):+.1f}% vs J {mean(G['J']):+.1f}% -> 사전 등록 기준 {'통과' if t_beats_j else '못 넘음'}")
    if not t_beats_j:
        print("  => 원래 방법(T)을 그대로 쓰고 연산은 결과가 같은 방법(MPS, 병목 제거 등)으로 줄인다(사용자 지시 2026-11-17). 단, 반사실(T)의 이득이 이 설정에서 J 보다 낫다고 증명되지 않았다는 점을 보고한다.")
        if mean(G["J"]) - mean(G["T"]) >= 2.0:
            print("  !! J 가 T 보다 평균 2.0%p 이상 낫다: 원래대로 갈지 사용자 결정이 필요하다")
        return
    accepted = []
    for name in ("M", "V"):
        if G[name]:
            ok = mean(G[name]) >= mean(G["T"]) - 1.0 and worst(G[name]) <= worst(G["T"]) + 0.05
            print(f"  2단계: {name} {mean(G[name]):+.1f}% vs T {mean(G['T']):+.1f}% (허용 하한 {mean(G['T']) - 1.0:+.1f}%) -> {'같다고 봄' if ok else '못 미침'}")
            if ok:
                accepted.append(name)
    pick = accepted[0] if accepted else "T"
    print(f"  => {pick} 를 쓴다" + ("" if accepted else " (M, V 모두 T 에 못 미침, 시간 비용이 큼)") + (f" | 시드 표준편차 {sd:.1f}%p: 평균 차이가 이보다 작으면 '같다'가 아니라 '구별 못 함'" if True else ""))


if __name__ == "__main__":
    decide(main())
