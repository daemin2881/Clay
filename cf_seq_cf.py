"""반사실 행동 시퀀스 탐색: 과거 행동을 '최대 k 개'만 바꿔서 영역 결과를 가장 좋게 만드는 개입 묶음을 고른다.
(Tsirtsis, De, Gomez-Rodriguez, NeurIPS 2023 "Finding Counterfactually Optimal Action Sequences in Continuous State Spaces" 의 구조: 바꾸는 행동 수 제한 + 최선 우선(A* 형) 탐색.
 그 논문의 최적성 보장은 이산 행동과 립시츠 가정에 기대므로 5차원 연속 푸시에서는 성립하지 않는다. 여기서는 구조만 빌리고, 평가는 항상 Genesis 재생 결과로 한다.)

이미 있는 것과의 차이
  cf_tree.py            '시간 구간 묶음'을 통째로 재생해서 효과 없는 묶음을 버린다(어느 행동이 원인인지 좁힘).
  이 모듈               개입(행동 번호 + 변형) 후보 중 최대 k 개를 골라 '함께' 재생했을 때의 효과를 찾는다(개입끼리 합쳐질 때 효과가 더 커지는 조합).

방법 (평가 함수 evaluate(개입 묶음) -> 이득, 클수록 좋음. 실제 사용에서는 Genesis 재생으로 영역 점수 개선량을 돌려준다)
  1. 단일 개입 이득 g_c 는 이미 재생해 둔 값(또는 evaluate 로 한 번씩 재서) 쓴다.
  2. 묶음의 예측 이득 = 단일 이득의 합 - 겹침 벌점(같은 곳을 바꾸는 개입끼리는 효과가 겹쳐 합이 과대평가된다: overlap(c1, c2) 로 0~1).
  3. 최선 우선 탐색: 예측 이득 + 낙관 보너스(아직 안 잰 조합일수록, 그리고 지금까지 잔차(실측-예측)가 클수록 크게)가 가장 큰 묶음부터 실제 재생으로 잰다.
     잴 때마다 잔차를 모아 낙관 보너스와 예측 보정을 갱신한다(예측이 과대/과소평가인 경향을 배운다).
  4. 예산(재생 횟수 또는 시간) 안에서 실측 이득이 가장 큰 묶음을 돌려준다. 단일 개입도 후보에 포함되므로 결과가 가장 좋은 단일보다 나빠지지 않는다.
Genesis/torch 불필요(순수 함수). 연결은 cf_compensate 의 ②T 단계에 옵션으로 붙인다(아직 연결 전).
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Tuple

Cand = Hashable          # 보통 (행동 번호, 변형 이름)
Bundle = FrozenSet[Cand]


def action_index(c) -> int:
    return int(c[0]) if isinstance(c, (tuple, list)) else int(c)


def valid(bundle: Iterable[Cand]) -> bool:
    """같은 행동 번호를 두 번 바꾸는 묶음은 만들지 않는다."""
    idx = [action_index(c) for c in bundle]
    return len(idx) == len(set(idx))


def predicted_gain(bundle: Bundle, g1: Dict[Cand, float], overlap: Callable[[Cand, Cand], float], bias: float = 0.0, scale: float = 1.0) -> float:
    """단일 이득의 합 - 겹침 벌점(쌍마다 작은 쪽 이득 x 겹침). bias, scale 은 지금까지 잰 조합에서 배운 보정(실측 ~ scale*예측 + bias)."""
    cs = list(bundle)
    tot = sum(g1[c] for c in cs)
    for a, b in itertools.combinations(cs, 2):
        tot -= min(max(g1[a], 0.0), max(g1[b], 0.0)) * float(overlap(a, b))
    return scale * tot + bias * (len(cs) - 1)


def _fit_correction(rows: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """실측 ~ scale*예측 (원점 통과 최소제곱, 0.3~1.5 로 제한). 조합 수가 적으면 보정 없음."""
    if len(rows) < 3:
        return 1.0, 0.0
    num = sum(p * t for p, t in rows)
    den = sum(p * p for p, t in rows) + 1e-12
    return min(max(num / den, 0.3), 1.5), 0.0


def search(cands: Sequence[Cand], g1: Dict[Cand, float], evaluate: Callable[[Bundle], float], overlap: Callable[[Cand, Cand], float] = lambda a, b: 0.0,
           k: int = 2, budget: int = 12, beta: float = 0.5, cost: Optional[Callable[[Bundle], float]] = None, seconds: Optional[float] = None,
           known: Optional[Dict[Bundle, float]] = None, pool: int = 8) -> dict:
    """k 개 이하 개입 묶음 탐색.
    cands: 개입 후보, g1: 단일 개입의 실측 이득, evaluate: 묶음 -> 실측 이득(재생), overlap: 두 개입의 겹침(0~1),
    budget: 허용 재생 횟수(이미 아는 값은 제외), cost/seconds: 묶음 재생 비용 함수와 시간 예산(둘 다 주면 시간이 우선),
    beta: 낙관 보너스 계수, pool: 단일 이득 상위 몇 개 후보 안에서만 조합을 만든다(조합 폭발 방지).
    반환: dict(best=(묶음, 실측 이득), measured=[(묶음, 이득, 예측)...], n_replays, spent)."""
    cs = sorted(cands, key=lambda c: -g1[c])[:max(int(pool), 1)]
    measured: Dict[Bundle, float] = {frozenset([c]): float(g1[c]) for c in cs}
    if known:
        measured.update({b: float(v) for b, v in known.items()})
    log: List[tuple] = []
    rows: List[Tuple[float, float]] = []
    resid_max, spent, n_rep = 0.0, 0.0, 0
    # 조합 후보(2..k 개)
    pool_bundles: List[Bundle] = []
    for r in range(2, int(k) + 1):
        for comb in itertools.combinations(cs, r):
            if valid(comb):
                pool_bundles.append(frozenset(comb))
    while True:
        if seconds is None and n_rep >= int(budget):
            break
        scale, bias = _fit_correction(rows)
        best_b, best_u = None, None
        for b in pool_bundles:
            if b in measured:
                continue
            pg = predicted_gain(b, g1, overlap, bias, scale)
            nov = 1.0 - 0.5 * (len(b) - 2) / max(k - 1, 1) if len(b) > 2 else 1.0
            ub = pg + beta * (resid_max + 0.1 * abs(pg)) * nov           # 낙관 보너스: 잔차가 컸던 만큼, 예측값의 10% 만큼은 늘 위로 본다
            if best_u is None or ub > best_u:
                best_b, best_u = b, ub
        if best_b is None:
            break
        c = float(cost(best_b)) if cost else 1.0
        if seconds is not None and spent + c > seconds:
            break
        pred = predicted_gain(best_b, g1, overlap, bias, scale)
        true = float(evaluate(best_b))
        measured[best_b] = true
        rows.append((pred if pred != 0 else 1e-9, true))
        resid_max = max(resid_max, abs(true - pred))
        log.append((best_b, true, pred))
        spent += c
        n_rep += 1
    best = max(measured.items(), key=lambda kv: kv[1])
    return dict(best=best, measured=log, n_replays=n_rep, spent=spent, single_best=max(((frozenset([c]), g1[c]) for c in cs), key=lambda kv: kv[1]))
