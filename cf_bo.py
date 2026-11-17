"""BADS 식 국소 GP 정제 제안 + πBO 사전 가중 + 국소 벌점 배치 (cf_compensate --refine-mode gp).

빔 단계와 앞선 정제에서 이미 Genesis 로 눌러 본 (행동 5차원, 영역 점수 J) 로 작은 GP 를 맞추고, 상위 행동 주변의 후보 중
기대 개선(EI)이 큰 것을 한 묶음(워커 수)만큼 고른다. 같은 묶음 안에서는 서로 가까운 후보의 우선순위를 낮춘다(국소 벌점).
사전(prior)이 주어지면 acq *= (prior + 0.05) ** (beta / 관측 수) 로 곱한다(πBO: 관측이 쌓일수록 사전의 영향이 줄어든다).
GP 는 순위를 '제안'할 뿐이고 확정은 항상 Genesis 로 잰 J 로 한다. 접촉이 바뀌면 예측이 틀릴 수 있어서 기본(끔)은 기존 패턴 탐색이다.

입력 정규화: 위치(x, y)는 범위로 나누고, 방향 th 는 (cos, sin) 로 바꿔 한 바퀴를 이어 붙이며, 길이와 깊이는 단위를 맞춘다.
"""
import math

import numpy as np
from scipy.stats import norm


def feat(a):
    x, y, th, L, d = [float(v) for v in a]
    return np.array([(x - 0.47) / 0.09, y / 0.11, math.cos(th), math.sin(th), (L - 0.01) / 0.03, (d - 0.01) / 0.01])


def gp_predict(X, y, Xs, ls=1.5, noise=1e-4):
    """등방 RBF GP. 평균·표준편차를 y 의 원래 단위로 돌려준다."""
    ym, ysd = float(y.mean()), float(y.std()) + 1e-9
    yn = (y - ym) / ysd

    def k(A, B):
        return np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / ls ** 2)

    K = k(X, X) + noise * np.eye(len(X))
    Lc = np.linalg.cholesky(K)
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yn))
    Ks = k(Xs, X)
    mu = Ks @ alpha
    v = np.linalg.solve(Lc, Ks.T)
    var = np.clip(1.0 - (v ** 2).sum(0), 1e-9, None)
    return mu * ysd + ym, np.sqrt(var) * ysd


def local_candidates(tops, rng, n=80):
    """상위 행동 주변 후보: 도구 진행 방향·옆 방향 이동, 방향, 길이·깊이 비율을 가우시안으로 흔든다."""
    out = []
    per = max(n // max(len(tops), 1), 1)
    for a in tops:
        x, y, th, L, d = [float(v) for v in a]
        px, py, tx, ty = -math.sin(th), math.cos(th), math.cos(th), math.sin(th)
        for _ in range(per):
            lat, lon = rng.normal(0.0, 3e-3), rng.normal(0.0, 4e-3)
            out.append([x + lat * px + lon * tx, y + lat * py + lon * ty, th + rng.normal(0.0, 0.3),
                        L * math.exp(rng.normal(0.0, 0.3)), d * math.exp(rng.normal(0.0, 0.25))])
    return out


def propose(points, tops, n_batch, rng, *, clamp, key, seen, prior=None, beta=0.0, ls=1.5, div=0.8, n_cand=80):
    """points: [(action, J)] 이미 시험한 단일 푸시. tops: 주변을 탐색할 상위 행동들. 반환: 시험할 행동 n_batch 개(서로 떨어진 것)."""
    if len(points) < 6:
        return []
    X = np.array([feat(a) for a, _ in points])
    y = np.array([float(j) for _, j in points])
    cands, ks = [], set()
    for a in local_candidates(tops, rng, n_cand):
        a = clamp(a)
        k = key(a)
        if k in seen or k in ks:
            continue
        ks.add(k)
        cands.append(a)
    if not cands:
        return []
    Xc = np.array([feat(a) for a in cands])
    mu, sd = gp_predict(X, y, Xc, ls)
    ymin = float(y.min())
    z = (ymin - mu) / np.maximum(sd, 1e-9)
    acq = (ymin - mu) * norm.cdf(z) + sd * norm.pdf(z) + 1e-12            # 최소화용 기대 개선
    if prior is not None and beta > 0.0:
        pi = np.array([prior(a) for a in cands], dtype=np.float64)
        acq = acq * (pi + 0.05) ** (beta / max(len(points), 1))
    chosen = []
    for _ in range(min(n_batch, len(cands))):
        i = int(np.argmax(acq))
        if acq[i] < 0:
            break
        chosen.append(cands[i])
        d2 = ((Xc - Xc[i]) ** 2).sum(1)
        acq = acq * (1.0 - np.exp(-0.5 * d2 / div ** 2))                   # 국소 벌점: 방금 고른 후보 근처의 우선순위를 낮춘다
        acq[i] = -1.0
    return chosen
