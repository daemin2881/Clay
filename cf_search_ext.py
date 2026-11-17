"""빔 탐색 이후 국소 보상 단계에 쓰는 탐색 방법 모음 (연구 메모의 제안 적용). 전부 순수 함수라 Genesis 없이 시험할 수 있다.
cf_compensate.py 가 --refine-mode / --pibo / --diverse / --pair-branch 로 연결한다.

  act_feat              5차원 푸시 [x, y, theta, L, d] -> 거리 계산용 6차원 특징(방향은 cos/sin 으로 이어 붙이고 길이·깊이는 정규화)
  fit_gp                국소 가우시안 과정(RBF, 길이 척도는 주변 우도로 선택) -> predict(특징) = (평균 J, 표준편차)   [BADS 의 국소 대리모델, 논문 3]
  expected_improvement  최소화 문제의 기대 개선(EI)
  sample_around         좋은 후보 주변에서 위치·방향·길이·깊이를 함께 흔든 제안(스케일이 줄어들며 수렴)
  select_batch          한 묶음(워커 수만큼)에서 서로 가까운 후보가 몰리지 않게 지역 벌점으로 고른다               [Batch BO, 논문 4]
  alignment_map         반사실 변화 C* 와 지금 필요한 변화(목표-현재)가 같은 방향인 곳 = min(|C*|, |e|) (mm)       [πBO 사전의 재료, 논문 2]
  prior_values / normalize_prior / pibo_gamma   푸시가 지나는 길의 평균 일치도 -> (0,1] 사전, 관측이 쌓이면 지수 gamma=beta/(1+n) 로 영향 감소
  spsa_directions / spsa_update   동시 섭동(두 번 평가로 기울기 한 번) 비교 방법                                      [논문 7]
"""
import math

import numpy as np
from scipy import ndimage
from scipy.stats import norm


def act_feat(a):
    x, y, th, L, d = [float(v) for v in a]
    return np.array([(x - 0.47) / 0.09, y / 0.11, math.cos(th), math.sin(th), (L - 0.01) / 0.03, (d - 0.01) / 0.01])


def fit_gp(X, y, ls_grid=(0.5, 0.8, 1.2, 1.8, 2.8), noise=1e-3):
    """RBF 가우시안 과정. 길이 척도는 로그 주변 우도가 가장 큰 값으로 고른다. 반환: predict(Xs) -> (평균, 표준편차), predict.ls"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ym, ys = float(y.mean()), float(y.std()) + 1e-9
    yn = (y - ym) / ys
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    best = None
    for ls in ls_grid:
        K = np.exp(-0.5 * d2 / ls ** 2) + noise * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        al = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        lml = -0.5 * float(yn @ al) - float(np.log(np.diag(L)).sum())
        if best is None or lml > best[0]:
            best = (lml, ls, L, al)
    if best is None:
        raise ValueError("가우시안 과정을 맞추지 못했다(공분산이 특이)")
    _, ls, L, al = best

    def predict(Xs):
        Xs = np.asarray(Xs, dtype=np.float64)
        Ks = np.exp(-0.5 * ((Xs[:, None, :] - X[None, :, :]) ** 2).sum(-1) / ls ** 2)
        mu = Ks @ al
        v = np.linalg.solve(L, Ks.T)
        var = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
        return mu * ys + ym, np.sqrt(var) * ys

    predict.ls = ls
    return predict


def expected_improvement(mu, sd, best, xi=0.0):
    """최소화: best 보다 낮아질 기대량. sd 가 0 에 가까우면 0 이상의 개선량만 남긴다."""
    mu, sd = np.asarray(mu, dtype=np.float64), np.maximum(np.asarray(sd, dtype=np.float64), 1e-12)
    imp = best - mu - xi
    z = imp / sd
    return np.maximum(imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)


def sample_around(rng, act, scale, n, clamp):
    """act 주변의 제안 n개. 위치 4 mm, 방향 0.35 rad, 길이 +-30%, 깊이 +-25% 를 scale 배로 흔든다. clamp 는 행동 범위로 자르는 함수."""
    x, y, th, L, d = [float(v) for v in act]
    out = []
    for _ in range(n):
        out.append(clamp([x + rng.normal(0, 4e-3 * scale), y + rng.normal(0, 4e-3 * scale), th + rng.normal(0, 0.35 * scale),
                          L * math.exp(rng.normal(0, 0.30 * scale)), d * math.exp(rng.normal(0, 0.25 * scale))]))
    return out


def select_batch(F, score, k, ell=0.6):
    """점수가 큰 순으로 k개를 고르되, 고른 점 가까이의 후보는 점수를 깎는다(지역 벌점). 반환: 고른 인덱스 목록.
    F: 특징 행렬(n x 6), score: 0 이상의 획득 점수, ell: 벌점이 미치는 거리(특징 공간)."""
    F = np.asarray(F, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64).copy()
    taken = np.zeros(len(s), dtype=bool)
    chosen = []
    for _ in range(min(int(k), len(s))):
        i = int(np.argmax(np.where(taken, -1.0, s)))
        if taken[i]:
            break
        chosen.append(i)
        taken[i] = True
        s = s * (1.0 - np.exp(-0.5 * ((F - F[i]) ** 2).sum(1) / ell ** 2))
    return chosen


def alignment_map(C, e, wsoft):
    """C: 반사실 변화(m), e: 목표 - 현재(m), wsoft: 영역 창 가중치. 두 변화가 같은 방향인 곳의 min(|C|, |e|) (mm), 그 밖은 0."""
    c = np.asarray(C, dtype=np.float64) * 1000.0
    ee = np.asarray(e, dtype=np.float64) * 1000.0
    return np.where(np.sign(c) * np.sign(ee) > 0, np.minimum(np.abs(c), np.abs(ee)), 0.0) * np.asarray(wsoft, dtype=np.float64)


def smooth_for_tool(A, tool_r_px):
    """도구 크기(지름)로 평균을 내 둔다: 푸시가 지나는 길 위에서 값 하나만 읽어도 도구 폭의 일치도가 된다."""
    return ndimage.uniform_filter(np.asarray(A, dtype=np.float64), size=max(3, int(round(2.0 * tool_r_px))), mode="constant")


def prior_values(A_s, rows, cols):
    """푸시 길(rows, cols = first_beam.path_points)을 따라 도구 크기로 평균 낸 일치도의 평균."""
    r = np.clip(np.round(np.asarray(rows)).astype(int), 0, A_s.shape[0] - 1)
    c = np.clip(np.round(np.asarray(cols)).astype(int), 0, A_s.shape[1] - 1)
    return float(A_s[r, c].mean())


def normalize_prior(vals, floor=0.05):
    """후보들의 일치도 -> (floor, 1] 사전. 가장 잘 맞는 후보가 1, 하나도 안 맞으면 모두 floor(= 사전이 순위를 바꾸지 않는다)."""
    v = np.asarray(vals, dtype=np.float64)
    mx = float(v.max()) if len(v) else 0.0
    if mx <= 1e-9:
        return np.ones(len(v))
    return floor + (1.0 - floor) * v / mx


def pibo_gamma(beta, n_obs):
    """πBO: 획득 함수 x 사전^(beta/n). 관측이 쌓일수록(n 이 클수록) 사전의 영향이 줄어든다."""
    return float(beta) / (1.0 + max(int(n_obs), 0))


# ------------------------------------------------------------------ SPSA (비교용)
SPSA_UNIT = (4e-3, 4e-3, 0.3, 3e-3, 2e-3)       # 한 단위 = 위치 4 mm, 방향 0.3 rad, 길이 3 mm, 깊이 2 mm


def spsa_directions(rng, n_dirs=2):
    return [rng.choice([-1.0, 1.0], size=5) for _ in range(n_dirs)]


def spsa_points(act, dirs, k, clamp, c0=1.0):
    """이번 회차에 평가할 점: 방향마다 act +- c_k * delta (단위는 SPSA_UNIT). 반환: [(plus, minus), ...]"""
    ck = c0 / (k + 1) ** 0.101
    base = np.array([float(v) for v in act])
    unit = np.array(SPSA_UNIT)
    return [(clamp(list(base + ck * d * unit)), clamp(list(base - ck * d * unit))) for d in dirs]


def spsa_update(act, dirs, jpairs, k, clamp, c0=1.0, step0=1.0):
    """jpairs: 방향마다 (J(+), J(-)). 기울기를 추정하고, 기울기의 최대 성분으로 나눈 방향으로 step0/(k+1)^0.602 단위만큼 내려간다."""
    ck = c0 / (k + 1) ** 0.101
    g = np.zeros(5)
    for d, (jp, jm) in zip(dirs, jpairs):
        g += (jp - jm) / (2.0 * ck) * d              # delta 의 성분이 +-1 이라 1/delta = delta
    g /= max(len(dirs), 1)
    mx = float(np.abs(g).max())
    if mx < 1e-12:
        return clamp([float(v) for v in act])
    step = step0 / (k + 1) ** 0.602
    base = np.array([float(v) for v in act])
    return clamp(list(base - step * (g / mx) * np.array(SPSA_UNIT)))


# ---------------------------------------------------------------------------------------------
# TuRBO 식 다중 국소 모델 (Eriksson 외, NeurIPS 2019 [8]) 의 이 시스템용 간소판. 순수 함수라 Genesis 없이 시험할 수 있다.
#   - 시작점(신뢰영역 중심)을 서로 떨어지게 여러 개 고른다 (pick_starts)
#   - 시작점마다 신뢰영역 길이를 둔다: 성공이 연속되면 늘리고, 실패가 연속되면 줄인다 (TrustRegions). 길이는 sample_around 의 scale 로 쓴다
#   - 시작점마다 그 주변 관측만으로 국소 가우시안 과정을 맞춘다 (local_subset + turbo_predict)
#   - 한 배치의 평가 예산은 전체 제안을 모아 기대 개선 순으로 고르므로 유망한 신뢰영역에 자동으로 더 간다 (cf_compensate 쪽 select_batch)
# 원래 TuRBO 와 다른 점: Thompson 샘플링이 아니라 기대 개선(EI), 신뢰영역은 박스가 아니라 sample_around 의 크기 배율이다.
# ---------------------------------------------------------------------------------------------
def pick_starts(F, J, k, ell=0.6, force_first=None, force_second=None):
    """J 가 낮은 순으로, 이미 고른 점에서 특징 거리 ell 이상 떨어진 것만 k 개까지 고른다. 모자라면 남은 것을 J 순으로 채운다.
    force_first/force_second: 인덱스를 주면 첫째/둘째 시작점으로 우선 쓴다('실측 최선' + '반사실 힌트가 가리킨 곳'). 반환: 인덱스 목록."""
    F = np.asarray(F, dtype=np.float64)
    order = [int(i) for i in np.argsort(np.asarray(J, dtype=np.float64))]
    chosen = []
    for forced in (force_first, force_second):
        if forced is not None and forced not in chosen and len(chosen) < k:
            if not chosen or np.sqrt(((F[forced] - F[chosen]) ** 2).sum(1)).min() >= ell:
                chosen.append(int(forced))
    for i in order:
        if len(chosen) >= k:
            break
        if i in chosen:
            continue
        if not chosen or np.sqrt(((F[i] - F[chosen]) ** 2).sum(1)).min() >= ell:
            chosen.append(i)
    for i in order:                       # 떨어진 점이 부족하면 가까운 점으로라도 채운다
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen


def local_subset(F, center, radius, min_n=6):
    """center 에서 특징 거리 radius 이내의 관측 인덱스. min_n 개보다 적으면 가장 가까운 min_n 개."""
    F = np.asarray(F, dtype=np.float64)
    d = np.sqrt(((F - np.asarray(center, dtype=np.float64)) ** 2).sum(1))
    idx = np.nonzero(d <= radius)[0]
    if len(idx) < min(min_n, len(F)):
        idx = np.argsort(d)[:min(min_n, len(F))]
    return idx


class TrustRegions:
    """시작점(키)마다 신뢰영역 길이. 성공 succ_tol 번 연속이면 x1.5(상한 hi), 실패 fail_tol 번 연속이면 x0.5(하한 lo)."""

    def __init__(self, init=1.0, lo=0.1, hi=2.0, succ_tol=2, fail_tol=2):
        self.init, self.lo, self.hi, self.succ_tol, self.fail_tol = init, lo, hi, succ_tol, fail_tol
        self.s = {}

    def _get(self, key):
        return self.s.setdefault(key, dict(len=self.init, succ=0, fail=0))

    def length(self, key):
        return self._get(key)["len"]

    def update(self, key, success):
        st = self._get(key)
        if success:
            st["succ"], st["fail"] = st["succ"] + 1, 0
            if st["succ"] >= self.succ_tol:
                st["len"], st["succ"] = min(st["len"] * 1.5, self.hi), 0
        else:
            st["fail"], st["succ"] = st["fail"] + 1, 0
            if st["fail"] >= self.fail_tol:
                st["len"], st["fail"] = max(st["len"] * 0.5, self.lo), 0
        return st["len"]


def turbo_predict(F_all, J_all, start_feats, prop_feats, owners, lengths, radius_mult=3.0, min_n=6):
    """시작점마다 그 주변 관측으로 국소 가우시안 과정을 맞춰 자기 제안의 (평균, 표준편차)를 구한다.
    owners[i] = 제안 i 의 시작점 번호(start_feats 의 행), lengths[s] = 그 시작점의 신뢰영역 길이. 반환: (mu, sd, 시작점별 국소 관측 수)."""
    F_all, J_all = np.asarray(F_all, dtype=np.float64), np.asarray(J_all, dtype=np.float64)
    prop_feats = np.asarray(prop_feats, dtype=np.float64)
    owners = np.asarray(owners, dtype=int)
    mu, sd, n_loc = np.zeros(len(prop_feats)), np.zeros(len(prop_feats)), []
    for s, sf in enumerate(start_feats):
        sel = np.nonzero(owners == s)[0]
        if len(sel) == 0:
            n_loc.append(0)
            continue
        idx = local_subset(F_all, sf, radius_mult * lengths[s], min_n)
        pred = fit_gp(F_all[idx], J_all[idx])
        mu[sel], sd[sel] = pred(prop_feats[sel])
        n_loc.append(int(len(idx)))
    return mu, sd, n_loc


# ---------------------------------------------------------------------------------------------
# 제약 베이지안 최적화 (Gardner 외 2014, Gelbart 외 2014 의 '제약 기대 개선'): 목적 J 를 낮추되 부수 변화 g(a) = 먼 곳 최대 변화 <= limit 를 지키는 푸시를 찾는다.
# 제약도 푸시 파라미터의 함수라서 가우시안 과정으로 배운다. 획득 점수 = EI x P(g <= limit). 한도를 넘은 곳에서 평가를 낭비하지 않고, 어느 위치·방향의 푸시가
# 주변을 크게 망가뜨리는지 관측이 쌓일수록 배운다(예: 평평한 이웃을 파서 점토를 끌어오는 푸시).
# ---------------------------------------------------------------------------------------------
def prob_feasible(mu_g, sd_g, limit):
    """g 의 가우시안 과정 예측 (mu_g, sd_g) 에서 P(g <= limit)."""
    sd = np.maximum(np.asarray(sd_g, dtype=np.float64), 1e-9)
    return norm.cdf((limit - np.asarray(mu_g, dtype=np.float64)) / sd)
