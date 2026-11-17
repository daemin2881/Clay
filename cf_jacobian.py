"""자코비안 + Adam + Broyden 후보 제안 (cf_compensate.py 의 JAC 계열). 대상(A, 하트, ...)과 영역 구성에 의존하지 않는다.

문제: 행동 a=[x,y,theta,L,d] 를 바꾸면 J(영역 형태/기울기/테두리 + 주변 변화)가 어떻게 변하나?
  1) 사전(prior) 기울기  g0 = dJ_model/da   Param2Deform 이 예측한 다음 높이맵으로 잰 J 를 유한 차분(중앙 차분)해서 구한다. 모델이 싸므로 가능하지만
                         모델은 실제와 어긋나(순위 상관 0.7 정도) 방향만 믿을 수 있고 크기는 믿을 수 없다.
  2) Broyden 보정        Genesis 에서 '실제로 실행해 본' 이웃 후보들의 J(물리) 로 기울기를 갱신한다.
                         각 이웃 a_k 에 대해 secant 조건 g.(a_k-a0) = J_phys(a_k)-J_phys(a0) 를 만족하도록 rank-1 갱신(먼 것 -> 가까운 것 순서).
                         => 모델의 기울기를 물리 측정으로 고친 g.  (사전 + 물리 측정 = 설계안의 'C*/모델 사전 + 실행 결과 갱신')
  3) Adam 스텝           g 로 Adam(1차/2차 모멘트)과 정규화 경사하강 방향을 만들고 여러 크기(alphas)로 이동한 후보를 만든다.
  4) 순위               J_hat(a) = J_phys(a0) + [J_model(a) - J_model(a0)] + (g - g0).(a - a0)
                         = 모델의 비선형 예측 + Broyden 이 알아낸 '모델의 국소 선형 오차' 보정. 상위 top 개만 물리 평가한다.
모든 미분·스텝은 SCALE 단위(x,y 1mm, theta 0.05rad, L 1mm, d 1mm)에서 한다. 확정은 언제나 Genesis 실행 후 롤백 검증으로만 한다.
"""
import math

import numpy as np

SCALE = np.array([1e-3, 1e-3, 0.05, 1e-3, 1e-3])
NAMES = ("x", "y", "theta", "L", "d")


def _wrap(t):
    return (t + math.pi) % (2 * math.pi) - math.pi


def delta_units(a, a0):
    d = (np.asarray(a, float) - np.asarray(a0, float))
    d[2] = _wrap(d[2])
    return d / SCALE


class JacState:
    """기준 행동 하나에 대한 상태: g0(모델 사전 기울기), g(Broyden 보정 후), Adam 모멘트. 한 라운드 안에서만 유지한다(확정으로 상태가 바뀌면 다시 만든다)."""

    def __init__(self, g0):
        self.g0 = np.asarray(g0, float).copy()
        self.g = self.g0.copy()
        self.m = np.zeros(5)
        self.v = np.zeros(5)
        self.t = 0


def model_J(model, h_cur, acts, score_fn, prior=None, bs=12):
    """모델이 예측한 다음 높이맵의 J. prior=(T, win, dT0, w_cf) 이면 반사실 목표 T 까지의 거리 항(C* 사전)을 더한다."""
    import torch
    from vs_planner import predict_next_batch
    dev = next(model.parameters()).device
    cur = torch.as_tensor(h_cur, dtype=torch.float32, device=dev)[None, None]
    out = []
    for i in range(0, len(acts), bs):
        a = torch.tensor(np.asarray(acts[i:i + bs], dtype=np.float32), dtype=torch.float32, device=dev)
        with torch.no_grad():
            maps = predict_next_batch(model, cur, a)[:, 0].double().cpu().numpy()
        for m in maps:
            j = float(score_fn(m))
            if prior is not None and prior[3] > 0:
                T, win, dT0, w = prior
                j += w * float(np.abs(m - T)[win].mean() * 1000.0) / max(dT0, 1e-6)
            out.append(j)
    return np.array(out)


def fd_gradient(model, h_cur, a0, score_fn, clamp, prior=None):
    """모델 J 의 중앙 차분 기울기 (SCALE 단위당 J). 경계에 걸려 한쪽이 잘리면 실제로 이동한 만큼으로 나눈다."""
    acts = []
    for i in range(5):
        for s in (+1.0, -1.0):
            e = np.zeros(5)
            e[i] = s * SCALE[i]
            acts.append(clamp(list(np.asarray(a0, float) + e)))
    J = model_J(model, h_cur, acts, score_fn, prior)
    g = np.zeros(5)
    for i in range(5):
        da = delta_units(acts[2 * i], acts[2 * i + 1])[i]
        g[i] = (J[2 * i] - J[2 * i + 1]) / da if abs(da) > 1e-9 else 0.0
    return g


def broyden_correct(g0, a0, J0, samples, radius=10.0, damp=0.7):
    """물리로 측정한 이웃 samples=[(행동, J_phys)] 로 기울기를 보정한다. 반환: (g, 사용한 이웃 수, 이웃별 잔차)."""
    rows = []
    for a, Jp in samples:
        da = delta_units(a, a0)
        n = float(np.linalg.norm(da))
        if 1e-6 < n <= radius and np.isfinite(Jp) and Jp < 1e8:
            rows.append((n, da, Jp - J0))
    rows.sort(key=lambda r: -r[0])          # 먼 것부터, 가장 가까운 것을 마지막에(가까울수록 선형 가정이 맞다)
    g = np.asarray(g0, float).copy()
    for n, da, dJ in rows:
        g = g + damp * ((dJ - g @ da) / (da @ da)) * da
    if not np.all(np.isfinite(g)):
        return np.asarray(g0, float).copy(), 0
    return g, len(rows)


def adam_dirs(st, g, b1=0.9, b2=0.999):
    """Adam 방향(성분별 정규화)과 정규화 경사하강 방향. 기울기가 거의 0 인 성분은 잡음이므로 억제한다."""
    st.t += 1
    st.m = b1 * st.m + (1 - b1) * g
    st.v = b2 * st.v + (1 - b2) * g * g
    mh, vh = st.m / (1 - b1 ** st.t), st.v / (1 - b2 ** st.t)
    eps = 0.05 * float(np.max(np.abs(g))) + 1e-12
    d_adam = -mh / (np.sqrt(vh) + eps)
    d_gd = -g / (np.linalg.norm(g) + 1e-12) * math.sqrt(5.0)
    return [("adam", d_adam), ("gd", d_gd)]


def propose(model, h_cur, a0, J0, phys, score_fn, st, clamp, ok_end, seen=(), prior=None, top=4, alphas=(1.0, 2.0, 4.0, 8.0), radius=10.0, key=None):
    """a0 주변에서 자코비안 방향 후보 top 개를 제안한다.
    phys: 지금까지 물리로 평가한 [(행동, J_phys)] (Broyden 보정에 쓴다), st: JacState(없으면 만든다), seen: 이미 평가한 행동 키 집합.
    반환: (후보 [(행동, J_hat)], st, 진단 dict)"""
    if st is None:
        st = JacState(fd_gradient(model, h_cur, a0, score_fn, clamp, prior))
    st.g, nb = broyden_correct(st.g0, a0, J0, phys, radius)
    dirs = adam_dirs(st, st.g)
    cands, seenk = [], set(seen)
    for nm, d in dirs:
        for al in alphas:
            a = clamp(list(np.asarray(a0, float) + al * SCALE * d))
            if not ok_end(a):
                continue
            k = key(a) if key else tuple(np.round(a, 4))
            if k in seenk:
                continue
            seenk.add(k)
            cands.append((a, f"{nm}x{al:g}"))
    diag = dict(g0=st.g0.copy(), g=st.g.copy(), n_nb=nb, dg=float(np.linalg.norm(st.g - st.g0)), n_cand=len(cands))
    if not cands:
        return [], st, diag
    Jm = model_J(model, h_cur, [c[0] for c in cands] + [list(a0)], score_fn, prior)
    Jm0 = Jm[-1]
    corr = st.g - st.g0
    out = []
    for (a, tag), jm in zip(cands, Jm[:-1]):
        da = delta_units(a, a0)
        out.append((a, float(J0 + (jm - Jm0) + corr @ da), tag))
    out.sort(key=lambda t: t[1])
    return out[:top], st, diag
