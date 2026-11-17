"""과거 기록 전체를 이용한 다단계(계층) 반사실 국소 보상. 영역 집합(RegionSet)의 모든 영역을 한꺼번에 다룬다.

대상(A, 고양이, 하트, ...)이나 시점(조형 K번째 상태)을 코드가 가정하지 않는다. 필요한 것은 조형 이력 폴더와 영역 집합뿐이다.
  - 이력: visual_sculpting.py --history-dir 가 만든 폴더. --upto K 로 앞쪽 K개 행동만 '과거'로 보고 K번째 상태를 '현재'로 삼는다.
  - 영역: build_regions.py 가 만든 폴더(자동 / 마스크 지정), live_edit.py 출력(VLM+SAM2), 또는 예전 형식. 이름과 개수는 자유.
  - 크기에 관한 값은 이력 manifest 의 도구 반경을 기준으로 잡고, 행동 범위는 플래너의 안전 범위(vs_planner.SAFE_PLAN_*)를 그대로 쓴다.

흐름 (영역마다)
  ① 관련 과거 행동 탐색 : 영역에 닿았거나 영역 높이를 바꾼 과거 행동 전부에서, 기록된 높이 변화 dH_t 로 현재 오차를 설명하는 제한 최소제곱 가중치 w_t 를 구하고
                          (w_t>0 "더 눌렀어야", w_t<0 "덜 눌렀어야") 영향 순으로 순위를 매긴다(beam).
  ② 과거 반사실 재생    : 재생을 믿을 수 있는 범위(현재 직전 --replay-len 개 행동) 안의 상위 행동을 Genesis 에서 '그 행동을 바꿔 현재까지 다시 실행'한다.
                          C* = H_cf - H_ctrl = "그때 달리 했다면 지금 점토가 이만큼 달랐을 것". control 재생이 기록과 맞아야(신뢰 기준) 쓰고,
                          T = 현재 + C* 로 J 가 좋아질 때(q>0)만 반사실 목표로 삼는다.
  ③ 현재 보상           : T 를 만들기 위해 그 과거 행동을 되돌리는 후보(CFR)와 아래 후보들을 현재 점토에서 실제로 실행해 보고 되돌려서(검증),
                          J 가 좋아지는 것 중 T 에 가장 가까운 것만 확정한다. 확정 후에는 바뀐 현재에서 다시 탐색한다(라운드 반복).
재생 범위 밖의 과거 행동은 ②를 건너뛰고 ③의 후보로만 쓰인다. 재생이 신뢰 기준을 못 넘거나 q<=0 이면 그 영역은 반사실 없이(abstain) 현재 상태 후보 평가만 한다.

후보 출처(무작위 없음)
  G1 과거 행동 전부   : 영역에 닿았거나 영역 높이를 바꾼 (현재까지의) 모든 행동(위 ①의 순위). 기록된 높이 변화 dH_t(전/후 높이맵)를 사전으로 삼아
                        현재 오차 r = 목표-현재 를 sum_t w_t dH_t 로 설명하는 제한 최소제곱(-1<=w<=1.5)을 푼다.
                        w_t>0 은 "그때 더 눌렀어야", w_t<0 은 "덜 눌렀어야"라는 반사실 정보다. |w_t|*영향 순으로 변형(얕게, 옆으로 +-4mm, 반대 방향, 다시 누르기)을 많이 만든다.
  G2 점토 이동 계획   : 영역 안 부족(구덩이) 덩어리마다 주변의 남는 점토(+) 방향을 찾아 그쪽에서 구덩이 쪽으로 미는 후보(시작 거리, 길이, 깊이 격자).
  G3 남는 점토 누르기 : 영역 안 튀어나온 덩어리 위를 누르는 후보.
  이월 빔            : 이전 라운드의 상위 후보를 새 상태에서 다시 평가.
계층 탐색
  1단계 영역 선택(오차가 큰 영역부터) -> 2단계 후보군 전체 평가 -> 3단계 상위 K 후보의 5차원 행동 주변 패턴 탐색(위치, 각도, 길이, 깊이) -> 4단계 확정 후 라운드 반복.
평가 J = 영역 평균[ w_h*높이/높이0 + w_g*기울기오차/기울기0 + w_e*테두리오차/테두리0 ] + mu*(영역 주변 고리 변화/높이0 평균). 0 은 기준 상태의 값이라 시작 J = 가중치 합.
        높이 = 0.5(MAE+RMSE) mm, 기울기·테두리 = 외형 오차(mm / 도구 반경). 현재보다 eps 이상 좋아질 때만 확정하므로 J 는 절대 나빠지지 않는다(단, 눈에 띄게 좋아진다는 보장은 아니다).

사용: DISPLAY=:1 python cf_compensate.py --history-dir H [--upto K] --regions DIR --out DIR --workers 4 [--viewer]
      (한 번에 영역 생성부터 전/후 그림까지: cf_edit.py)
"""
import argparse
import json
import math
import os
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import lsq_linear

import cf_search_ext as E       # 빔 이후 탐색 방법(적응형 정제, πBO 사전, 묶음 지역 벌점, SPSA). 옵션을 켤 때만 쓰인다

SIZE = 512


class FewObservations(ValueError):
    """한도 안에서 측정된 후보가 모자라 대리모델을 맞출 수 없는 정상적인 경우. --strict 에서도 실패가 아니라 패턴 탐색으로 대체하고 그 사실을 기록한다."""
# 행동 범위 (x_min, x_max, y_min, y_max, L_min, L_max, d_min, d_max). main() 이 cf_common.action_bounds() 로 채운다.
BOUNDS = None


def wrap(t):
    return math.atan2(math.sin(t), math.cos(t))


def clampA(a):
    x, y, th, L, d = a
    x0, x1, y0, y1, l0, l1, d0, d1 = BOUNDS
    return [float(min(max(x, x0), x1)), float(min(max(y, y0), y1)), wrap(float(th)), float(min(max(L, l0), l1)), float(min(max(d, d0), d1))]


def akey(a):
    return (round(a[0] / 2e-4), round(a[1] / 2e-4), round(a[2] / 0.01), round(a[3] / 2e-4), round(a[4] / 2e-4))


# ---------------------------------------------------------------- 평가 함수 (worker 와 main 이 같이 쓴다)
def build_cfg(rs, h_ref, mu, ring_px, far_px, tool_r_px, weights=(1.0, 1.0, 1.0), blur_tools=0.4, fp=None, target=None, w_focus=0.0, w_tail=0.0, pit_blur_tools=0.12):
    """rs: RegionSet. regs = [(이름, 마스크, 목표)]. weights = (형태(높이), 기울기(외형), 테두리(선명도)) 가중치.
    모든 항은 도구 반경의 blur_tools 배로 흐린 지도에서 재고(점토 입자 질감 제거), 기준 상태(h_ref)의 값으로 나눈 비율이다.
    fp(cf_common.relief_focus 결과)와 target 을 주면 '목표의 모양(부조/음각) + 근처'만 본다:
      - 주변 고리(ring)와 영역 밖 변화(far)는 초점 안에서만 잰다 = 배경은 점수·제약에서 빠지고, 글자의 다른 부분이 망가지는 것만 막는다.
      - w_focus > 0 이면 초점 전체의 형태·기울기·테두리 오차도 J 에 넣는다(영역 항과 w_focus : 1 로 섞음, 시작 J 는 그대로 가중치 합).
    w_tail > 0 이면 '목표를 지나친 양'의 상위 5% 평균(꼬리, 도구 반경 x pit_blur_tools 블러에서 잼, 기준 상태의 형태 오차로 나눔)도 영역 항에 더한다
      = 한 곳만 깊이 파인 구덩이/솟음에 벌점. 0 이면 이전과 같은 J."""
    import cf_common as cc
    regs = [(n, rs.masks[n], rs.targets[n]) for n in rs.names]
    union = np.zeros((SIZE, SIZE), bool)
    for _, m, _ in regs:
        union |= m
    sigma = cc.band_sigma(tool_r_px, blur_tools)
    ring = ndimage.binary_dilation(union, iterations=ring_px) & ~union
    far = ~ndimage.binary_dilation(union, iterations=far_px)
    if fp is not None:
        ring &= fp["focus"]
        far = fp["focus"] & far
    ring_r = {}                                      # 영역별 주변 고리(다른 영역은 제외): --ring-budget-mm 에 영역별 한도를 줄 때 쓴다
    for n_, m_, _ in regs:
        rr_ = ndimage.binary_dilation(m_, iterations=ring_px) & ~union
        ring_r[n_] = (rr_ & fp["focus"]) if fp is not None else rr_
    cfg = dict(regs=regs, union=union, ring=ring, ring_r=ring_r, far=far, h_ref=h_ref.astype(np.float64), mu=mu,
               vis_pre=cc.appearance_pre(rs.masks, rs.targets, sigma, h_ref=(h_ref if w_tail > 0 else None), pit_sigma_px=cc.band_sigma(tool_r_px, pit_blur_tools)), tool_r_px=float(tool_r_px), sigma=sigma, w=tuple(float(v) for v in weights),
               fpre=None, wf=0.0, w_tail=float(w_tail))
    per0, vis0, _ = _terms(cfg["h_ref"], cfg)
    cfg["norm"] = {n: (max(vis0[n][2], 0.1), max(vis0[n][0], 0.1), max(vis0[n][1], 0.1)) for n in per0}      # (형태, 기울기, 테두리)
    cfg["coll_norm"] = float(np.mean([v[0] for v in cfg["norm"].values()]))
    if fp is not None and target is not None:
        cfg["fpre"] = cc.focus_pre(fp, target, sigma)
        cfg["wf"] = float(w_focus)
        f0 = cc.focus_scores(cfg["h_ref"], cfg["fpre"], tool_r_px, sigma, "local")
        cfg["fnorm"] = (max(f0["shape"], 0.1), max(f0["grad"], 0.1), max(f0["edge"], 0.1))
    return cfg


def _terms(h, cfg):
    """per: 원본(안 흐린) 높이 MAE/RMSE mm (보고용), vis: (기울기, 테두리, 형태) 흐린 지도 기준, coll: 영역 주변 고리 변화 mm"""
    import cf_common as cc
    per = {}
    for n, m, t in cfg["regs"]:
        d = (h - t)[m] * 1000
        per[n] = (float(np.abs(d).mean()), float(np.sqrt((d ** 2).mean())))
    return per, cc.appearance(h, cfg["vis_pre"], cfg["tool_r_px"], cfg["sigma"]), float(np.abs(h - cfg["h_ref"])[cfg["ring"]].mean() * 1000)


def region_J(n, vis, cfg):
    wh, wg, we = cfg["w"]
    sn, gn, en = cfg["norm"][n]
    j = wh * vis[n][2] / sn + wg * vis[n][0] / gn + we * vis[n][1] / en
    if cfg.get("w_tail", 0.0) > 0.0:
        j += cfg["w_tail"] * vis[n][3] / sn          # 꼬리(과행 mm)는 형태0(mm) 단위로: 기준 상태의 형태 오차만큼 지나치면 가중치만큼 벌점
    return j


def focus_J(fs, cfg):
    wh, wg, we = cfg["w"]
    sn, gn, en = cfg["fnorm"]
    return wh * fs["shape"] / sn + wg * fs["grad"] / gn + we * fs["edge"] / en


def score(h, cfg, h_cur=None):
    """J = [영역 평균(w_h*형태/형태0 + w_g*기울기/기울기0 + w_e*테두리/테두리0) + w_focus*초점 전체의 같은 항] / (1 + w_focus) + mu*(영역 주변 고리 변화/형태0 평균).
    형태·기울기·테두리는 도구 반경 x blur_tools 로 흐린 지도에서 잰다(입자 질감 제외). 0 은 기준 상태(h_ref)의 값이라 시작 J = 가중치 합, 낮을수록 목표에 가깝다.
    초점(목표의 모양 + 근처) 항은 cfg['fpre'] 가 있을 때만. 원본 높이 MAE/RMSE 는 per 로 함께 돌려주지만 J 에는 넣지 않는다."""
    import cf_common as cc
    per, vis, coll = _terms(h, cfg)
    e = float(np.mean([region_J(n, vis, cfg) for n in per]))
    dh_ = np.abs(h - cfg["h_ref"])
    out = dict(e=e, per=per, vis=vis, coll=coll,
               coll_r={n_: (float(dh_[rr_].mean() * 1000) if rr_.any() else 0.0) for n_, rr_ in cfg["ring_r"].items()})
    if cfg.get("fpre") is not None:
        fs = cc.focus_scores(h, cfg["fpre"], cfg["tool_r_px"], cfg["sigma"], "local")
        jf = focus_J(fs, cfg)
        out.update(focus=fs, Jf=jf)
        e_all = (e + cfg["wf"] * jf) / (1.0 + cfg["wf"])
    else:
        e_all = e
    far = float(np.abs(h - h_cur)[cfg["far"]].max() * 1000) if (h_cur is not None and cfg["far"].any()) else 0.0
    out.update(J=e_all + cfg["mu"] * coll / cfg["coll_norm"], far_max=far)
    if cfg.get("protect") and h_cur is not None:      # --protect-skipped: 보호 영역 안에서의 변화(mm)
        pt_ = cfg.get("protect_t")
        if pt_:      # 기본: '목표에서 멀어진 양'만 잰다. 목표 모양에 가까워지는 변화는 0 이라 한도에 걸리지 않는다(화소별 max(0, |새-목표| - |현재-목표|) 의 영역 안 최대)
            pm = {n_: float(np.maximum(np.abs(h - pt_[n_]) - np.abs(h_cur - pt_[n_]), 0.0)[m_].max() * 1000) for n_, m_ in cfg["protect"].items()}
        else:        # --protect-abs: 예전 방식(절대 변화)
            pm = {n_: float(np.abs(h - h_cur)[m_].max() * 1000) for n_, m_ in cfg["protect"].items()}
        out.update(prot=pm, prot_max=max(pm.values()))
    if cfg.get("active"):                      # --own-score: 후보 순위·확정은 지금 보상하는 영역 하나의 점수로만 한다(영역 점수는 따로 계산, 다른 영역 점수는 보지 않는다)
        out.update(J_all=out["J"], J=region_J(cfg["active"], vis, cfg))
    return out


def local_gate(hc, h_rec, win, rep=None, max_mae=0.15, max_p99=0.5):
    """영역 국소 게이트: control 재생이 기록된 상태와 '영역 창 안'에서 맞는가.
    전역 게이트(0.5 mm 초과 화소 <=5개, p99.9 <=0.2 mm)는 맵 어딘가의 한두 화소(접촉 가장자리의 입자 하나)만 어긋나도 탈락시킨다.
    저장된 재생 실패 4건은 평균 어긋남이 0.001~0.009 mm 였고 0.5 mm 초과가 1/1/38/1 화소(전체 262,144)뿐이었다.
    반사실 효과는 같은 엔진의 두 재생의 차이(C = H_cf - H_ctrl)라서 영역 창 안의 어긋남이 효과보다 충분히 작으면 쓸 수 있다.
    hc: control 재생 높이맵, h_rec: 기록된 현재, win: 영역 창(bool), rep: 같은 control 을 한 번 더 돌린 복제(있으면).
    noise = 복제와의 창 안 평균 차이(없으면 기록과의 평균 차이로 대신, 보수적). 단위 mm."""
    dd = np.abs(hc - h_rec)[win] * 1000.0
    mae, p99 = float(dd.mean()), float(np.quantile(dd, 0.99))
    noise = mae if rep is None else float(np.abs(hc - rep)[win].mean() * 1000.0)
    return dict(mae=mae, p99=p99, noise=noise, ok=bool(mae <= max_mae and p99 <= max_p99))


# ---------------------------------------------------------------- worker 프로세스
def credit_adjust(r, base, side, cfg, w, J0):
    """다중 영역 이득 계산(--credit-side). 한 행동이 요청한 영역 여러 개를 동시에 고치면 그 이득도 센다. 대상·영역 구성에 의존하지 않는다.
    r: 워커가 돌려준 평가 결과(r["vis"] 에 모든 영역의 형태·기울기·테두리가 있다). r["J"] 는 지금 고치는(활성) 영역의 J 였다.
    base[k]: 라운드 시작 상태에서 요청 영역 k 의 region_J, side: 활성 영역 말고 요청한(건너뛰지 않은) 영역들, J0: 라운드 시작의 활성 영역 J.
    결과: J_active(원래 값) 보관, J = J_active - w * (side 영역들의 이득 합), side_gain(이득 합), side_worse(가장 나빠진 요청 영역의 악화량), active_worse(활성 영역 악화량)."""
    vis = r.get("vis")
    if not side or vis is None or r.get("J", 1e9) >= 1e8:
        r.setdefault("J_active", r.get("J"))
        r["side_gain"], r["side_worse"], r["active_worse"] = 0.0, 0.0, 0.0
        return r
    gains = {k: base[k] - float(region_J(k, vis, cfg)) for k in side}
    r["J_active"] = r["J"]
    r["side_gain"] = float(sum(gains.values()))
    r["side_worse"] = float(max(0.0, -min(gains.values())))
    r["active_worse"] = float(max(0.0, r["J_active"] - J0))
    r["J"] = r["J_active"] - w * r["side_gain"]
    return r


def worker_main(wid, tq, rq, cfg, start_state, viewer, display, material, tipcorr, check_sig):
    os.environ["VS_MATERIAL"] = material
    os.environ["VS_TIP_CORRECTION"] = "1" if tipcorr else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if viewer else "0"
    if display:
        os.environ["DISPLAY"] = display
    import torch
    import vs_simulator as sim
    try:
        from vs_heightmap_fast import apply_fast_height
        apply_fast_height(sim)      # 입자->높이맵 변환의 빠른 버전(결과 동일, 파일은 안 바꿈). VS_FAST_HEIGHT=0 이면 끔
    except Exception as _e_fh:
        print(f"[빠른 높이맵 적용 실패: {_e_fh!r}] 원본 함수를 쓴다", flush=True)
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature, tensor_save

    adapter = GenesisStateAdapter(sim)
    if check_sig is not None and physics_signature(sim) != check_sig:
        rq.put(("error", wid, "physics signature 불일치"))
        return
    adapter.restore(load_snapshot(start_state))
    h_cur = adapter.observe().numpy().astype(np.float64)
    state_tag, tcount = 0, 0
    save_dir = Path(cfg["save_dir"]) if cfg.get("save_dir") else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    rq.put(("ready", wid, h_cur.astype(np.float32)))
    while True:
        msg = tq.get()
        if msg[0] == "stop":
            break
        if msg[0] == "load":
            adapter.restore(load_snapshot(msg[1]))
            state_tag = msg[2] if len(msg) > 2 else state_tag
            h_cur = adapter.observe().numpy().astype(np.float64)
            rq.put(("loaded", wid))
        elif msg[0] == "active":                            # --own-score: 이 라운드에 보상하는 영역(이후 평가의 J 는 그 영역 점수)
            cfg["active"] = msg[1]
        elif msg[0] == "eval":
            _, cid, items = msg[:3]
            tgt = msg[3] if len(msg) > 3 else None          # 과거 반사실 목표 T (없으면 None)
            saved = adapter.capture()
            out, t0 = [], time.time()
            prefix = None                                   # --pair-branch: (첫 푸시, 그 직후 전체 물리 상태)
            for key, act in items:
                try:
                    seq = act if isinstance(act[0], (list, tuple)) else [act]      # 행동 하나 또는 연속 행동(2-step 앞내다보기)
                    if len(seq) >= 2 and cfg.get("pair_branch"):
                        pk = tuple(round(float(v), 9) for v in seq[0])
                        if prefix is not None and prefix[0] == pk:
                            adapter.restore(prefix[1])      # 같은 첫 푸시: 그 직후 상태에서 갈라진다(첫 푸시를 다시 실행하지 않는다)
                        else:
                            adapter.execute(torch.tensor(seq[0], dtype=torch.float32), 0)
                            prefix = (pk, adapter.capture())
                        for a_i in seq[1:]:
                            adapter.execute(torch.tensor(a_i, dtype=torch.float32), 0)
                    else:
                        for a_i in seq:
                            adapter.execute(torch.tensor(a_i, dtype=torch.float32), 0)
                    h = adapter.observe().numpy().astype(np.float64)
                    if save_dir is not None and len(seq) == 1:                      # 실행해 본 (상태, 행동, 결과)는 그대로 모델 미세조정용 데이터가 된다
                        np.savez_compressed(save_dir / f"w{wid}_{tcount:05d}.npz", action=np.asarray(seq[0], dtype=np.float64), delta=(h - h_cur).astype(np.float16), state_tag=state_tag)
                        tcount += 1
                    s = score(h, cfg, h_cur)
                    if tgt is not None:
                        s["dT"] = float(np.abs(h - tgt["T"])[tgt["win"]].mean() * 1000)   # 창 안에서 T 까지의 평균 거리(mm)
                except Exception as e:  # 한 후보의 실패가 전체를 멈추지 않게
                    s = dict(J=1e9, e=1e9, coll=0.0, far_max=1e9, per={}, error=repr(e))
                s["key"] = key
                s["action"] = [[float(v) for v in a_i] for a_i in act] if isinstance(act[0], (list, tuple)) else [float(v) for v in act]
                out.append(s)
                adapter.restore(saved)
            rq.put(("eval", wid, cid, out, time.time() - t0))
        elif msg[0] == "replay":
            # 과거 반사실 재생: 행동 start 직전 상태로 돌아가 (edit 가 있으면 그 행동만 바꿔) 마지막 행동까지 다시 실행하고 높이맵을 돌려준다.
            _, jid, rroot, start, last, edit, *rest_ = msg
            only_ = rest_[0] if rest_ else None                   # 주어지면 이 번호의 행동만 시간 순서대로 재생(나머지는 건너뜀)
            import counterfactual_replay as cfr
            saved = adapter.capture()
            t0 = time.time()
            try:
                pre, records, rngs = cfr.load_plan(Path(rroot), start)
                edits = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in (edit or {}).items()}
                if only_:
                    h_rep, _ = cfr.run_replay_subset(adapter, pre, records, rngs, [int(x) for x in only_], edits)
                else:
                    h_rep, _ = cfr.run_replay(adapter, pre, records, rngs, start, last, edits)
                res = (h_rep.astype(np.float32), None)
            except Exception as e:
                res = (None, repr(e))
            adapter.restore(saved)
            rq.put(("replay", wid, jid, res[0], res[1], time.time() - t0))
        elif msg[0] == "commit":
            _, act, path = msg[:3]
            state_tag = msg[3] if len(msg) > 3 else state_tag
            for a_i in (act if isinstance(act[0], (list, tuple)) else [act]):
                adapter.execute(torch.tensor(a_i, dtype=torch.float32), 0)
            h_cur = adapter.observe().numpy().astype(np.float64)
            tensor_save(Path(path), adapter.capture())
            rq.put(("committed", wid, h_cur.astype(np.float32)))


# ---------------------------------------------------------------- 후보 생성
class Geo:
    def __init__(self, g):
        self.g = g

    def pix2w(self, row, col):
        return (self.g["x0"] + (col + 0.5) / self.g["px_per_m_x"], self.g["y0"] + (row + 0.5) / self.g["px_per_m_y"])

    def mm_px(self):
        return 1e-3 * self.g["px_per_m_x"]


def variants_of(a, kind):
    x, y, th, L, d = a
    px, py = -math.sin(th), math.cos(th)
    ds, sh = kind
    if ds == "rev":
        return [x + L * math.cos(th), y + L * math.sin(th), wrap(th + math.pi), L, d * 0.5]
    return [x + sh * 1e-3 * px, y + sh * 1e-3 * py, th, L, d * ds]


def model_candidates(model, h_cur, cfg, n, tgt, mask, geo, top=6, pos_step_mm=6.0, n_theta=8, n_refine=8, deep=False):
    """모델 기반 제안(MDL): Param2Deform 으로 영역 근처의 (x,y,theta) 격자 x 공칭(길이,깊이)를 예측해 J(영역 n)로 순위를 매기고(beam),
    상위 위치마다 (길이,깊이) 격자를 다시 예측해 고른다(설계의 '자코비안+Adam' 자리를 격자 탐색으로 대신함; 모델이 싸서 가능).
    모델은 실제와 어긋나므로 이 결과는 '후보 제안'일 뿐이고, 확정은 Genesis 실행 후 롤백 검증으로만 한다. 반환: [(행동, 예측 J)]"""
    import torch
    from vs_planner import predict_next_batch
    x0, x1, y0, y1, l0, l1, d0, d1 = BOUNDS
    dev = next(model.parameters()).device
    win = ndimage.binary_dilation(mask, iterations=int(round(0.5 * geo.g["tool_r_px"])))
    step = max(int(round(pos_step_mm / geo.g["px_mm"])), 1)
    rr, cc = np.where(win)
    keep = (rr % step == 0) & (cc % step == 0)
    pts = [geo.pix2w(r, c) for r, c in zip(rr[keep], cc[keep])]
    cur = torch.as_tensor(h_cur, dtype=torch.float32, device=dev)[None, None]

    def pred_J(acts):
        out = []
        for i in range(0, len(acts), 12):
            a = torch.tensor(acts[i:i + 12], dtype=torch.float32, device=dev)
            with torch.no_grad():
                maps = predict_next_batch(model, cur, a)[:, 0].double().cpu().numpy()
            for m in maps:
                per_, vis_, coll_ = _terms(m, cfg)
                out.append(region_J(n, vis_, cfg) + cfg["mu"] * coll_ / cfg["coll_norm"])
        return out

    def ok_end(a):
        ex, ey = a[0] + a[3] * math.cos(a[2]), a[1] + a[3] * math.sin(a[2])
        return x0 <= ex <= x1 and y0 <= ey <= y1
    grid = [clampA([x, y, k * 2 * math.pi / n_theta - math.pi, 0.012, 0.015]) for (x, y) in pts for k in range(n_theta)]
    grid = [a for a in grid if ok_end(a)]
    if not grid:
        return []
    pj = pred_J(grid)
    order = np.argsort(pj)[:n_refine]
    cand = []
    for i in order:
        x, y, th = grid[i][:3]
        ref = [clampA([x, y, th, L, d]) for L in (0.006, 0.009, 0.012, 0.016, 0.020) for d in ((0.008, 0.012, 0.016, 0.020, 0.026, 0.030) if deep else (0.008, 0.012, 0.016, 0.020, 0.026))]
        ref = [a for a in ref if ok_end(a)]
        cand += list(zip(ref, pred_J(ref)))
    cand.sort(key=lambda t: t[1])
    out, seen = [], set()
    for a_, j_ in cand:
        k = akey(a_)
        if k not in seen:
            seen.add(k)
            out.append((a_, j_))
        if len(out) >= top:
            break
    return out


def model_rank_J(model, h_cur, cfg, n, acts, batch=12):
    """학습한 동역학 모델(Param2Deform)로 후보 행동의 결과를 예측해 영역 n 의 J 를 돌려준다(싸다). 후보를 Genesis 로 평가하기 전에 위치·크기를 고르는 데 쓴다.
    모델은 실제와 어긋날 수 있어서 '후보 제안'일 뿐이고, 확정은 항상 Genesis 실행 결과로 한다."""
    import torch
    from vs_planner import predict_next_batch
    dev = next(model.parameters()).device
    cur = torch.as_tensor(h_cur, dtype=torch.float32, device=dev)[None, None]
    out = []
    for i in range(0, len(acts), batch):
        a_ = torch.tensor(acts[i:i + batch], dtype=torch.float32, device=dev)
        with torch.no_grad():
            maps = predict_next_batch(model, cur, a_)[:, 0].double().cpu().numpy()
        for m in maps:
            _, vis_, _ = _terms(m, cfg)
            out.append(float(region_J(n, vis_, cfg)))
    return out


def model_rank_match(model, h_cur, T, win, acts, far_w=0.1, batch=12, blob=None, coll_w=0.5):
    """학습한 동역학 모델로 후보 푸시의 결과를 예측해, 창 안에서 반사실 목표 T 까지의 평균 거리(mm)를 돌려준다. 작을수록 '얼마나 더 올리고/눌러야 하는지'(D = T - 현재)를 잘 맞춘 푸시다.
    창 밖 최대 변화(mm) x far_w 를 더해 엉뚱한 곳을 건드리는 후보를 뒤로 민다. 후보 제안일 뿐이고 확정은 항상 Genesis 실행 결과로 한다."""
    import torch
    from vs_planner import predict_next_batch
    dev = next(model.parameters()).device
    cur = torch.as_tensor(h_cur, dtype=torch.float32, device=dev)[None, None]
    win = np.asarray(win, bool)
    Tn = np.asarray(T, np.float64)
    blob = None if blob is None else np.asarray(blob, bool)
    rest = None if blob is None else (win & ~blob)
    out = []
    for i in range(0, len(acts), batch):
        a_ = torch.tensor(acts[i:i + batch], dtype=torch.float32, device=dev)
        with torch.no_grad():
            maps = predict_next_batch(model, cur, a_)[:, 0].double().cpu().numpy()
        for m in maps:
            far = far_w * float(np.abs(m - h_cur)[~win].max() * 1000.0)
            if blob is not None and blob.sum() >= 20:      # 덩어리 안에서 T 까지 남은 거리 + 덩어리 밖 창 안 부수 변화(coll_w 배, 제가 정한 가중치)
                out.append(float(np.abs(m - Tn)[blob].mean() * 1000.0) + coll_w * (float(np.abs(m - h_cur)[rest].mean() * 1000.0) if rest.any() else 0.0) + far)
            else:
                out.append(float(np.abs(m - Tn)[win].mean() * 1000.0) + far)
    return out


def pick_diverse_by_pred(acts, pj, k, per_pos=2):
    """예측 J 가 낮은 순으로 k 개를 고르되, 같은 시작 위치에서는 per_pos 개까지만(깊이만 다른 거의 같은 후보가 몰리지 않게)."""
    order = np.argsort(pj)
    cnt, out = {}, []
    for i in order:
        key = (round(float(acts[i][0]), 4), round(float(acts[i][1]), 4))
        if cnt.get(key, 0) >= per_pos:
            continue
        cnt[key] = cnt.get(key, 0) + 1
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def parse_cf_variants(spec):
    """'skip,1:0.5,1:1:4' -> [('skip',None,None), ('l1_d0.5',1.0,0.5), ('l1_d1_sh4',1.0,(1.0,4.0))]
    토큰: skip = 거의 안 누르는 행동으로 교체 | 길이배율:깊이배율 | 길이배율:깊이배율:옆으로 이동(mm, 진행 방향의 왼쪽이 +).
    옆으로 이동은 깊이 자리(df)에 (깊이배율, 이동mm) 튜플로 실려 variant_action 만 해석한다."""
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok == "skip":
            out.append(("skip", None, None))
            continue
        # 위치·방향 편집(2026-11-16, 반사실 확대): sh<mm> = 옆으로 이동(진행 방향 왼쪽 +), al<mm> = 진행 방향 앞으로 이동, rot<도> = 방향 회전. 깊이·길이는 그대로
        if tok.startswith("sh") and len(tok) > 2 and tok[2] in "+-0123456789.":
            mm = float(tok[2:])
            out.append((f"sh{mm:+g}", 1.0, (1.0, mm, 0.0, 0.0)))
            continue
        if tok.startswith("al") and len(tok) > 2 and tok[2] in "+-0123456789.":
            mm = float(tok[2:])
            out.append((f"al{mm:+g}", 1.0, (1.0, 0.0, mm, 0.0)))
            continue
        if tok.startswith("rot") and len(tok) > 3 and tok[3] in "+-0123456789.":
            dg = float(tok[3:])
            out.append((f"rot{dg:+g}", 1.0, (1.0, 0.0, 0.0, dg)))
            continue
        v = [float(x) for x in tok.split(":")]
        if len(v) == 3:
            out.append((f"l{v[0]:g}_d{v[1]:g}_sh{v[2]:g}", v[0], (v[1], v[2])))
        else:
            out.append((f"l{v[0]:g}_d{v[1]:g}", v[0], v[1]))
    return out


def variant_action(a, lf, df):
    """과거 행동 a 의 변형 (재생에 쓴다). lf=None 이면 skip: 길이·깊이 최소."""
    import vs_config as vc
    x, y, th, L, d = [float(v) for v in a]
    lmax = BOUNDS[5] if BOUNDS else vc.LENGTH_MAX        # 로봇이 안전하게 실행하는 범위(플래너와 같음)를 넘기지 않는다
    dmax = BOUNDS[7] if BOUNDS else vc.DEPTH_MAX
    if lf is None:
        return [x, y, th, vc.LENGTH_MIN, vc.DEPTH_MIN]
    sh = al = rot = 0.0
    if isinstance(df, tuple):                              # (깊이배율, 옆으로 이동 mm[, 앞으로 이동 mm, 회전 도])
        df, sh, *rest_ = df
        al = rest_[0] if len(rest_) > 0 else 0.0
        rot = rest_[1] if len(rest_) > 1 else 0.0
    if sh or al:                                           # 이동은 원래 방향(th) 기준: 옆 = 왼쪽(+), 앞 = 진행 방향(+)
        x, y = x + sh * 1e-3 * (-math.sin(th)) + al * 1e-3 * math.cos(th), y + sh * 1e-3 * math.cos(th) + al * 1e-3 * math.sin(th)
        if BOUNDS:
            x, y = float(min(max(x, BOUNDS[0]), BOUNDS[1])), float(min(max(y, BOUNDS[2]), BOUNDS[3]))
    if rot:
        th = th + math.radians(rot)
    return [x, y, th, float(min(max(L * lf, vc.LENGTH_MIN), lmax)), float(min(max(d * df, vc.DEPTH_MIN), dmax))]


def reverse_candidates(a, fracs=(0.25, 0.5, 1.0), shifts=(0.0, 4.0, -4.0)):
    """과거 행동 a 를 '되돌리는' 현재 행동 후보: 끝점에서 시작점 쪽으로 미는 역방향 푸시. 깊이는 원래의 fracs 배, 옆으로 shifts mm.
    과거에 덜 눌렀다면(C*) 현재 점토에서 그 효과를 만들려면 그만큼을 되돌려야 하기 때문이다."""
    x, y, th, L, d = [float(v) for v in a]
    px, py = -math.sin(th), math.cos(th)
    ex, ey = x + L * math.cos(th), y + L * math.sin(th)
    return [[ex + sh * 1e-3 * px, ey + sh * 1e-3 * py, wrap(th + math.pi), L, d * f] for f in fracs for sh in shifts]


def adaptive_quota(base_q, stats, prior_n=4.0, floor=1):
    """계열별 성공률(= J 가 현재보다 좋아진 후보 비율)에 따라 다음 라운드 쿼터를 옮긴다. 총 개수는 그대로.
    성공률은 전체 평균 쪽으로 당긴(prior_n 개분) 추정치를 쓰고, 어떤 계열도 floor 아래로 내리지 않는다(탐색 유지).
    stats: {계열: [개선 수, 평가 수]}"""
    fams = list(base_q)
    tot_imp = sum(v[0] for v in stats.values())
    tot_n = sum(v[1] for v in stats.values())
    p0 = (tot_imp + 1.0) / (tot_n + 4.0)
    p = {f: (stats.get(f, [0, 0])[0] + prior_n * p0) / (stats.get(f, [0, 0])[1] + prior_n) for f in fams}
    B, Z = sum(base_q.values()), sum(p.values())
    q = {f: max(floor, int(round(B * p[f] / Z))) for f in fams}
    while sum(q.values()) > B:                      # 반올림으로 늘어난 만큼 가장 큰 계열에서 덜어낸다
        f = max((f for f in fams if q[f] > floor), key=lambda f: q[f], default=None)
        if f is None:
            break
        q[f] -= 1
    while sum(q.values()) < B:
        q[max(fams, key=lambda f: p[f] / (q[f] + 1))] += 1
    return q, p


def _ok_end(a):
    """스트로크 끝점이 플래너의 안전 범위 안인가"""
    x0, x1, y0, y1 = BOUNDS[:4]
    ex, ey = a[0] + a[3] * math.cos(a[2]), a[1] + a[3] * math.sin(a[2])
    return x0 <= ex <= x1 and y0 <= ey <= y1


def neighbors(a):
    x, y, th, L, d = a
    px, py = -math.sin(th), math.cos(th)
    tx, ty = math.cos(th), math.sin(th)
    out = []
    for s in (-2e-3, 2e-3):
        out.append([x + s * px, y + s * py, th, L, d])
    for s in (-3e-3, 3e-3):
        out.append([x + s * tx, y + s * ty, th, L, d])
    for s in (-0.25, 0.25):
        out.append([x, y, th + s, L, d])
    for f in (0.7, 1.4):
        out.append([x, y, th, L * f, d])
    for f in (0.7, 1.4):
        out.append([x, y, th, L, d * f])
    return out


def blobs(mask, min_px):
    lab, n = ndimage.label(mask)
    out = []
    for k in range(1, n + 1):
        b = lab == k
        if b.sum() >= min_px:
            out.append(b)
    return out


def transport_candidates(geo, h, tgt, mask, n_blobs=2, thr=0.002):
    """영역 안 부족 덩어리(구덩이)마다 주변 남는 점토 쪽에서 구덩이 쪽으로 미는 후보."""
    mm = geo.mm_px()
    res = tgt - h
    near = ndimage.binary_dilation(mask, iterations=int(round(1.2 * geo.g["tool_r_px"])))
    deficit = ndimage.binary_opening((res > thr) & mask, iterations=1)
    bl = sorted(blobs(deficit, 25), key=lambda b: -float(res[b].sum()))[:n_blobs]
    excess = np.clip(-res, 0, None) * near
    rr, cc = np.mgrid[0:SIZE, 0:SIZE]
    out = []
    for b in bl:
        w = res[b]
        cy, cx = float((rr[b] * w).sum() / w.sum()), float((cc[b] * w).sum() / w.sum())
        sc = []
        for k in range(16):
            ph = k * math.pi / 8
            ux, uy = math.cos(ph), math.sin(ph)
            # 구덩이에서 -u 방향 8~25mm 부채꼴에 있는 남는 점토
            dx, dy = (cc - cx) / mm, (rr - cy) / mm
            dist = np.hypot(dx, dy) + 1e-9
            cosang = -(dx * ux + dy * uy) / dist
            sector = (dist > 8) & (dist < 25) & (cosang > math.cos(math.radians(25)))
            sc.append((float(excess[sector].sum()), ph))
        sc.sort(reverse=True)
        for _, ph in sc[:3]:
            ux, uy = math.cos(ph), math.sin(ph)
            for rho in (6, 12):
                sx, sy = cx - rho * mm * ux, cy - rho * mm * uy
                x, y = geo.pix2w(sy, sx)
                for L in (0.012,):
                    for d in (0.010, 0.018):
                        out.append(clampA([x, y, ph, L, d]))
    return out


def flatten_candidates(geo, h, tgt, mask, n_blobs=2, thr=0.002):
    res = h - tgt
    ex = ndimage.binary_opening((res > thr) & mask, iterations=1)
    rr, cc = np.mgrid[0:SIZE, 0:SIZE]
    out = []
    for b in sorted(blobs(ex, 30), key=lambda b: -float(res[b].sum()))[:n_blobs]:
        w = res[b]
        cy, cx = float((rr[b] * w).sum() / w.sum()), float((cc[b] * w).sum() / w.sum())
        x, y = geo.pix2w(cy, cx)
        for ph in (0.0, math.pi / 2):
            out.append(clampA([x, y, ph, 0.006, 0.010]))
    return out


def dictionary(actions, before, after, fps, mask, tgt, h_cur, near_px, win_px, thr_mm=0.15):
    """기록된 높이 변화 dH_t 로 현재 오차를 설명하는 제한 최소제곱. 반환: [(t, w, 영역 안 영향 rms mm)]
    near_px: 접촉영역이 영역에서 이 거리 안을 지나면 관련 행동. win_px: 최소제곱을 푸는 영역 주변 창의 폭(둘 다 도구 반경 기준)."""
    near = ndimage.binary_dilation(mask, iterations=near_px)
    W = ndimage.binary_dilation(mask, iterations=win_px)
    rel = []
    for t in range(len(actions)):
        d = (after[t] - before[t]) * 1000
        if (fps[t] & near).any() or float(np.abs(d)[mask].mean()) >= thr_mm:
            rel.append((t, d))
    if not rel:          # 이 영역에 닿았거나 높이를 바꾼 과거 행동이 없다 -> G1 후보 없음 (G2/G3 만 쓴다)
        return []
    A = np.stack([d[W] for _, d in rel], 1).astype(np.float64)
    r = ((tgt - h_cur) * 1000)[W].astype(np.float64)
    lam = 0.05 * float((A ** 2).sum(0).mean())
    Aa = np.vstack([A, math.sqrt(lam) * np.eye(A.shape[1])])
    ba = np.concatenate([r, np.zeros(A.shape[1])])
    w = lsq_linear(Aa, ba, bounds=(-1.0, 1.5)).x
    out = []
    for (t, d), wt in zip(rel, w):
        out.append((t, float(wt), float(np.sqrt((d[mask] ** 2).mean()))))
    return out


# ---------------------------------------------------------------- 그림
def make_frame(path, h, cfg, hstar_all, bg, crop, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r0, r1, c0, c1 = crop
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#fcfcfb")
    kw = dict(cmap="turbo", vmin=-10, vmax=14)
    ax[0].imshow(((h - bg) * 1000)[r0:r1, c0:c1], **kw); ax[0].set_title("Current height map (mm above background)", fontsize=9)
    ax[1].imshow(((hstar_all - bg) * 1000)[r0:r1, c0:c1], **kw); ax[1].set_title("Target (requested regions)", fontsize=9)
    im = ax[2].imshow(((h - hstar_all) * 1000)[r0:r1, c0:c1], cmap="RdBu_r", vmin=-6, vmax=6); ax[2].set_title("Current - target (mm): red = too high, blue = too low", fontsize=9)
    for a in ax:
        for _, m, _ in cfg["regs"]:
            a.contour(m[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
        a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=85, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int, default=None, help="조형 K번째 상태를 '현재'로 삼고 앞쪽 K개 행동만 과거로 본다(없으면 이력 전체)")
    ap.add_argument("--regions", "--s80b", dest="regions", required=True, type=Path,
                    help="영역 집합 폴더(build_regions.py 출력, live_edit.py 출력, 또는 <이름>_mask.npy/<이름>_target.npy 짝이 있는 폴더)")
    ap.add_argument("--ref-h", type=Path, default=None,
                    help="영역 밖 변화를 재는 기준 높이맵(m). 없으면 영역 폴더의 h_ref.npy, 그것도 없으면 보상 시작 상태")
    ap.add_argument("--start-state", type=Path, default=None, help="보상을 시작할 상태 스냅샷. 없으면 (--upto 가 가리키는) 이력 마지막 post_state")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--names", default="", help="영역 일부만 쓸 때 쉼표로 구분한 이름. 없으면 전부")
    ap.add_argument("--depth-min-mm", type=float, default=2.0, help="반사실 후보의 깊이 하한(깊이를 줄인 변형을 보려고 플래너 하한보다 낮춘다)")
    ap.add_argument("--near-tools", type=float, default=0.5, help="영역에서 도구 반경의 이 배수 안을 지난 행동을 관련 행동으로 본다")
    ap.add_argument("--win-tools", type=float, default=1.2, help="과거 행동 가중치를 푸는 영역 주변 창의 폭(도구 반경의 배수)")
    ap.add_argument("--ring-tools", type=float, default=0.94, help="'주변 변화' 고리의 폭(도구 반경의 배수)")
    ap.add_argument("--far-tools", type=float, default=1.17, help="영역 밖 변화 제한을 재는 영역에서의 거리(도구 반경의 배수)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--viewer", action="store_true", help="워커 0 에 Genesis 창을 띄운다. 기본으로 이 워커도 후보 평가에 참여해(눌러 보고 되돌리는 모습이 보임) 확정도 이 창에서 실행한다")
    ap.add_argument("--no-viewer-eval", dest="viewer_eval", action="store_false",
                    help="창 워커는 확정 행동만 실행하고 후보 평가는 창 없는 워커들만 한다(창 렌더링으로 느려지는 것을 피함)")
    ap.add_argument("--max-commits", type=int, default=30)
    ap.add_argument("--budget-min", type=float, default=120)
    ap.add_argument("--eps", type=float, default=0.01, help="확정에 필요한 J 개선량(mm)")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--blur-tools", type=float, default=0.4, help="형태·외형을 재는 흐림 크기(도구 반경의 배수). 0 이면 흐리지 않은 원본 픽셀(입자 질감 포함)")
    ap.add_argument("--w-height", type=float, default=1.0, help="J 의 형태(흐린 높이) 오차 가중치")
    ap.add_argument("--w-tail", type=float, default=0.0,
                    help="J 의 꼬리('목표를 지나친 양' 상위 5% 평균, 가벼운 블러에서 잼) 가중치. 0=끔(이전과 같은 J). 홈 파기에서 한 번 눌러 구덩이 하나만 생기는 것을 막으려면 1 이상. 값은 제 선택이고 검증된 것이 아님")
    ap.add_argument("--w-grad", type=float, default=1.0, help="J 의 외형(기울기 오차, Lviz 계열) 가중치. 0 이면 높이만 본다")
    ap.add_argument("--w-edge", type=float, default=1.0, help="J 의 테두리 선명도 가중치")
    ap.add_argument("--far-limit-mm", type=float, default=6.0, help="초점(글자+근처) 안에서 영역 밖(영역에서 far-tools 이상 떨어진 곳)의 변화 상한. 배경(초점 밖)은 제한하지 않는다")
    ap.add_argument("--focus-tools", type=float, default=0.0,
                    help="목표의 모양(부조/음각)에서 도구 반경의 이 배수 이내를 '글자+근처'로 보고 그 안만 점수·제약에 쓴다. 0(기본)이면 끄고 영역 주변만 본다(v5 방식). 글자 전체를 따로 보고 싶을 때만 쓴다")
    ap.add_argument("--w-focus", type=float, default=0.0, help="글자+근처 전체의 형태·기울기·테두리 오차를 J 에 섞는 가중치(영역 항 1 : w_focus). 0 이면 영역만")
    ap.add_argument("--base-mm", type=float, default=None, help="목표의 판 높이(mm). 기본: 이력 manifest 의 목표 변환에서 읽고 없으면 히스토그램 최빈값")
    ap.add_argument("--target-npy", type=Path, default=None, help="초점을 정할 목표(512x512, m). 없으면 조형 때 쓴 목표")
    ap.add_argument("--g1-top", type=int, default=8)
    ap.add_argument("--carry", type=int, default=12)
    ap.add_argument("--refine-top", type=int, default=4)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--mode", choices=["full", "beam"], default="beam",
                    help="full: 후보 전부를 물리 평가. beam: 기록 dH 대리 점수로 후보 전부를 순위 매긴 뒤 계열별 쿼터(다양성 빔)만 물리 평가")
    ap.add_argument("--quota", default="G1:6,G2:5,G3:3,carry:4,CFR:9,MDL:6,PLAN:4,DEEP:6", help="beam 모드 계열별 물리 평가 수 (CFR = 과거 반사실을 되돌리는 후보)")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/param2deform_medium_paper_planner_range_deep150.pt"),
                    help="모델 기반 제안(MDL)에 쓰는 Param2Deform 체크포인트 (읽기만 한다)")
    ap.add_argument("--pairs", type=int, default=4,
                    help="2-step 앞내다보기: 상위 K 개 단일 후보로 만든 순서쌍(a1 후 a2)을 물리로 평가해 둘 다 해야 J 가 좋아지는 경우를 찾는다(한 번에 하나씩만 보면 국소 최솟값에 갇힘). 0 이면 끔")
    ap.add_argument("--no-adaptive-quota", dest="adaptive_quota", action="store_false",
                    help="계열별 쿼터를 --quota 로 고정한다(기본은 지금까지의 계열별 성공률에 따라 라운드마다 옮김)")
    ap.add_argument("--lin-cf", type=int, default=0,
                    help="재생 없이 쓰는 선형(가산) 반사실: 영역마다 영향 큰 과거 행동 상위 N개(시간 거리 무관, 80개 전체)에 대해 C = 배율 x (그 행동이 기록한 높이 변화)를 현재에 더한 목표 T 를 만든다. "
                         "w<0(덜 눌렀어야)이면 배율 -1(건너뛰기), -0.5(절반), w>0(더 눌렀어야)이면 +1(한 번 더), +0.5. 재생 범위(--replay-len) 밖 행동도 다룰 수 있다. 재생 범위 안 10~18개 전 행동에서 진짜 재생과 상관 0.80~0.96 으로 검증, 더 먼 과거는 미검증")
    ap.add_argument("--ring-budget-mm", type=str, default=None,
                    help="영역 주변 고리의 평균 변화 허용 한도(mm). 숫자 하나(0.47)는 모든 영역 주변 전체(합친 고리)에 같은 한도. 영역마다 다르게: 'groove_top=0.47,groove_bottom=0.74,*=0.5'"
                         "(*는 나머지 영역) 또는 conditions.py 출력 JSON 파일 경로(영역별 ring_budget_mm). 영역별 한도는 후보가 '모든' 한도를 동시에 지킬 때만 통과시킨다(어느 영역의 주변도 허용 이상으로 망가뜨리지 않음). "
                         "주면 가중치(--mu)는 0 으로 두고, 이 한도 이하인 후보 중에서 요청 영역 점수가 가장 좋은 것을 고른다. "
                         "기준은 실행 시작 상태(--ref-h) 대비 누적 변화다. 가중치처럼 대상마다 맞출 추상적인 값이 아니라 mm 단위의 한도라서 대상이 바뀌어도 해석이 같다(예: 0.1)")
    ap.add_argument("--plan-cands", type=int, default=0,
                    help="계획기(greedy MPC)가 지금 상태에서 요청 영역의 목표를 향해 제안하는 첫 행동을 이 횟수만큼 뽑아 후보(PLAN)로 넣는다(0 이면 끔). 영역에서 멀리 떨어진 곳을 누르는 행동도 후보가 된다")
    ap.add_argument("--deep", action="store_true",
                    help="깊게 누르는 후보를 추가한다: 영향 큰 과거 행동의 깊이 x1.5, x2 변형(DEEP)과 모델 제안의 깊이 격자를 30 mm 까지 확장")
    ap.add_argument("--rel-replay", type=int, default=0,
                    help="영역과 가장 관련 있는 행동 M개(시간 순서와 무관, 영향이 큰 순)를 작업 목록으로 잡아 그 행동들만 시간 순서대로 재생한다(0 이면 끔). "
                         "그 사이의 다른 행동은 영역과 무관하다고 보고 건너뛰므로 M 개만 재생하면 되고, 행동들이 과거에 멀리 흩어져 있어도 된다. control 은 영역 창 안에서 기록과 맞는지 확인한다")
    ap.add_argument("--rel-top", type=int, default=1, help="--rel-replay 목록 중 영역마다 실제로 바꿔 볼 행동 수(영향이 큰 순)")
    ap.add_argument("--far-top", type=int, default=0,
                    help="재생 범위(--replay-len) 밖의 영향 큰 과거 행동을 영역마다 이 개수만큼 '구간 재생 + 덮어쓰기 마스크'로 다룬다(0 이면 끔). "
                         "그 행동부터 --replay-len 개 행동만 재생(검증된 길이)해 효과를 얻고, 그 뒤 행동의 접촉영역이 덮어쓴 곳은 효과가 사라진 것으로 보고 뺀다")
    ap.add_argument("--persist-tools", type=float, default=0.5,
                    help="덮어쓰기 판정: 구간 이후 행동의 접촉영역을 도구 반경의 이 배수만큼 넓힌 곳은 효과가 남지 않은 것으로 본다(가정이며, 점토가 주변으로 흐르는 것은 못 잡음)")
    ap.add_argument("--skip-regions", default="", help="쉼표로 구분한 영역 이름: 이 영역은 직접 고치는 라운드를 하지 않는다(J 에는 남아 다른 영역을 고치다 망가지는지 감시). 이전 실행에서 이미 다룬 영역을 건너뛸 때 쓴다")
    ap.add_argument("--jac-top", type=int, default=4, help="자코비안+Adam+Broyden(JAC) 후보를 기준 행동마다 물리 평가하는 수. 0 이면 끔 (모델이 있어야 함)")
    ap.add_argument("--jac-bases", type=int, default=2, help="JAC 의 기준 행동 수(물리 평가에서 J 가 낮은 상위 개수)")
    ap.add_argument("--jac-iters", type=int, default=2, help="JAC 반복 횟수(매번 새로 물리로 잰 이웃으로 Broyden 보정을 다시 한다)")
    ap.add_argument("--jac-radius", type=float, default=10.0, help="Broyden 보정에 쓰는 이웃의 최대 거리(SCALE 단위: x,y,L,d 1mm, theta 0.05rad)")
    ap.add_argument("--save-transitions", type=Path, default=None,
                    help="물리로 실행해 본 후보마다 (행동, 결과 변화)를 저장하는 폴더. Param2Deform 미세조정용 on-policy 데이터로 쓴다")
    ap.add_argument("--no-model", dest="use_model", action="store_false", help="모델 기반 제안(MDL)을 끈다")
    ap.add_argument("--mdl-top", type=int, default=6, help="영역마다 모델이 제안하는 후보 수")
    ap.add_argument("--no-cf-replay", dest="cf_replay", action="store_false",
                    help="과거 반사실 재생 단계를 끈다(현재 상태에서의 후보 평가만 한다)")
    ap.add_argument("--replay-len", type=int, default=20, help="재생을 믿을 수 있는 범위: 현재 상태 직전 이 개수의 행동만 재생한다 (replay 검증: 길이 20 통과)")
    ap.add_argument("--cf-top", type=int, default=1, help="영역마다 재생할 과거 행동 수 (재생이 비싸서 작게)")
    ap.add_argument("--cf-variants", default="skip,1:0.5", help="과거 행동을 '덜 했어야'(가중치 w<0) 할 때의 변형: skip, 길이배율:깊이배율")
    ap.add_argument("--cf-variants-more", default="1:1.5,1.3:1", help="과거 행동을 '더 했어야'(w>0) 할 때의 변형. 방향이 w 의 부호를 따라야 반사실이 목표 쪽으로 간다. 길이배율:깊이배율[:옆이동mm] 예) 1:1.5,1.3:1,1:1:4 "
                         "(깊이 x1.5 는 효과가 선형 근사의 약 21%로 포화한다는 회귀 결과가 있어 길이·위치 변형을 함께 두는 것을 권한다)")
    ap.add_argument("--cf-both-dirs", action="store_true",
                    help="과거 행동마다 가중치 w 의 부호가 가리키는 방향의 변형뿐 아니라 반대 방향 변형(덜 눌렀다면 + 더 눌렀다면)도 재생한다('이만큼 눌렀으면 목표였을 텐데'를 양쪽으로 탐색). w 가 0 에 가까우면 방향이 불확실하다. "
                         "부호 방향 변형이 앞에 오므로 트리·묶음 탐색이 쓰는 첫 변형은 그대로다. 재생 작업이 늘어난다")
    ap.add_argument("--replay-cache", type=Path, default=None,
                    help="재생 결과 캐시 폴더. 같은 이력·같은 시점(K)·같은 시작 행동·같은 변형이면 재생을 다시 하지 않는다 (기본: 이력 폴더 옆 cf_replay_cache/K<K>)")
    ap.add_argument("--max-drift-mm", type=float, default=0.5, help="control 재생이 기록과 맞는지 보는 기준")
    ap.add_argument("--max-bad-px", type=int, default=5)
    ap.add_argument("--max-p999-mm", type=float, default=0.2)
    ap.add_argument("--w-cf", type=float, default=1.0, help="확정 후보 중 반사실 목표 T 에 가까운 것을 고르는 가중치 (J 가 좋아지는 후보만 대상)")
    ap.add_argument("--gate", choices=["global", "local"], default="global",
                    help="과거 재생(② 20행동 범위, ②b 구간)의 control 게이트. global=맵 전체 기준(기본, 이전과 같음), local=영역 창 안 기준(local_gate). ②L 장거리는 항상 local")
    ap.add_argument("--local-drift-mm", type=float, default=0.15, help="local 게이트: 영역 창 안에서 control 재생과 기록의 평균 어긋남 상한(mm). 제가 정한 값")
    ap.add_argument("--local-p99-mm", type=float, default=0.5, help="local 게이트: 같은 창 안 어긋남의 99 분위 상한(mm). 제가 정한 값")
    ap.add_argument("--min-snr", type=float, default=5.0, help="②L 장거리: 반사실 효과(영역 평균 |C|) / 잡음 이 이 값 이상일 때만 사용")
    ap.add_argument("--long-top", type=int, default=0,
                    help="②L 장거리 반사실: 영역마다 재생 범위 밖(보통 20행동 전보다 오래된)에서 영향 큰 과거 행동 N개를 현재까지 '전체 길이' 재생한다(0=끔). "
                         "행동당 17~20초 x 길이라 #2 는 20분 이상. 홈을 판 #2, #29 같은 오래된 큰 효과를 쓰려는 것")
    ap.add_argument("--long-budget-min", type=float, default=0.0,
                    help="②L 장거리 재생에 쓸 시간 예산(분, 워커 병렬 반영한 벽시계 추정). 0 이면 --long-top 개수로 고른다. >0 이면 이득/비용 순으로 예산이 찰 때까지 고른다: "
                         "이득 = |w| x 영역 안 영향, 비용 = 재생해야 하는 행동 수 x (control + 복제 + 변형 수)")
    ap.add_argument("--replay-sec-per-action", type=float, default=20.0, help="재생 시간 추정용: 행동 하나를 다시 실행하는 데 걸리는 초 (이전 측정 17~20초, 다른 Genesis 작업과 겹치면 더 걸림)")
    ap.add_argument("--long-group", type=int, default=0,
                    help="②L 그룹(시퀀스) 반사실: 영역마다 재생 범위 밖에서 영향 큰 과거 행동 상위 M개를 '한꺼번에' 바꿔(각자 w 의 부호 방향 변형) 가장 이른 행동부터 현재까지 한 번 재생한다(0=끔, 2 이상). "
                         "홈처럼 여러 번의 누름이 쌓여 만든 모양을 개별 행동이 아니라 묶음으로 바꿔 보는 것. control 은 개별 재생과 같은 것을 공유한다")
    ap.add_argument("--long-select", choices=["rank", "tree", "seq", "both"], default="rank",
                    help="②T 과거 행동 선택 방식. rank=예전(--long-top/--long-budget-min/--long-group). tree=계층(트리) 탐색: 관련 행동을 시간 구간 묶음(노드)으로 나눠 묶음째 Genesis 에서 재생하고, "
                         "효과가 있는 묶음만 쪼개 원인 행동을 좁힌다(효과 없는 묶음은 통째로 버림). 재생 횟수를 줄이고 시퀀스(묶음) 반사실을 만든다. 가지치기는 근사가 아니라 Genesis 재생 결과로 한다")
    ap.add_argument("--tree-init", type=int, default=4, help="②T: 처음에 관련 행동을 사전 점수 질량이 비슷하게 나눌 구간(노드) 수")
    ap.add_argument("--tree-min-prior-frac", type=float, default=0.05,
                    help="②T: 효과 있는 묶음을 쪼갤 때 사전 점수(|w|*영향 합)가 묶음의 이 비율 미만인 쪽은 재생하지 않는다(0=끔). 낮은 점수 행동까지 긴 재생을 하는 낭비를 줄이지만 그쪽의 효과는 알 수 없다")
    ap.add_argument("--tree-budget-min", type=float, default=60.0, help="②T: 영역마다 재생에 쓸 벽시계 분 예산(실측 시간으로 차감). 다 쓰면 남은 노드는 재생하지 않는다")
    ap.add_argument("--control", choices=["replay", "record"], default="replay",
                    help="②T 의 control. replay=같은 구간을 바꾸지 않고 다시 재생(기본). record=기록된 현재 상태를 control 로 쓴다: control·복제 재생이 필요 없어 재생이 절반으로 준다. "
                         "캐시에 이 영역 창에서 국소 게이트를 통과한 control 재생이 있고 그 길이의 --control-extrap 배 이내일 때만 쓰고, 더 긴 구간은 control 을 직접 재생한다. "
                         "검증(A, 52개 쌍): 영역 점수 효과 차이 최대 0.022, 큰 효과는 1% 미만, 부호 100% 일치, Spearman +0.965")
    ap.add_argument("--control-extrap", type=float, default=1.3, help="--control record: 검증된 control 최대 재생 길이의 몇 배까지 외삽을 허용할지")
    ap.add_argument("--noise-per-action-mm", type=float, default=0.0005, help="--control record 의 잡음 추정: 재생 길이(행동 수) x 이 값(mm). A 에서 같은 control 복제 차이가 79행동 0.033mm, 52행동 0.021mm")
    ap.add_argument("--noise-floor-mm", type=float, default=0.01, help="--control record 의 잡음 하한(mm)")
    ap.add_argument("--long-replicates", type=int, default=1,
                    help="②L: control 을 몇 번 돌릴지(2 이상이면 같은 control 복제로 재생 잡음을 직접 잰다)")
    ap.add_argument("--no-hints", action="store_true",
                    help="절제 실험: 과거 기록 힌트를 모두 끈다. 과거 행동에서 파생한 후보(G1 변형, DEEP, CFR/TRK)와 모든 과거 재생(②, ②L, ②T)·선형 반사실을 쓰지 않고, "
                         "현재 상태에서 만든 후보(G2/G3 점토 이동·평탄화, MDL, PLAN)와 그 주변 탐색(refine)·2-step 만 쓴다. 같은 설정에서 과거 힌트가 얼마나 도움이 되는지 재는 용도")
    ap.add_argument("--refine-mode", choices=["pattern", "bads", "spsa"], default="pattern",
                    help="빔 평가 뒤 정제 방식. pattern=기존(상위 후보 주변 이웃을 모두 평가). bads=이미 평가한 후보(위치·방향·길이·깊이 -> 영역 J)로 국소 가우시안 과정을 맞추고 "
                         "기대 개선이 큰 소수를 묶음으로 평가(BADS 원리, 묶음 안 후보가 몰리지 않게 지역 벌점). spsa=동시 섭동 비교 방법. 대리모델이 실패하면 pattern 으로 대신한다")
    ap.add_argument("--bads-evals", type=int, default=12, help="적응형 정제(bads/spsa)가 쓰는 물리 평가 수 상한")
    ap.add_argument("--bads-batch", type=int, default=0, help="적응형 정제에서 한 번에 평가하는 후보 수(0=평가 워커 수)")
    ap.add_argument("--pibo", action="store_true",
                    help="πBO 사전: 과거 반사실 변화 C* 와 지금 필요한 변화(목표-현재)가 같은 방향인 곳을 지나는 푸시에 사전 가중치를 준다. 빔 후보 순위와 적응형 정제의 획득 함수에 "
                         "사전^(beta/(1+관측 수))를 곱해, 이 라운드의 물리 평가가 쌓일수록 영향이 줄어든다. 반사실이 없는 영역(힌트 없는 팔 포함)에서는 순위를 바꾸지 않는다")
    ap.add_argument("--pibo-beta", type=float, default=3.0, help="πBO 사전의 초기 지수(관측 0개일 때). 제가 정한 값")
    ap.add_argument("--diverse", action="store_true", help="빔이 계열별 후보를 고를 때 서로 가까운 후보가 몰리지 않게 지역 벌점을 적용한다(기본은 순서대로 일정 간격 추출)")
    ap.add_argument("--penalize-ell", type=float, default=0.6, help="지역 벌점이 미치는 거리(정규화한 푸시 특징 공간). 제가 정한 값")
    ap.add_argument("--pair-branch", action="store_true",
                    help="2-step 순서쌍 평가에서 첫 푸시 직후의 전체 물리 상태를 저장해 두고, 같은 첫 푸시로 시작하는 쌍들이 그 상태에서 갈라지게 한다(첫 푸시를 쌍마다 다시 실행하지 않는다)")
    ap.add_argument("--pair-seconds", type=int, default=0, help="--pair-branch 에서 두 번째 푸시 후보 수(0=--pairs 와 같음). 첫 푸시는 상위 --pairs 개")
    ap.add_argument("--refine-patience", type=int, default=0,
                    help="패턴 탐색(정제) 단계의 조기 종료: 최선 J 가 0.002 이상 좋아지지 않은 채 연속으로 이 개수를 평가하면 남은 이웃 평가를 건너뛴다(0=끔). "
                         "저장된 11개 라운드 기록 모의: 4 -> 평가 수 5.9/17.9(-67%%), 최선 J 평균 손실 0.006(최대 0.031), 6 -> 7.9개, 손실 0.006, 8 -> 11.1개, 손실 0.002")
    ap.add_argument("--level-adjust", action="store_true",
                    help="영역 목표의 높이 자동 조정: 베이스라인(adjust_target_to_state_gpu)이 목표를 점토 총량이 보존되게 맞추는 것과 같은 원리. 공급(깎을 곳의 남는 점토 + 요청 영역 둘레에서 눌러 얻는 점토) 보다 수요(채울 곳의 부족분)가 크면 "
                         "부족분 목표를 같은 비율 alpha 로 줄인다(목표 = 현재 + alpha x (목표 - 현재)). 점토가 모자란 요구를 영역 밖을 파서 채우려다 구덩이가 생기는 것을 막는다")
    ap.add_argument("--level-ring-tools", type=float, default=1.2, help="--level-adjust: 요청 영역 둘레에서 눌러 얻을 수 있는 점토를 셀 폭(도구 반경의 배수). 어떤 영역에도 속하지 않은 곳만 센다")
    ap.add_argument("--level-depth-mm", type=float, default=0.5, help="--level-adjust: 둘레 화소당 눌러 얻을 수 있는 깊이(mm). 제가 정한 값")
    ap.add_argument("--squeeze", type=int, default=0,
                    help="0 보다 크면 '짜내기' 후보(SQ 계열)를 만든다(cf_transport.squeeze_candidates): 채울 곳(구덩이) 바로 바깥 둘레에서 높은 지점을 골라 중심 쪽으로 짧게 눌러 눌린 점토가 구덩이로 밀려 올라오게 한다(여드름 짜듯이). 점토를 멀리서 가져오지 않고 기존 linear_push 와 같은 행동. 값 = 빔 쿼터. 보호(건너뛴) 영역과 겹치는 지점은 뺀다")
    ap.add_argument("--t-metric", choices=["window", "blob"], default="window",
                    help="--cf-target fixed: T 까지 거리(dT)를 어디서 잴지. window = 영역 창 전체 평균(기본, 기존 동작). blob = T 를 처음 고정했을 때 |T-현재| 가 --t-blob-mm 보다 컸던 화소(= 더 올리거나 눌러야 하는 덩어리)에서만. "
                         "창 전체 평균은 푸시의 부수 변화가 덩어리 안의 이득보다 크게 잡혀 'T 에 가까워진 후보'가 거의 없었다(A groove_top: 덩어리는 창의 30%). 부수 변화는 J 허용치·영역 밖 한도·보호 영역이 따로 막는다")
    ap.add_argument("--t-blob-mm", type=float, default=0.5, help="--t-metric blob: 덩어리로 보는 |T-현재| 문턱(mm). 제가 정한 값")
    ap.add_argument("--hint-match", type=int, default=0,
                    help="0 보다 크면 '힌트 맞춤' 후보(HM 계열)를 만든다(cf_transport.hint_match_candidates). 반사실 목표 T 와 현재의 차이 D = T - 현재(얼마나 더 올리고/눌러야 하는지)가 큰 덩어리마다 "
                         "푸시 격자(위치 x 길이 x 깊이)를 만들고, Param2Deform 이 예측한 결과가 T 에 가장 가까운 것 이 개수만 Genesis 평가에 올린다. 힌트(유효한 반사실)가 있는 영역에서만 만든다. --cf-target fixed 와 같이 쓰는 것을 전제로 한다")
    ap.add_argument("--ot-donors", type=int, default=0,
                    help="0 보다 크면 점토 수송 후보(OT 계열)를 만든다(cf_transport.py): 채울 영역에 필요한 점토를 어디서 가져올지 최적 수송(선형계획)으로 정하고 그 흐름을 따라 미는 푸시. 값 = 상위 몇 개 흐름을 후보로 만들지. 빔 쿼터에 OT:값 을 자동으로 더한다")
    ap.add_argument("--ot-depth-mm", type=float, default=0.5, help="--ot-donors: 어떤 영역에도 속하지 않는 판에서 가져갈 수 있는 화소당 깊이(mm). 얕을수록 한 곳을 깊게 파지 않는다(제가 정한 값)")
    ap.add_argument("--ot-relief-weight", type=float, default=3.0, help="--ot-donors: 판 배경보다 솟은 칸(구조)에서 점토를 가져오는 비용 가중(1 + 이 값 x 솟은 정도 0~1)")
    ap.add_argument("--credit-side", type=float, default=0.0,
                    help="0 보다 크면 다중 영역 이득을 센다: 한 행동이 지금 고치는 영역 말고 요청한 다른 영역(건너뛰지 않은 영역)도 좋아지게 하면 그 이득의 합 x 이 값을 후보 점수 J 에서 뺀다(1 = 그대로 합산). 기본 0 이면 지금 고치는 영역 하나의 점수만 본다")
    ap.add_argument("--side-worsen-tol", type=float, default=0.3,
                    help="--credit-side: 요청한 다른 영역 하나의 J 가 이 값(J 단위: 기준 상태 오차 = 항당 1)보다 더 나빠지는 후보는 뺀다. 이웃 요청 영역을 크게 망가뜨리는 행동을 막는다(제가 정한 값)")
    ap.add_argument("--active-slack", type=float, default=0.05,
                    help="--credit-side: 지금 고치는 영역의 J 가 이 값보다 더 나빠지는 후보는 다른 영역 이득이 커도 뺀다(활성 영역을 희생해 다른 영역만 고치는 것을 막음)")
    ap.add_argument("--protect-skipped", action="store_true",
                    help="고치지 않기로 한 영역(--skip-regions)을 보호한다: 그 영역 안의 변화가 허용치를 넘는 후보는 평가 결과에서 한도 밖으로 본다. 주변 고리(--ring-budget-mm)와 먼 곳(--far-limit-mm)은 영역들의 합집합을 빼고 재서 그 영역 내부를 아무 제약도 안 보고 있었다")
    ap.add_argument("--protect-limit-auto", type=float, default=0.0,
                    help="0 보다 크면 보호 영역의 허용 변화(mm)를 과거 기록에서 잰다: 그 영역에 닿지 않은 과거 푸시들이 번짐으로 영역을 바꾼 최대 변화의 이 분위수. 닿지 않은 푸시가 5개 미만이면 --protect-limit-mm 를 쓴다")
    ap.add_argument("--protect-limit-mm", type=float, default=2.0, help="--protect-skipped: 자동 산정을 못 쓸 때의 보호 영역 허용 변화(mm)")
    ap.add_argument("--far-limit-auto", type=float, default=0.0,
                    help="0 보다 크면 --far-limit-mm 을 사람이 정하지 않고 이 대상의 과거 기록에서 잰다: 과거 푸시 각각이 영역에서 먼 곳을 바꾼 최대치의 이 분위수(1.0 = 과거 푸시 중 가장 큰 값). 조형 중 실제로 받아들여진 푸시보다 더 큰 부수 변화는 막는다")
    ap.add_argument("--cbo", action="store_true",
                    help="--refine-mode bads: 제약 베이지안 최적화. 평가한 후보의 먼 곳 최대 변화로 제약 모델(가우시안 과정)을 맞추고, 획득 점수에 '한도를 지킬 확률'을 곱해 한도를 넘을 만한 푸시에 평가를 쓰지 않는다")
    ap.add_argument("--avoid-protected", action="store_true",
                    help="짜내기(SQ)·힌트 맞춤(HM) 후보에서 푸시 접촉 영역이 보호(건너뛴) 영역과 2% 넘게 겹치면 평가도 안 하고 버리던 예전 사전 필터를 켠다. 기본은 끔: 보호 영역을 목표 쪽으로 움직이는 푸시를 미리 막지 않고, 실제로 눌러 본 뒤 '목표에서 멀어진 양' 한도가 가린다 (--protect-abs 를 쓰면 이 필터도 켜진다)")
    ap.add_argument("--protect-abs", action="store_true",
                    help="--protect-skipped: 보호 영역의 변화를 예전처럼 절대 변화(mm)로 잰다. 기본은 '그 영역 목표에서 멀어진 양'(화소별 max(0, |새-목표| - |현재-목표|)의 영역 안 최대)만 잰다 = 원하는 모양에 가까워지는 변화는 한도로 막지 않는다")
    ap.add_argument("--no-fewobs-relax", action="store_true",
                    help="끈다: --refine-mode bads --cbo 에서 한도 안 단일 후보가 4개 미만이면 예전처럼 패턴 탐색으로 대체한다. 기본은 대체하지 않고 한도를 넘은 후보까지 목적 대리모델에 쓰며 '한도를 지킬 확률'로 거른다(한도 안 후보가 충분하면 이 옵션과 상관없이 같은 동작)")
    ap.add_argument("--turbo", action="store_true",
                    help="--refine-mode bads: TuRBO 식 다중 국소 모델 [8]. 서로 떨어진 시작점(첫째=실측 최선, 둘째=반사실 사전이 가장 높은 관측) 마다 신뢰영역 길이를 성공/실패로 늘리고 줄이며, 시작점마다 주변 관측으로 국소 모델을 맞춘다")
    ap.add_argument("--spsa-after", type=int, default=0,
                    help="--refine-mode bads: BADS 평가가 끝난 뒤 SPSA [7] 를 이어서 이 개수만큼 더 평가한다(총 평가 = --bads-evals + 이 값). 0 이면 쓰지 않는다")
    ap.add_argument("--pair-adapt", action="store_true",
                    help="순서쌍 예산을 라운드마다 조절한다(메모 3쪽): 단일 최선의 상대 개선이 25%% 이상이면 1x2, 10%% 이상이면 2x3, 그보다 작으면 --pairs x --pair-seconds. 단일 개선이 좋아도 최소 1x2 는 시험해 놓치는 조합을 추정한다")
    ap.add_argument("--strict", action="store_true",
                    help="요청한 기능(적응형 정제, ②T 트리, ②S 묶음 탐색)이 실패하면 건너뛰고 계속하지 않고 예외로 실행을 실패시킨다. 비교 실험에서 기능이 조용히 빠지는 것을 막는다")
    ap.add_argument("--seed", type=int, default=0,
                    help="적응형 정제(bads)의 제안 난수 시드 오프셋(0=기존과 같음). 같은 조건을 시드만 바꿔 반복해 분산을 잴 때 쓴다")
    ap.add_argument("--seq-k", type=int, default=2, help="--long-select seq: 한 묶음에서 바꿀 과거 행동 수 상한(2 이상). cf_seq_cf.py (Tsirtsis & Gomez-Rodriguez 2023 [10] 의 구조)")
    ap.add_argument("--seq-pool", type=int, default=6,
                    help="--long-select seq: 영역에 영향이 큰 과거 행동 상위 몇 개를 후보로 삼을지. 과거 전체(#1~#K)에서 고르고 재생 범위(--replay-len) 제한을 받지 않는다")
    ap.add_argument("--seq-budget", type=int, default=8, help="--long-select seq: 묶음 재생 횟수 상한(단일 개입 재생은 따로). --seq-budget-min 이 0 보다 크면 시간이 우선")
    ap.add_argument("--seq-budget-min", type=float, default=0.0,
                    help="--long-select seq: 묶음 재생에 쓸 시간 예산(분). 묶음 하나의 비용 = 재생 행동 수 x --replay-sec-per-action (워커 1개 기준). 0 이면 --seq-budget 을 쓴다")
    ap.add_argument("--own-score", action="store_true",
                    help="영역 점수를 따로 계산해서 쓴다: 요청의 영역 여러 개를 한 번에 넣되, 라운드마다 보상하는 영역의 후보를 '그 영역 하나의 점수'로 평가·확정한다(다른 영역 점수는 확정 판단에 넣지 않는다). "
                         "확정한 행동은 같은 점토에 계속 쌓여 영역별 보상이 한 결과로 합쳐진다. 기본(끔)은 모든 영역 점수의 평균 J 로 확정한다")
    ap.add_argument("--cf-target", choices=["off", "fixed"], default="off",
                    help="반사실 세계 T 를 목표로 삼아 여러 번 눌러 접근한다(T-추적). off=예전(T 는 후보 고르는 보조이고 매 라운드 현재 + C 로 다시 계산해 이미 판 만큼 T 가 더 깊어지는 이중 계산이 있다). "
                         "fixed=T 를 처음 한 번만 고정(현재 + C, 창 안)하고, ① T 방향 후보(TRK: T 와 현재 차이로 누르기·옮기기)를 더하고 ② 계획기 목표를 T 로 바꾸고 "
                         "③ '고정 T 까지의 거리가 줄어드는' 후보만 확정한다(요청 기준 J 는 --t-slack 안에서 유지). 확정·가지치기는 모두 Genesis 실행 결과로만 한다")
    ap.add_argument("--trk-thr-mm", type=float, default=0.6,
                    help="--cf-target fixed: T 방향 후보(TRK)를 만드는 최소 높이 차(mm). 기존 점토 옮기기/평탄화 후보의 기본값 2 mm 는 T 가 현재보다 1 mm 안팎만 깊을 때 후보가 하나도 안 나온다")
    ap.add_argument("--trk-blobs", type=int, default=3, help="--cf-target fixed: T 와의 차이가 큰 덩어리를 몇 개까지 후보로 만들지")
    ap.add_argument("--eps-t", type=float, default=0.005, help="--cf-target fixed: 확정하려면 고정 T 까지의 창 안 평균 거리가 이만큼(mm) 이상 줄어야 한다")
    ap.add_argument("--t-jband", type=float, default=0.0,
                    help="--cf-target fixed: 0 보다 크면 확정 규칙을 'T 에 가장 가까운 것'에서 'J 가 (T 에 다가가고 한도를 지키는 후보 중) 최선에서 이 값 이내인 후보들 중 T 에 가장 가까운 것'으로 바꾼다. "
                         "T 만 따르면 T 가 약할 때(영역 점수 개선 약 0.14) 목표 쪽으로 거의 안 간다(A@80 fill2: J 2.61 후보 대신 J 2.98 후보가 확정됨). T 에 다가가는 후보가 없으면 J 최선 후보를 확정한다. 0 = 예전 동작")
    ap.add_argument("--commit-by-j", action="store_true",
                    help="--cf-target fixed: 반사실 T 는 후보 생성(CFR, TRK), 계획기 목표, 정제 사전에만 쓰고 확정은 T-추적 없이 J 로만 한다(--w-cf 0 이면 J 최선, 2-step 쌍은 J 가 더 좋을 때만). "
                         "T 가 확정에 끼어들어 J 를 양보하게 하는 효과와 T 힌트가 후보를 좋게 하는 효과를 가르는 A/B 의 C 팔. 기본은 끔(기존 명령은 그대로)")
    ap.add_argument("--t-slack", type=float, default=0.10, help="--cf-target fixed: 확정 후보의 요청 기준 J 가 현재보다 이 값까지는 나빠져도 허용(데드락 방지). J 시작값은 가중치 합(3.0)")
    ap.add_argument("--patience", type=int, default=0,
                    help="속도 옵션(0=끔): 1단계 후보 평가를 앞쪽(과거 힌트 계열)부터 하다가 최선이 이 개수 평가 동안 개선되지 않으면 남은 후보를 생략한다. 평가가 22~52%로 줄지만 라운드당 J 를 약 0.15 잃는다(A 기록 시뮬레이션). "
                         "--refine-top 1 과 같이 쓰면 주변 탐색도 줄어든다")
    ap.add_argument("--per-region-commits", type=int, default=0,
                    help="영역 하나에 확정할 수 있는 최대 횟수(0=제한 없음). 오차가 큰 영역이 확정을 독점하지 않게 한다(예전 보고서: 오른쪽 받침만 세 라운드)")
    ap.add_argument("--stop-progress", type=float, default=0.0,
                    help="영역의 목표 진행률(영역 평균 높이 변화 / 영역 목표가 요구한 변화)이 이 값 이상이면 그 영역은 더 확정하지 않는다(0=끔). 파기 목표를 넘어 파는 것을 막는다")
    ap.add_argument("--long-screen", choices=["off", "model"], default="off",
                    help="--long-select seq|both 의 ②S 에서: model = 영역 기준으로 뽑은 관련 과거 행동 전부 x 모든 변형(skip, 길이·깊이 배율) x 묶음(최대 --seq-k 개)을 Param2Deform 롤아웃으로 먼저 훑고(후보당 몇 초), "
                         "모델 예측 영역 점수 이득이 큰 상위 --screen-verify 개만 Genesis 재생으로 검증한다. 모델은 순위만 정하고 반사실 효과·확정은 Genesis 결과로만 한다. off = 예전 동작(상위 --seq-pool 개를 Genesis 로 하나씩 잼)")
    ap.add_argument("--screen-actions", type=int, default=24, help="--long-screen model: 모델로 훑을 관련 과거 행동 수(영향 순 상위)")
    ap.add_argument("--screen-pairs", type=int, default=8, help="--long-screen model: 묶음(2개 이상)을 만들 단일 후보 풀 크기(모델 예측 이득 상위)")
    ap.add_argument("--ctrl-valid-len", type=int, default=0,
                    help="--control record: 이 캐시 폴더에서 아직 검증한 control 이 없어도, 기록을 control 로 써도 되는 재생 길이(행동 수)를 이만큼으로 본다(--control-extrap 배까지 외삽). "
                         "체인에서는 K 가 한 단계 커질 때마다 캐시 폴더가 새로 생겨 검증된 길이가 0 이 되고, 그러면 재생마다 control 재생을 따로 돌려 시간이 약 1/4 늘어난다. 앞 단계에서 검증한 길이(A@80: 79)를 이어받는 용도. 0 = 예전 동작")
    ap.add_argument("--screen-max-len", type=int, default=0,
                    help="--long-screen model: 재생 길이(행동 수)가 이 값을 넘는 후보는 Genesis 검증에서 제외한다(1위 포함). 0 = 제한 없음(예전 동작). 모델 예측으로는 계속 훑는다")
    ap.add_argument("--screen-budget-actions", type=int, default=0,
                    help="--long-screen model: Genesis 검증 후보들의 재생 행동 수 합 상한. 모델 예측 이득 1위는 길이와 상관없이 항상 검증하고, 그 뒤로는 예산 안에 드는 후보만(예산을 넘는 긴 재생은 건너뛰고 더 짧은 후보를 찾는다). 0 = 예산 없음(예전 동작)")
    ap.add_argument("--screen-verify", type=int, default=6, help="--long-screen model: Genesis 로 검증할 상위 후보 수의 상한(시간 예산 --seq-budget-min 안에서)")
    ap.add_argument("--min-commit-gain", type=float, default=-1e9,
                    help="확정하려면 후보의 (예측) J 이득 = 현재 J - 후보 J 가 이 값 이상이어야 한다. 기본(-1e9)은 끔(예전 동작: T 모드에서 --t-slack 만큼 J 가 나빠지는 후보도 T 에 가까우면 확정). "
                         "2026-11-17 A 5단계에서 J 예측 2.922(현재 2.902)인 후보가 확정돼 84분을 쓰고 영역 MAE 가 오히려 나빠진 뒤 추가. 이득이 모자라면 그 영역은 확정하지 않고 멈춤 처리한다(다른 영역을 먼저 시도)")
    ap.add_argument("--screen-model-only", action="store_true",
                    help="--long-screen model: Genesis 검증 재생 없이 모델 반사실(C = 모델(변형) - 모델(원래))만으로 T 를 만든다(상위 1개). 재생 시간이 사라져 단계가 약 1/3 로 줄지만 모델 예측이 그대로 쓰인다. "
                         "--long-select seq --cf-top 0 과 함께 써서 ② 의 Genesis 재생을 모두 끈다. A/B 의 M 팔")
    ap.add_argument("--screen-variants", default="",
                    help="--long-screen model 에서만 쓰는 변형 목록(쉼표 구분, --cf-variants 와 같은 문법 + sh<mm> 옆 이동, al<mm> 앞 이동, rot<도> 방향 회전). 예) skip,1:0.5,1:1.5,1.3:1,sh-10,sh-5,sh+5,sh+10,al-8,al+8,rot-45,rot-20,rot+20,rot+45. 비우면 --cf-variants(-more) 를 그대로 쓴다")
    ap.add_argument("--screen-objective", choices=["J", "mae"], default="J",
                    help="--long-screen model 의 모델 예측 이득과 Genesis 검증 후보 선택 기준. J = 영역 점수 J 이득(예전 동작). mae = 영역 MAE 이득(mm) + --screen-j-weight x J 이득 (모양이 깨지는 편집을 막는다). 확정 후보 자격은 어느 쪽이든 J 이득 > --eps")
    ap.add_argument("--screen-j-weight", type=float, default=1.0, help="--screen-objective mae 에서 J 이득에 곱하는 가중치")
    ap.add_argument("--screen-greedy", type=int, default=0,
                    help="--long-screen model: 2 보다 크면 단일 최선에서 시작해 편집을 하나씩 탐욕적으로 더해 최대 이 개수까지 묶음을 만든다(모델 예측만 쓰고 비용은 후보당 수 초). Genesis 검증은 크기별로 가장 큰 묶음, 가운데 크기, 앞 3개 묶음(--screen-verify 로 상한). 0 = 쓰지 않음")
    ap.add_argument("--screen-greedy-min", type=float, default=0.02, help="--screen-greedy: 편집 하나를 더했을 때 목적값이 이만큼 이상 늘지 않으면 멈춘다")
    ap.add_argument("--refine-t-weight", type=float, default=0.0,
                    help="0 보다 크면 적응형 정제(--refine-mode bads/spsa, TuRBO 포함)의 목적값을 J 가 아니라 J + 가중치 x dT 로 쓴다(dT = 후보 푸시 뒤 높이맵과 반사실 목표 T 사이 창 평균 거리, mm). 반사실 T 가 없으면 효과 없음. 1 이면 J 띠 0.1 과 비슷한 비중(dT 0.1 mm = J 0.1). 0 = J 만(예전 동작)")
    ap.add_argument("--cf-top1", action="store_true",
                    help="반사실 단계(②, ②T, ②S, ②L)를 오차가 가장 큰 영역 하나에만 한다(--max-commits 1 로 단계마다 확정 1번만 하는 체인용: 어차피 그 영역만 고치므로 나머지 영역의 재생에 시간을 쓰지 않는다)")
    ap.add_argument("--plan-refine", type=int, default=0,
                    help="0 보다 크면 계획기 제안(PLAN/PLANSEQ)을 greedy 로 한 번 뽑고 끝내지 않고, vs_planner.refine_plan_gpu 로 이 횟수만큼 모델 기울기(Param2Deform 롤아웃 전체)로 다듬는다. "
                         "목표 맵 = 현재 맵에서 창(반사실 목표 T 가 있으면 T, 없으면 고치는 영역의 목표)만 바꾼 것이라, 창 밖·보호 영역을 건드리는 푸시는 모델 손실에서 이미 불리하다. "
                         "다듬은 연쇄는 --plan-seq 로 Genesis 에서 묶음째 평가한다. 확정은 항상 Genesis 실행 결과로만 한다(0=끔, 기존 동작)")
    ap.add_argument("--plan-seq", type=int, default=0,
                    help="계획기 시퀀스 제안(PLANSEQ): 계획기가 요청 영역 목표를 향해 짠 계획의 앞 K개 행동(2 이상)을 한 묶음으로 Genesis 에서 실행해 보고, 묶음 전체가 J 를 개선하면 확정한다 (0=끔). "
                         "--plan-cands 회 계획을 뽑는다. 단일 후보·2-step 으로는 못 닿는 여러 번 눌러야 하는 변화(깊은 홈)를 위한 것. 확정은 Genesis 실행 결과로만 한다")
    ap.add_argument("--min-reach", type=float, default=0.5,
                    help="영역 화소 중 도구가 닿을 수 있는 비율(행동 범위 상자 + 도구 반경)이 이보다 낮은 영역은 후보가 닿지 못하므로 빼고 알린다. 0 이면 끔")
    a = ap.parse_args()
    if a.no_hints:
        a.cf_replay, a.long_top, a.long_select, a.lin_cf, a.cf_target, a.long_group, a.far_top, a.rel_replay = False, 0, "rank", 0, "off", 0, 0, 0
    if a.squeeze > 0 and "SQ" not in a.quota:
        a.quota += f",SQ:{a.squeeze}"
    if a.hint_match > 0 and "HM" not in a.quota:
        a.quota += f",HM:{a.hint_match}"            # 힌트 맞춤 후보 계열(빔 쿼터에 없으면 물리 평가 대상에서 빠진다)
    if a.ot_donors > 0 and "OT" not in a.quota:
        a.quota += f",OT:{a.ot_donors}"             # 수송 후보 계열(빔 쿼터에 없으면 물리 평가 대상에서 빠진다)
    if a.cf_target == "fixed" and "TRK" not in a.quota:
        a.quota += ",TRK:8"                         # T 방향 후보 계열(빔 쿼터에 없으면 물리 평가 대상에서 빠진다)
    global BOUNDS
    t_start = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frames").mkdir(exist_ok=True)
    (a.out / "states").mkdir(exist_ok=True)
    import cf_common as cc
    from region_set import RegionSet
    import first_beam as fb
    root = cc.make_view(a.history_dir, a.upto, a.out / "history_view")
    man, actions, before, after = fb.load_history(root)
    g = cc.geometry(root)
    geo = Geo(g)
    r_tool = g["tool_r_px"]
    near_px, win_px = int(round(a.near_tools * r_tool)), int(round(a.win_tools * r_tool))
    ring_px, far_px = int(round(a.ring_tools * r_tool)), int(round(a.far_tools * r_tool))
    BOUNDS = cc.action_bounds(a.depth_min_mm)
    rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_h=a.ref_h, ref_default=cc.state_height(root))
    # 도달 가능성: 대상이 바뀌면 영역이 판 가장자리(행동 범위 밖)에 놓일 수 있다. 어떤 후보도 닿지 못하는 영역은 라운드 예산만 쓴다.
    rmask = cc.reach_mask(g, BOUNDS)
    reach = {n: cc.reach_fraction(rs.masks[n], g, rmask=rmask) for n in rs.names}
    print("영역별 도달 가능 비율(도구 중심이 행동 범위 안일 때 도구 반경 이내인 화소): " + ", ".join(f"{n} {100 * reach[n]:.0f}%" for n in rs.names), flush=True)
    unreachable = [n for n in rs.names if a.min_reach > 0 and reach[n] < a.min_reach]
    if unreachable:
        print(f"[도달 불가] 영역 {unreachable} 은(는) 도달 비율이 {100 * a.min_reach:.0f}% 미만이라 뺍니다. "
              f"행동 범위 x[{BOUNDS[0] * 1000:.0f},{BOUNDS[1] * 1000:.0f}] y[{BOUNDS[2] * 1000:.0f},{BOUNDS[3] * 1000:.0f}] mm 밖입니다 (--min-reach 0 이면 유지).", flush=True)
        keep = [n for n in rs.names if n not in unreachable]
        if not keep:
            raise SystemExit("보상할 수 있는 영역이 없습니다(모든 영역이 행동 범위 밖).")
        rs = rs.subset(keep)
    names = rs.names
    label = rs.labels
    h_ref = rs.h_ref
    level_info = {}
    if a.level_adjust:      # 영역 목표의 높이 자동 조정(점토 총량 보존, 베이스라인과 같은 원리): 달성할 수 없는 양을 요구하지 않는다
        import cf_transport as ctp
        skip_la = {x for x in a.skip_regions.split(",") if x} & set(names)
        adj_t, level_info = ctp.level_adjust_region_targets(cc.state_height(root).astype(np.float64), names, rs.masks, rs.targets, [n_ for n_ in names if n_ not in skip_la], g["px_mm"], g["tool_r_px"],
                                                            ring_tools=a.level_ring_tools, depth_mm=a.level_depth_mm)
        (a.out / "level_adjust").mkdir(exist_ok=True)
        for n_ in level_info.get("regions", []):
            np.save(a.out / "level_adjust" / f"{n_}_target_original.npy", np.asarray(rs.targets[n_], dtype=np.float32))
            rs.targets[n_] = adj_t[n_]
            np.save(a.out / "level_adjust" / f"{n_}_target_adjusted.npy", np.asarray(adj_t[n_], dtype=np.float32))
        print(f"[높이 자동 조정] 요청 영역 {len(level_info.get('regions', []))}개: 수요 {level_info['need_mm3']:,} mm3, 공급 {level_info['supply_mm3']:,} mm3 (깎을 곳 {level_info['cut_mm3']:,} + 둘레 {level_info['ring_mm3']:,}) -> alpha = {level_info['alpha']:g}"
              + (" (목표를 줄임)" if level_info["alpha"] < 1.0 else " (변화 없음)"), flush=True)
    fp, tgt_focus = None, None
    if a.focus_tools > 0:
        tgt_focus = cc.load_target(root, a.target_npy).astype(np.float64)
        try:
            fp = cc.relief_focus(tgt_focus, r_tool, near_tools=a.focus_tools, base=(a.base_mm * 1e-3 if a.base_mm is not None else cc.plate_level(root)))
        except ValueError as e:
            print(f"[초점 없음] {e} -> 예전처럼 영역 주변만 본다", flush=True)
    if fp is not None:
        print(f"초점(목표의 모양 + 도구 반경 {a.focus_tools:g}배 근처): 판 높이 {fp['base'] * 1000:.1f}mm, 모양 {int(fp['relief'].sum())}px, 초점 {int(fp['focus'].sum())}px = 맵의 {100 * fp['focus'].mean():.0f}% "
              f"| 배경(초점 밖)은 점수·제약에서 제외, 글자 안의 다른 부분이 망가지는 것만 막는다", flush=True)
        for n in list(rs.names):
            m0 = rs.masks[n]
            m1 = m0 & fp["focus"]
            if m1.sum() < 0.3 * m0.sum():
                print(f"  [주의] 영역 '{n}' 의 {100 * (1 - m1.sum() / max(m0.sum(), 1)):.0f}% 가 초점 밖이라 마스크를 자르지 않고 둔다", flush=True)
            else:
                if m1.sum() < m0.sum():
                    print(f"  영역 '{n}': 초점 밖 {int(m0.sum() - m1.sum())}px 를 뺌", flush=True)
                rs.masks[n] = m1
    rb_all, rb_by = None, {}
    if a.ring_budget_mm is not None:
        a.mu = 0.0                                  # 주변 변화는 가중치가 아니라 한도(--ring-budget-mm)로 제한한다
        txt = str(a.ring_budget_mm)
        if Path(txt).is_file():                     # conditions.py 출력: 영역별 ring_budget_mm
            for s_ in json.loads(Path(txt).read_text()).get("spec", []):
                if s_.get("ring_budget_mm") is not None and s_["region"] in names:
                    rb_by[s_["region"]] = float(s_["ring_budget_mm"])
        elif "=" in txt:
            for tok in [t.strip() for t in txt.split(",") if t.strip()]:
                k_, v_ = tok.split("=")
                rb_by[k_.strip()] = float(v_)
            star = rb_by.pop("*", None)
            for k_ in [k for k in rb_by if k not in names]:
                print(f"  (경고: --ring-budget-mm 의 영역 '{k_}' 은(는) 이번 영역 집합에 없어 무시: {names})", flush=True)
                rb_by.pop(k_)
            if star is not None:
                for n_ in names:
                    rb_by.setdefault(n_, star)
        else:
            rb_all = float(txt)
        if rb_all is not None:
            print(f"주변 변화는 가중치 없이 한도 {rb_all:g} mm(모든 영역을 합친 주변 고리의 평균 변화)로 제한한다", flush=True)
        else:
            print("주변 변화는 가중치 없이 영역별 한도(mm)로 제한한다: " + ", ".join(f"{k_} {v_:g}" for k_, v_ in rb_by.items()) + " (후보는 모든 한도를 동시에 지켜야 통과)", flush=True)

    prot_lim = {}      # --protect-skipped: 보호 영역 -> 허용 변화(mm). fps 가 만들어진 뒤에 채운다

    def violation(r):
        """제약 위반 정도 = max(먼 곳 변화 / 한도, 보호 영역 변화 / 그 영역 한도). 1 이하면 한도를 지킨다(제약 베이지안 최적화의 제약값)."""
        v = r["far_max"] / max(a.far_limit_mm, 1e-9)
        for n_, d_ in r.get("prot", {}).items():
            v = max(v, d_ / max(prot_lim.get(n_, a.protect_limit_mm), 1e-9))
        if a.credit_side > 0:
            v = max(v, r.get("side_worse", 0.0) / max(a.side_worsen_tol, 1e-9), r.get("active_worse", 0.0) / max(a.active_slack, 1e-9))
        return v

    def within(r):
        if r["far_max"] > a.far_limit_mm:
            return False
        for n_, d_ in r.get("prot", {}).items():      # --protect-skipped: 고치지 않기로 한 영역은 허용 변화 이내여야 한다
            if d_ > prot_lim.get(n_, a.protect_limit_mm):
                return False
        if a.credit_side > 0 and (r.get("side_worse", 0.0) > a.side_worsen_tol or r.get("active_worse", 0.0) > a.active_slack):
            return False      # 요청한 다른 영역을 크게 망가뜨리거나, 활성 영역을 희생하는 행동
        if rb_all is not None and r.get("coll", 0.0) > rb_all:
            return False
        cr_ = r.get("coll_r", {})
        return all(cr_.get(k_, 0.0) <= v_ for k_, v_ in rb_by.items())
    cfg = build_cfg(rs, h_ref, a.mu, ring_px, far_px, r_tool, (a.w_height, a.w_grad, a.w_edge), a.blur_tools, fp=fp, target=tgt_focus, w_focus=a.w_focus if fp is not None else 0.0, w_tail=a.w_tail)
    cfg["save_dir"] = str(a.save_transitions) if a.save_transitions else None
    if a.far_limit_auto > 0 and cfg["far"].any():      # 한도를 과거 기록에서 잰다 (cf_common.far_tolerance: 대상에 의존하지 않는다)
        a.far_limit_mm, far_hist = cc.far_tolerance(before, after, cfg["far"], a.far_limit_auto)
        print(f"[한도] --far-limit-auto {a.far_limit_auto:g}: 과거 {len(far_hist)}푸시가 먼 곳을 바꾼 최대치(중앙값 {np.median(far_hist):.1f}, 90% {np.quantile(far_hist, 0.9):.1f}, 최대 {far_hist.max():.1f}mm)의 분위수 -> --far-limit-mm = {a.far_limit_mm:.1f} mm", flush=True)
    # 같은 점수식을 '원래 목표' 기준으로도 잰다(보고용). 요청과 원래 목표가 반대 방향인 영역(예: 더 파기 vs 구멍 메우기)이 한눈에 드러난다.
    cfg_orig = None
    try:
        goals_o = cc.original_target_goals(rs, root, h_ref, a.target_npy)
        cfg_orig = build_cfg(RegionSet(rs.names, rs.masks, goals_o, rs.labels, h_ref, rs.meta), h_ref, 0.0, ring_px, far_px, r_tool, (a.w_height, a.w_grad, a.w_edge), a.blur_tools)
    except Exception as e_o:
        print(f"(원래 목표 기준 점수는 계산하지 못함: {e_o!r})", flush=True)
    print(f"형태·외형 오차는 반경 {r_tool:.1f}px 도구의 {a.blur_tools:g}배 = 흐림 {cfg['sigma']:.1f}px (점토 입자 질감 제외)에서 잰다", flush=True)
    masks = {n: m for n, m, _ in cfg["regs"]}
    tgts = {n: t for n, _, t in cfg["regs"]}
    print(f"이력 {Path(a.history_dir).name}: 과거 행동 {len(actions)}개 (조형 {len(actions)}번째 상태를 현재로 삼음) | 영역 {len(names)}개: {', '.join(names)}", flush=True)
    print(f"도구 반경 {r_tool:.1f}px ({g['tool_r'] * 1000:.0f}mm) -> 관련 행동 거리 {near_px}px, 가중치 창 {win_px}px, 주변 고리 {ring_px}px, 밖 {far_px}px", flush=True)
    print("관련 행동을 만들기 위해 모든 과거 행동의 접촉영역을 계산 중 ...", flush=True)
    fps = [fb.footprint(geo.g, actions[t]) for t in range(len(actions))]
    if a.protect_skipped:      # 고치지 않기로 한 영역(--skip-regions) 보호: 어떤 제약도 그 영역 안의 변화를 안 보고 있었다(고리·먼 곳은 영역 합집합 밖만 잰다)
        skip_prot = [n for n in names if n in {x for x in a.skip_regions.split(",") if x}]
        if skip_prot:
            cfg["protect"] = {n: rs.masks[n] for n in skip_prot}
            if not a.protect_abs:      # 보호 영역의 변화를 그 영역 목표 기준으로 잰다(목표에 가까워지는 변화는 막지 않는다). 목표는 영역 폴더의 <영역>_target.npy
                cfg["protect_t"] = {n: np.asarray(rs.targets[n], dtype=np.float64) for n in skip_prot}
                print("[보호] 변화 측정: 영역 목표에서 멀어진 양만 센다(목표 쪽으로 가는 변화는 한도로 막지 않는다). 예전 방식 = --protect-abs", flush=True)
            for n in skip_prot:
                lim = None
                if a.protect_limit_auto > 0:
                    lim, pv = cc.protect_tolerance(before, after, fps, rs.masks[n], near_px, a.protect_limit_auto)
                    if lim is None:
                        print(f"[보호] '{n}': 그 영역에 닿지 않은 과거 푸시가 {len(pv)}개뿐이라 자동 산정 불가 -> --protect-limit-mm {a.protect_limit_mm:g}", flush=True)
                    else:
                        print(f"[보호] '{n}': 닿지 않은 과거 푸시 {len(pv)}개가 영역을 바꾼 최대치(중앙값 {np.median(pv):.2f}, 90% {np.quantile(pv, 0.9):.2f}, 최대 {pv.max():.2f}mm)의 {a.protect_limit_auto:g} 분위수 -> 허용 {lim:.2f} mm", flush=True)
                prot_lim[n] = float(lim) if lim is not None else float(a.protect_limit_mm)
            print("[보호] 고치지 않는 영역의 허용 변화(mm): " + ", ".join(f"{k} {v:.2f}" for k, v in prot_lim.items()), flush=True)
    det = man["signature"]["details"]
    start_state = a.start_state or (root / f"action_{len(actions):06d}" / "post_state.pt")
    ctx = mp.get_context("spawn")
    rq = ctx.Queue()
    tqs, procs = {}, {}
    n_total = a.workers + 1
    display = os.environ.get("DISPLAY")
    cfg["pair_branch"] = bool(a.pair_branch)             # 워커가 같은 첫 푸시로 시작하는 순서쌍의 첫 푸시 직후 상태를 재사용한다
    for wid in range(n_total):
        tqs[wid] = ctx.Queue()
        p = ctx.Process(target=worker_main, args=(wid, tqs[wid], rq, cfg, str(start_state), bool(a.viewer and wid == 0), display,
                                                    det["material"], bool(det["tip_correction"]), man["signature"] if wid == 0 else None), daemon=True)
        p.start()
        procs[wid] = p
    h_cur = None
    ready = 0
    while ready < n_total:
        m = rq.get(timeout=1800)
        if m[0] == "error":
            raise RuntimeError(m[2])
        if m[0] == "ready":
            ready += 1
            if m[1] == 0:
                h_cur = m[2].astype(np.float64)
    print(f"워커 {n_total}개 준비 완료 ({time.time() - t_start:.0f}초)", flush=True)
    eval_workers = list(range(0 if (not a.viewer or a.viewer_eval) else 1, n_total))      # 후보 평가에 참여하는 워커 (창 워커 0 포함 여부)
    replay_workers = [w for w in eval_workers if w != 0] if a.viewer else eval_workers      # 20개 길이 재생은 창 없는 워커만(창 렌더링으로 느려져 전체 시간이 늘어나는 것을 피함)
    model = None
    if a.use_model and Path(a.checkpoint).exists():
        import torch
        from vs_dynamics import Param2Deform
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = Param2Deform()
        model.load_state_dict(ck["model"])
        del ck
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        print(f"모델 기반 제안(MDL): {a.checkpoint.name} 로드", flush=True)
    elif a.use_model:
        print(f"(체크포인트가 없어 모델 기반 제안(MDL)은 건너뜀: {a.checkpoint})", flush=True)

    credit_state = dict(side=[], base={}, J0=0.0); eval_seq, stage_sec, stage_cnt = [0], {}, {}; rho_info, feat_used, pair_budget_r = {}, {}, [None]      # 순위 상관(메모 [6]), 실제로 쓰인 기능, 순서쌍 예산도 라운드마다 기록한다      # 라운드마다 비운다: 물리 평가가 도착한 순서(seq)와 단계별 평가 시간·개수. 같은 예산 비교(평가 n 개 시점의 최선 J)와 시간표용

    def run_eval(items, chunk=2, tgt=None, patience=0, base_J=None, gain=0.02, need=0.05, best0=None, stage=""):
        """후보를 워커들에 나눠 Genesis 로 평가한다. patience > 0 이면 조기 종료: 후보는 앞쪽(과거 힌트 계열: G1, CFR 등이 먼저)부터 평가하고, 한도 안(within) 최선 J 가
        현재(base_J)보다 0.05 이상 좋아진 뒤 최선이 --patience 개 평가 동안 0.02 이상 좋아지지 않으면 남은 후보를 평가하지 않는다.
        (A 의 6개 라운드 309개 평가 기록으로 시뮬레이션: 인내 6 -> 평가 22%, 인내 10 -> 33%, 인내 16 -> 52%, 최선 J 대비 평균 손실 +0.14~0.16)"""
        if not items:
            return []
        pending = [items[i:i + chunk] for i in range(0, len(items), chunk)]
        results, inflight, cid, t0 = [], set(), 0, time.time()
        best_j, since, skipped = (1e9 if best0 is None else best0), 0, 0       # best0: 이미 알고 있는 최선 J(패턴 탐색은 빔 단계 최선에서 시작)
        for w in eval_workers:
            if pending:
                tqs[w].put(("eval", cid, pending.pop(0), tgt))
                inflight.add(w)
                cid += 1
        while inflight:
            m = rq.get(timeout=3600)
            if m[0] != "eval":
                continue
            for r_ in m[3]:
                eval_seq[0] += 1
                r_["seq"], r_["stage"] = eval_seq[0], stage
                if a.credit_side > 0 and credit_state["side"]:      # 요청한 다른 영역의 이득도 후보 점수에 반영
                    credit_adjust(r_, credit_state["base"], credit_state["side"], cfg, a.credit_side, credit_state["J0"])
            results += m[3]
            if patience > 0 and base_J is not None:
                for r_ in m[3]:
                    if r_["J"] < 1e8 and within(r_):
                        if r_["J"] < best_j - gain:
                            best_j, since = r_["J"], 0
                        else:
                            since += 1
                    else:
                        since += 1
                if pending and best_j < base_J - need and since >= patience:
                    skipped = sum(len(c_) for c_ in pending)
                    pending.clear()
            if pending:
                tqs[m[1]].put(("eval", cid, pending.pop(0), tgt))
                cid += 1
            else:
                inflight.discard(m[1])
        dt = time.time() - t0
        n_done = len(results)
        stage_sec[stage] = stage_sec.get(stage, 0.0) + dt
        stage_cnt[stage] = stage_cnt.get(stage, 0) + n_done
        print(f"      후보 {n_done}개 물리 평가 {dt:.0f}초 ({dt / max(n_done, 1):.1f}초/개)" + (f" | 조기 종료: 최근 {since}개 평가에서 개선 없음(최선 J {best_j:.3f}) -> 남은 {skipped}개 생략" if skipped else ""), flush=True)
        return results

    bg = float(np.median(np.concatenate([h_ref[0], h_ref[-1], h_ref[:, 0], h_ref[:, -1]])))
    ys, xs = np.where(cfg["union"])
    crop = (max(ys.min() - 40, 0), min(ys.max() + 40, SIZE), max(xs.min() - 40, 0), min(xs.max() + 40, SIZE))
    hstar_all = h_ref.copy()
    for n in names:
        hstar_all[masks[n]] = tgts[n][masks[n]]
    s_cur = score(h_cur, cfg, h_cur)
    s_ref = score(h_ref, cfg, h_ref)
    fin = a.out / "frames"
    make_frame(fin / "frame_000.png", h_cur, cfg, hstar_all, bg, crop, f"start: J={s_cur['J']:.3f}  (reference state J={s_ref['J']:.3f})")
    np.save(a.out / "states" / "iter_000.npy", h_cur.astype(np.float32))
    fmt = lambda s: " | ".join(f"{n} 형태{s['vis'][n][2]:.2f} 기울기{s['vis'][n][0]:.2f} 테두리{s['vis'][n][1]:.2f}" for n in names) + (
        f" | [글자+근처 형태{s['focus']['shape']:.2f} 기울기{s['focus']['grad']:.2f} 테두리{s['focus']['edge']:.2f}]" if "focus" in s else "")
    print(f"시작 J={s_cur['J']:.3f} (영역 오차 평균 {s_cur['e']:.3f}, 주변 변화 {s_cur['coll']:.2f}) | 영역별 [형태/기울기/테두리 오차, 낮을수록 좋음]: {fmt(s_cur)}", flush=True)
    # ================================================================ ② 과거 반사실 재생 (Genesis)
    # 영역마다 재생 범위(현재 직전 replay_len 개 행동) 안의 관련 과거 행동을 골라, 그 행동을 바꿔서 현재까지 다시 실행해 본다.
    #   C* = H_cf - H_ctrl  ("그때 달리 했다면 지금 점토가 이만큼 달랐을 것")
    #   control(H_ctrl)이 기록된 현재와 맞아야(신뢰 기준) 쓴다. q = J(현재) - J(현재 + C*) > 0 이면 그 반사실은 목표에 가까워진다.
    from edit_spec import soft_mask
    K = len(actions)
    cf_best, cf_table, wsoft, wins = {n: None for n in names}, [], {}, {}
    for n in names:
        wins[n] = ndimage.binary_dilation(masks[n], iterations=win_px)
        wsoft[n] = soft_mask(wins[n], 4.0).astype(np.float64)
    t_replay0 = time.time()
    skip_cf = {x for x in a.skip_regions.split(",") if x} & set(names)
    cf_names = [n for n in names if n not in skip_cf]      # --skip-regions 영역은 과거 재생(②, ②L, ②T, ②S)도 하지 않는다: 고치지 않을 영역의 재생에 시간을 쓰지 않는다(점수는 계속 기록)
    if a.cf_top1 and len(cf_names) > 1:      # 라운드 1 에서 고칠 영역 = 오차(0.5*(MAE+RMSE))가 가장 큰 영역(라운드 루프의 선택 규칙과 같다). 그 영역만 반사실을 계산한다
        _per0, _, _ = _terms(h_cur, cfg)
        _top = max(cf_names, key=lambda k_: 0.5 * sum(_per0[k_]))
        print(f"[② 반사실] --cf-top1: 오차가 가장 큰 영역 '{_top}' 만 반사실을 계산한다(나머지 {len(cf_names) - 1}개 영역은 건너뜀)", flush=True)
        cf_names = [_top]
    if a.cf_replay:
        lo = max(1, K - a.replay_len + 1)
        cf_less, cf_more = parse_cf_variants(a.cf_variants), parse_cf_variants(a.cf_variants_more)
        vars_for = (lambda w: (cf_more + cf_less) if w > 0 else (cf_less + cf_more)) if a.cf_both_dirs else (lambda w: cf_more if w > 0 else cf_less)      # 반사실 방향은 과거 행동 가중치 w 의 부호를 따른다(--cf-both-dirs: 반대 방향도 뒤에 붙임)
        pick = {}
        for n in cf_names:
            d0 = [r for r in dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px) if r[0] + 1 >= lo]
            d0.sort(key=lambda r: -abs(r[1]) * r[2])
            pick[n] = [(t, w) for t, w, _ in d0[:a.cf_top]]
            print(f"[② 과거 반사실] 영역 '{n}': 재생 범위(#{lo}~#{K}) 안의 관련 과거 행동 {len(d0)}개 -> 재생: " + (", ".join(f"#{t + 1}(w={w:+.2f}: {'더 눌렀어야' if w > 0 else '덜 눌렀어야'})" for t, w, _ in d0[:a.cf_top]) or "없음"), flush=True)
        starts = sorted({t + 1 for v in pick.values() for t, _ in v})
        need = {}
        for v in pick.values():
            for t, w in v:
                for vv in vars_for(w):
                    need.setdefault(t + 1, {})[vv[0]] = vv
        jobs = []
        for s0 in starts:
            jobs.append((f"s{s0}_control", s0, None))
            for vname, lf, df in need[s0].values():
                jobs.append((f"s{s0}_{vname}", s0, {s0: variant_action(actions[s0 - 1], lf, df)}))
        cache = a.replay_cache or (Path(a.history_dir).resolve().parent / "cf_replay_cache" / f"K{K}")
        cache.mkdir(parents=True, exist_ok=True)
        rep, todo_jobs = {}, []
        for j in jobs:
            cp = cache / f"{j[0]}.npy"
            if cp.exists():
                rep[j[0]] = (np.load(cp), None)
            else:
                todo_jobs.append(j)
        est = sum(K - j[1] + 1 for j in todo_jobs) * 20.0 / max(len(replay_workers), 1) / 60
        print(f"    재생 작업 {len(jobs)}개 중 캐시에 있는 것 {len(jobs) - len(todo_jobs)}개, 새로 재생 {len(todo_jobs)}개 (시작 행동 {['#' + str(x) for x in starts]}, 변형 {sorted({k for d in need.values() for k in d})}) -> 약 {est:.0f}분 예상", flush=True)
        pend, inflight, free = sorted(todo_jobs, key=lambda j: -(K - j[1] + 1)), {}, list(replay_workers)
        t_rep = time.time()
        while pend or inflight:
            while pend and free:
                w = free.pop(0)
                jid, s0, edit = pend.pop(0)
                tqs[w].put(("replay", jid, str(root), s0, K, edit))
                inflight[w] = jid
            m = rq.get(timeout=7200)
            if m[0] != "replay":
                continue
            _, w, jid, h_r, err, sec = m
            rep[jid] = (h_r, err)
            if h_r is not None:
                np.save(cache / f"{jid}.npy", h_r)
            inflight.pop(w, None)
            free.append(w)
            print(f"      재생 {jid}: {'실패 ' + str(err) if err else '완료'} ({sec:.0f}초, 경과 {(time.time() - t_rep) / 60:.1f}분)", flush=True)
        h_rec_K = after[K - 1].astype(np.float64)
        ctrl = {}
        (a.out / "cf_replay").mkdir(exist_ok=True)
        for s0 in starts:
            h_r, err = rep[f"s{s0}_control"]
            if h_r is None:
                print(f"    #{s0} control 재생 실패: {err}", flush=True)
                continue
            dd = np.abs(h_r.astype(np.float64) - h_rec_K) * 1000
            bad, p999 = int((dd > a.max_drift_mm).sum()), float(np.quantile(dd, 0.999))
            ctrl[s0] = dict(h=h_r.astype(np.float64), ok=bad <= a.max_bad_px and p999 <= a.max_p999_mm, bad=bad, p999=p999, mx=float(dd.max()))
            print(f"    #{s0} control({K - s0 + 1}행동 재생): 기록 대비 {a.max_drift_mm}mm 초과 {bad}화소, p99.9 {p999:.3f}mm, 최대 {dd.max():.2f}mm -> {'통과' if ctrl[s0]['ok'] else '불통과(이 반사실은 신뢰 낮음)'}", flush=True)
        for n in cf_names:
            for t, w_t in pick[n]:
                s0 = t + 1
                if s0 not in ctrl:
                    continue
                for vname, lf, df in vars_for(w_t):
                    h_r, err = rep.get(f"s{s0}_{vname}", (None, "없음"))
                    if h_r is None:
                        continue
                    C = h_r.astype(np.float64) - ctrl[s0]["h"]
                    np.save(a.out / "cf_replay" / f"C_s{s0}_{vname}.npy", C.astype(np.float32))
                    T = h_cur + wsoft[n] * C
                    sT = score(T, cfg)
                    q_T = s_cur["J"] - sT["J"]                                   # 참고: 현재 + C* 를 만들었을 때의 J 개선
                    # 반사실 효과 = 같은 과거 상태의 control 오차 - 행동을 바꾼 재생 오차 (전체 MAE 가 아니라 '국소 점수'로)
                    #   영역: 형태(상대 높이)/기울기(홈 깊이·외형)/테두리 선명도,  영역 밖 보존: 주변 고리 변화와 창 밖 최대 변화
                    pc, vc, _ = _terms(ctrl[s0]["h"], cfg)
                    pf, vf, _ = _terms(h_r.astype(np.float64), cfg)
                    eff_region = region_J(n, vc, cfg) - region_J(n, vf, cfg)
                    sc, sf = score(ctrl[s0]["h"], cfg), score(h_r.astype(np.float64), cfg, ctrl[s0]["h"])
                    eff_total = sc["J"] - sf["J"]
                    pres_ring = float(np.abs(C)[cfg["ring"]].mean() * 1000)
                    q = eff_region
                    row = dict(region=n, start=s0, variant=vname, control_ok=ctrl[s0]["ok"], control_bad_px=ctrl[s0]["bad"], control_p999_mm=ctrl[s0]["p999"],
                               J_now=s_cur["J"], J_if=sT["J"], q_T=q_T, q=q, effect_region=eff_region, effect_total=eff_total, preserve_ring_mm=pres_ring,
                               region_shape_ctrl=vc[n][2], region_shape_cf=vf[n][2], region_grad_ctrl=vc[n][0], region_grad_cf=vf[n][0], region_edge_ctrl=vc[n][1], region_edge_cf=vf[n][1],
                               region_mean_change_mm=float(C[masks[n]].mean() * 1000), far_change_mm=float(np.abs(C)[~wins[n]].max() * 1000))
                    cf_table.append(row)
                    ok_n = ctrl[s0]["ok"] if a.gate == "global" else local_gate(ctrl[s0]["h"], h_rec_K, wins[n], None, a.local_drift_mm, a.local_p99_mm)["ok"]
                    use = ok_n and eff_region > a.eps
                    print(f"    [{n}] 과거 #{s0}(w={w_t:+.2f} -> {'더' if w_t > 0 else '덜'}) {vname:<8}: 반사실 효과(control-재생, 국소 점수) 영역 {eff_region:+.3f} (형태 {vc[n][2]:.2f}->{vf[n][2]:.2f} 기울기 {vc[n][0]:.2f}->{vf[n][0]:.2f} 테두리 {vc[n][1]:.2f}->{vf[n][1]:.2f}) 전체J {eff_total:+.3f} | 보존(주변 고리 평균 변화) {pres_ring:.2f}mm | 창 밖 최대 {row['far_change_mm']:.1f}mm | (참고 q_T={q_T:+.3f}) | {'사용' if use else ('재생 신뢰 낮음' if not ok_n else '개선 없음')}", flush=True)
                    if use and (cf_best[n] is None or q > cf_best[n]["q"]):
                        cf_best[n] = dict(start=s0, variant=vname, C=C, q=q)
        # ---- ②b 재생 범위 밖의 영향: 영향이 큰 오래된 행동이 지금까지 남아 있는지
        #   그 행동부터 replay_len 개만 재생(검증된 길이)해 효과 C 를 구간 끝에서 얻고, 구간 이후(e+1..K) 행동의 접촉영역이 지나간 곳은 덮어써진 것으로 보고 C 에서 뺀다.
        #   남은 효과(C_persist)를 현재에 더한 목표로 영역 국소 점수가 개선되면(control 통과 시) 반사실로 쓴다. 긴 재생(드리프트)을 하지 않는 근사다.
        if a.far_top > 0:
            try:
                far_pick, seg_jobs = {}, {}
                for n in cf_names:
                    dfar = [r for r in dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px) if r[0] + 1 < lo]
                    dfar.sort(key=lambda r: -abs(r[1]) * r[2])
                    far_pick[n] = [(t, w) for t, w, _ in dfar[:a.far_top]]
                    print(f"[②b 범위 밖] 영역 '{n}': 재생 범위 밖 관련 행동 {len(dfar)}개 -> 구간 재생: " + (", ".join(f"#{t + 1}(w={w:+.2f})" for t, w in far_pick[n]) or "없음"), flush=True)
                    for t, w in far_pick[n]:
                        s0 = t + 1
                        e = min(s0 + a.replay_len - 1, K)
                        seg_jobs.setdefault(f"seg{s0}_{e}_control", (f"seg{s0}_{e}_control", s0, e, None))
                        for vname, lf, df in vars_for(w):
                            seg_jobs.setdefault(f"seg{s0}_{e}_{vname}", (f"seg{s0}_{e}_{vname}", s0, e, {s0: variant_action(actions[s0 - 1], lf, df)}))
                segrep, todo = {}, []
                for j in seg_jobs.values():
                    cp = cache / f"{j[0]}.npy"
                    if cp.exists():
                        segrep[j[0]] = (np.load(cp), None)
                    else:
                        todo.append(j)
                print(f"    구간 재생 작업 {len(seg_jobs)}개 중 새로 {len(todo)}개 (각 최대 {a.replay_len}행동)", flush=True)
                pend, inflight, free = sorted(todo, key=lambda j: -(j[2] - j[1] + 1)), {}, list(replay_workers)
                t_seg = time.time()
                while pend or inflight:
                    while pend and free:
                        w_ = free.pop(0)
                        jid, s_, e_, ed_ = pend.pop(0)
                        tqs[w_].put(("replay", jid, str(root), s_, e_, ed_))
                        inflight[w_] = jid
                    m = rq.get(timeout=7200)
                    if m[0] != "replay":
                        continue
                    _, w_, jid, h_r, err, sec = m
                    segrep[jid] = (h_r, err)
                    if h_r is not None:
                        np.save(cache / f"{jid}.npy", h_r)
                    inflight.pop(w_, None)
                    free.append(w_)
                    print(f"      구간 재생 {jid}: {'실패 ' + str(err) if err else '완료'} ({sec:.0f}초, 경과 {(time.time() - t_seg) / 60:.1f}분)", flush=True)
                _, v_now, _ = _terms(h_cur, cfg)
                for n in cf_names:
                    for t, w_t in far_pick[n]:
                        s0 = t + 1
                        e = min(s0 + a.replay_len - 1, K)
                        hc, err = segrep.get(f"seg{s0}_{e}_control", (None, "없음"))
                        if hc is None:
                            print(f"    #{s0} 구간 control 실패: {err}", flush=True)
                            continue
                        hc = hc.astype(np.float64)
                        dd = np.abs(hc - after[e - 1].astype(np.float64)) * 1000
                        bad, p999 = int((dd > a.max_drift_mm).sum()), float(np.quantile(dd, 0.999))
                        ok_c = bad <= a.max_bad_px and p999 <= a.max_p999_mm
                        if a.gate == "local":
                            ok_c = local_gate(hc, after[e - 1].astype(np.float64), wins[n], None, a.local_drift_mm, a.local_p99_mm)["ok"]
                        later = np.zeros((SIZE, SIZE), bool)
                        for u in range(e, K):                      # 구간 끝 e 번 이후 행동(e+1..K)의 접촉영역
                            later |= fps[u]
                        touched = ndimage.binary_dilation(later, iterations=max(int(round(a.persist_tools * r_tool)), 0))
                        keep = (~touched).astype(np.float64)
                        keep_frac = float(keep[masks[n]].mean())
                        print(f"    #{s0} 구간(#{s0}~#{e}) control: 기록 대비 {a.max_drift_mm}mm 초과 {bad}화소, p99.9 {p999:.3f}mm -> {'통과' if ok_c else '불통과'} | 이후 행동 {K - e}개가 덮어쓴 곳 제외, 영역 {100 * keep_frac:.0f}% 가 안 건드려짐", flush=True)
                        for vname, lf, df in vars_for(w_t):
                            hf, err = segrep.get(f"seg{s0}_{e}_{vname}", (None, "없음"))
                            if hf is None:
                                continue
                            C = hf.astype(np.float64) - hc
                            Cp = C * keep
                            T_all, T_p = h_cur + wsoft[n] * C, h_cur + wsoft[n] * Cp
                            _, v_all, _ = _terms(T_all, cfg)
                            _, v_p, _ = _terms(T_p, cfg)
                            eff_all = region_J(n, v_now, cfg) - region_J(n, v_all, cfg)
                            eff_p = region_J(n, v_now, cfg) - region_J(n, v_p, cfg)
                            np.save(a.out / "cf_replay" / f"C_seg{s0}_{e}_{vname}.npy", C.astype(np.float32))
                            chg_all, chg_p = float(C[masks[n]].mean() * 1000), float(Cp[masks[n]].mean() * 1000)
                            cf_table.append(dict(region=n, start=s0, variant=vname, kind="segment", segment_end=e, control_ok=ok_c, control_bad_px=bad, control_p999_mm=p999,
                                                 region_untouched_frac=keep_frac, region_mean_change_mm=chg_all, region_mean_change_persist_mm=chg_p, effect_if_all_persists=eff_all,
                                                 effect_persist=eff_p, q=eff_p, far_change_mm=float(np.abs(Cp)[~wins[n]].max() * 1000)))
                            use = ok_c and eff_p > a.eps
                            print(f"    [{n}] 먼 과거 #{s0}(w={w_t:+.2f} -> {'더' if w_t > 0 else '덜'}) {vname:<8}: 구간 끝에서 영역 평균 변화 {chg_all:+.2f}mm -> 이후 행동이 안 덮은 부분만 {chg_p:+.2f}mm | "
                                  f"현재에 더했을 때 영역 점수 개선: 모두 남는다면 {eff_all:+.3f}, 덮어쓰기 반영 {eff_p:+.3f} | {'사용' if use else ('구간 재생 신뢰 낮음' if not ok_c else '개선 없음')}", flush=True)
                            if use and (cf_best[n] is None or eff_p > cf_best[n]["q"]):
                                cf_best[n] = dict(start=s0, variant=vname, C=Cp, q=eff_p)
            except Exception as e_:      # 이 단계가 실패해도 나머지 탐색은 계속한다
                print(f"    (②b 범위 밖 구간 재생 실패, 건너뜀: {e_!r})", flush=True)
        # ---- ②c 관련 행동만 재생: 영역과 가장 관련 있는 행동 M개(시간 순서와 무관)를 작업 목록으로 잡고 그 행동들만 시간 순서대로 재생한다.
        #   목록 밖의 행동은 영역과 무관하다고 보고 건너뛰므로(국소성 가정) 목록이 과거에 흩어져 있어도 재생 길이는 M 개다.
        #   control(목록 그대로 재생)이 영역 창 안에서 기록된 현재와 맞는지로 그 가정을 검증하고, 맞을 때만 쓴다.
        if a.rel_replay > 0:
            try:
                rel_lists, rel_pick, rel_jobs = {}, {}, {}
                for n in cf_names:
                    dall = dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px)
                    dall.sort(key=lambda r: -r[2])                      # 기록된 영역 영향이 큰 순으로 목록을 구성
                    rel_lists[n] = sorted(t + 1 for t, _, _ in dall[:a.rel_replay])
                    pk = sorted([r for r in dall if r[0] + 1 in rel_lists[n]], key=lambda r: -abs(r[1]) * r[2])[:a.rel_top]
                    rel_pick[n] = [(t, w) for t, w, _ in pk]
                    print(f"[②c 관련 행동만 재생] 영역 '{n}': 관련 행동 {len(dall)}개 중 영향 큰 {len(rel_lists[n])}개 "
                          f"(#{rel_lists[n][0]}~#{rel_lists[n][-1]}) 작업 목록 -> 바꿔 볼 행동: " + (", ".join(f"#{t + 1}(w={w:+.2f})" for t, w in rel_pick[n]) or "없음"), flush=True)
                    if not rel_lists[n]:
                        continue
                    st0 = rel_lists[n][0]
                    tag = f"rel{a.rel_replay}_K{K}_{n}"
                    rel_jobs[f"{tag}_control"] = (f"{tag}_control", st0, K, None, rel_lists[n])
                    for t, w in rel_pick[n]:
                        for vname, lf, df in vars_for(w):
                            rel_jobs[f"{tag}_{t + 1}_{vname}"] = (f"{tag}_{t + 1}_{vname}", st0, K, {t + 1: variant_action(actions[t], lf, df)}, rel_lists[n])
                relrep, todo = {}, []
                for j in rel_jobs.values():
                    cp = cache / f"{j[0]}.npy"
                    if cp.exists():
                        relrep[j[0]] = (np.load(cp), None)
                    else:
                        todo.append(j)
                print(f"    관련 행동 재생 작업 {len(rel_jobs)}개 중 새로 {len(todo)}개 (각 {a.rel_replay}행동만 실행)", flush=True)
                pend, inflight, free = list(todo), {}, list(replay_workers)
                t_rel = time.time()
                while pend or inflight:
                    while pend and free:
                        w_ = free.pop(0)
                        jid, s_, e_, ed_, only_ = pend.pop(0)
                        tqs[w_].put(("replay", jid, str(root), s_, e_, ed_, only_))
                        inflight[w_] = jid
                    m = rq.get(timeout=7200)
                    if m[0] != "replay":
                        continue
                    _, w_, jid, h_r, err, sec = m
                    relrep[jid] = (h_r, err)
                    if h_r is not None:
                        np.save(cache / f"{jid}.npy", h_r)
                    inflight.pop(w_, None)
                    free.append(w_)
                    print(f"      관련 행동 재생 {jid}: {'실패 ' + str(err) if err else '완료'} ({sec:.0f}초, 경과 {(time.time() - t_rel) / 60:.1f}분)", flush=True)
                _, v_now, _ = _terms(h_cur, cfg)
                h_rec_K2 = after[K - 1].astype(np.float64)
                for n in cf_names:
                    if not rel_lists.get(n):
                        continue
                    tag = f"rel{a.rel_replay}_K{K}_{n}"
                    hc, err = relrep.get(f"{tag}_control", (None, "없음"))
                    if hc is None:
                        print(f"    [{n}] 관련 행동 control 실패: {err}", flush=True)
                        continue
                    hc = hc.astype(np.float64)
                    W_ = wins[n]
                    dd = np.abs(hc - h_rec_K2)[W_] * 1000
                    bad, p999 = int((dd > a.max_drift_mm).sum()), float(np.quantile(dd, 0.999))
                    ok_c = bad <= a.max_bad_px and p999 <= a.max_p999_mm
                    print(f"    [{n}] 관련 {len(rel_lists[n])}개만 재생한 control: 영역 창({int(W_.sum())}화소) 안에서 기록 대비 {a.max_drift_mm}mm 초과 {bad}화소, p99.9 {p999:.3f}mm, 최대 {dd.max():.2f}mm "
                          f"-> {'통과(건너뛴 행동은 영역에 영향이 없다고 볼 수 있음)' if ok_c else '불통과(건너뛴 행동의 영향이 있음, 이 반사실은 신뢰 낮음)'}", flush=True)
                    for t, w_t in rel_pick[n]:
                        s0 = t + 1
                        for vname, lf, df in vars_for(w_t):
                            hf, err = relrep.get(f"{tag}_{s0}_{vname}", (None, "없음"))
                            if hf is None:
                                continue
                            C = hf.astype(np.float64) - hc
                            Tn_ = h_cur + wsoft[n] * C
                            _, v_T, _ = _terms(Tn_, cfg)
                            eff_r = region_J(n, v_now, cfg) - region_J(n, v_T, cfg)       # 현재에 C 를 더했을 때 그 영역 국소 점수의 개선
                            q_Tt = s_cur["J"] - score(Tn_, cfg)["J"]
                            np.save(a.out / "cf_replay" / f"C_rel_{n}_{s0}_{vname}.npy", C.astype(np.float32))
                            cf_table.append(dict(region=n, start=s0, variant=vname, kind="relevant_only", list_len=len(rel_lists[n]), control_ok=ok_c, control_bad_px=bad, control_p999_mm=p999,
                                                 effect_region=eff_r, q_T=q_Tt, q=eff_r, region_mean_change_mm=float(C[masks[n]].mean() * 1000), far_change_mm=float(np.abs(C)[~wins[n]].max() * 1000)))
                            use = ok_c and eff_r > a.eps
                            print(f"    [{n}] 관련 행동 #{s0}(w={w_t:+.2f} -> {'더' if w_t > 0 else '덜'}) {vname:<8}: 영역 평균 변화 {float(C[masks[n]].mean() * 1000):+.2f}mm | "
                                  f"현재에 더하면 영역 점수 개선 {eff_r:+.3f} (전체 J {q_Tt:+.3f}) | {'사용' if use else ('control 불통과' if not ok_c else '개선 없음')}", flush=True)
                            if use and (cf_best[n] is None or eff_r > cf_best[n]["q"]):
                                cf_best[n] = dict(start=s0, variant=vname, C=C, q=eff_r)
            except Exception as e_:      # 이 단계가 실패해도 나머지 탐색은 계속한다
                print(f"    (②c 관련 행동 재생 실패, 건너뜀: {e_!r})", flush=True)
        # ---- ②L 장거리 반사실: 재생 범위 밖(#s0 < lo)의 영향 큰 과거 행동을 '현재까지 전체 길이' 재생한다.
        #   홈을 판 #2, #29 는 78·51행동 전이라 ② 의 20행동 범위 밖이고, ②b 는 구간 끝에서 멈춰 이후 행동의 덮어쓰기를 근사로 처리한다.
        #   여기서는 근사 없이 s0 부터 현재(K)까지 control 과 변형을 같은 엔진에서 전부 재생하고, C = H_cf - H_ctrl 을 쓴다.
        #   게이트는 전역이 아니라 영역 창 안의 control-기록 어긋남(local_gate)과 효과 대 잡음비(SNR)로 본다. 복제(--long-replicates 2)가 있으면 잡음을 직접 잰다.
        if a.long_top > 0:
            try:
                lg_pick, lg_jobs, lg_group, lg_spent = {}, {}, {}, 0.0
                for n in cf_names:
                    dl = [r for r in dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px) if r[0] + 1 < lo]
                    dl.sort(key=lambda r: -abs(r[1]) * r[2])
                    if a.long_budget_min > 0:
                        pk = []
                        for t_c, w_c, inf_c in sorted(dl, key=lambda r: -abs(r[1]) * r[2] / (K - r[0])):          # 이득/비용 순
                            runs = 1 + max(a.long_replicates - 1, 0) + len(vars_for(w_c))
                            wall = (K - t_c) * runs * a.replay_sec_per_action / 60.0 / max(len(replay_workers), 1)
                            if lg_spent + wall <= a.long_budget_min:
                                pk.append((t_c, w_c))
                                lg_spent += wall
                        lg_pick[n] = pk
                    else:
                        lg_pick[n] = [(t, w) for t, w, _ in dl[:a.long_top]]
                    print(f"[②L 장거리] 영역 '{n}': 재생 범위(#{lo}~#{K}) 밖 관련 행동 {len(dl)}개 -> 현재까지 전체 길이 재생: "
                          + (", ".join(f"#{t + 1}(w={w:+.2f}, {K - t}행동)" for t, w in lg_pick[n]) or "없음"), flush=True)
                    if a.long_group >= 2 and len(dl) >= 2:
                        gl = [(t, w) for t, w, _ in dl[:a.long_group]]
                        members = sorted((t + 1, vars_for(w)[0]) for t, w in gl)            # (시작 행동 번호, (변형 이름, 길이배율, 깊이배율))
                        s_min = members[0][0]
                        gname = f"s{s_min}_grp_" + "-".join(f"{s_}{v[0]}" for s_, v in members)
                        lg_group[n] = dict(name=gname, s_min=s_min, members=members, top=gl[0][0] + 1)
                        lg_jobs.setdefault(f"s{s_min}_control", (f"s{s_min}_control", s_min, K, None))
                        for r_i in range(2, a.long_replicates + 1):
                            lg_jobs.setdefault(f"s{s_min}_control_rep{r_i}", (f"s{s_min}_control_rep{r_i}", s_min, K, None))
                        lg_jobs.setdefault(gname, (gname, s_min, K, {s_: variant_action(actions[s_ - 1], v[1], v[2]) for s_, v in members}))
                        print(f"[②L 그룹] 영역 '{n}': 상위 {len(members)}개 행동을 함께 바꿈: " + ", ".join(f"#{s_}({v[0]})" for s_, v in members) + f" -> #{s_min}부터 {K - s_min + 1}행동 재생", flush=True)
                    for t, w in lg_pick[n]:
                        s0 = t + 1
                        lg_jobs.setdefault(f"s{s0}_control", (f"s{s0}_control", s0, K, None))
                        for r_i in range(2, a.long_replicates + 1):
                            lg_jobs.setdefault(f"s{s0}_control_rep{r_i}", (f"s{s0}_control_rep{r_i}", s0, K, None))
                        for vname, lf, df in vars_for(w):
                            lg_jobs.setdefault(f"s{s0}_{vname}", (f"s{s0}_{vname}", s0, K, {s0: variant_action(actions[s0 - 1], lf, df)}))
                lgrep, todo = {}, []
                for j in lg_jobs.values():
                    cp = cache / f"{j[0]}.npy"
                    if cp.exists():
                        lgrep[j[0]] = (np.load(cp), None)
                    else:
                        todo.append(j)
                est = sum(K - j[1] + 1 for j in todo) * a.replay_sec_per_action / max(len(replay_workers), 1) / 60
                if a.long_budget_min > 0:
                    print(f"    시간 예산 {a.long_budget_min:g}분 중 {lg_spent:.0f}분 사용(이득/비용 순 선택, 영역 합산·control 공유 미반영의 보수적 추정)", flush=True)
                print(f"    장거리 재생 작업 {len(lg_jobs)}개 중 캐시에 있는 것 {len(lg_jobs) - len(todo)}개, 새로 {len(todo)}개 -> 워커 {len(replay_workers)}개 병렬로 약 {est:.0f}분 예상(17~20초/행동)", flush=True)
                pend, inflight, free = sorted(todo, key=lambda j: -(K - j[1] + 1)), {}, list(replay_workers)
                t_lg = time.time()
                while pend or inflight:
                    while pend and free:
                        w_ = free.pop(0)
                        jid, s_, e_, ed_ = pend.pop(0)
                        tqs[w_].put(("replay", jid, str(root), s_, e_, ed_))
                        inflight[w_] = jid
                    m = rq.get(timeout=14400)
                    if m[0] != "replay":
                        continue
                    _, w_, jid, h_r, err, sec = m
                    lgrep[jid] = (h_r, err)
                    if h_r is not None:
                        np.save(cache / f"{jid}.npy", h_r)
                    inflight.pop(w_, None)
                    free.append(w_)
                    print(f"      장거리 재생 {jid}: {'실패 ' + str(err) if err else '완료'} ({sec:.0f}초, 경과 {(time.time() - t_lg) / 60:.1f}분)", flush=True)
                h_rec_L = after[K - 1].astype(np.float64)
                _, v_now_l, _ = _terms(h_cur, cfg)
                for n in cf_names:
                    for t, w_t in lg_pick[n]:
                        s0 = t + 1
                        hc, err = lgrep.get(f"s{s0}_control", (None, "없음"))
                        if hc is None:
                            print(f"    [{n}] #{s0} 장거리 control 실패: {err}", flush=True)
                            continue
                        hc = hc.astype(np.float64)
                        reps = [lgrep[k_][0] for k_ in (f"s{s0}_control_rep{r_i}" for r_i in range(2, a.long_replicates + 1)) if lgrep.get(k_, (None, None))[0] is not None]
                        gt = local_gate(hc, h_rec_L, wins[n], reps[0].astype(np.float64) if reps else None, a.local_drift_mm, a.local_p99_mm)
                        print(f"    [{n}] #{s0} {K - s0 + 1}행동 control: 영역 창({int(wins[n].sum())}화소) 안에서 기록 대비 평균 {gt['mae']:.3f}mm, p99 {gt['p99']:.2f}mm "
                              f"(상한 {a.local_drift_mm:g}/{a.local_p99_mm:g}) | " + (f"같은 control 복제와의 차이(잡음) {gt['noise']:.4f}mm" if reps else "복제 없음: 잡음은 기록 대비 차이로 대신(보수적)")
                              + f" -> 국소 게이트 {'통과' if gt['ok'] else '불통과'}", flush=True)
                        for vname, lf, df in vars_for(w_t):
                            hf, err = lgrep.get(f"s{s0}_{vname}", (None, "없음"))
                            if hf is None:
                                continue
                            C = hf.astype(np.float64) - hc
                            effect = float(np.abs(C)[masks[n]].mean() * 1000)
                            snr = effect / max(gt["noise"], 1e-3)
                            Tn_ = h_cur + wsoft[n] * C
                            _, v_T, _ = _terms(Tn_, cfg)
                            eff_l = region_J(n, v_now_l, cfg) - region_J(n, v_T, cfg)
                            q_Tl = s_cur["J"] - score(Tn_, cfg)["J"]
                            far_l = float(np.abs(C)[~wins[n]].max() * 1000)
                            np.save(a.out / "cf_replay" / f"C_long_{n}_{s0}_{vname}.npy", C.astype(np.float32))
                            use = gt["ok"] and snr >= a.min_snr and eff_l > a.eps
                            cf_table.append(dict(region=n, start=s0, variant=vname, kind="long", replay_len=K - s0 + 1, control_ok=gt["ok"], local_drift_mae_mm=gt["mae"], local_drift_p99_mm=gt["p99"],
                                                 noise_mm=gt["noise"], replicates=1 + len(reps), effect_abs_mm=effect, snr=snr, effect_region=eff_l, q_T=q_Tl, q=eff_l,
                                                 region_mean_change_mm=float(C[masks[n]].mean() * 1000), far_change_mm=far_l))
                            why = "사용" if use else ("국소 게이트 불통과" if not gt["ok"] else (f"SNR {snr:.1f} < {a.min_snr:g}" if snr < a.min_snr else "개선 없음"))
                            print(f"    [{n}] 장거리 #{s0}(w={w_t:+.2f} -> {'더' if w_t > 0 else '덜'}) {vname:<8}: 영역 평균 변화 {float(C[masks[n]].mean() * 1000):+.2f}mm (|C| {effect:.2f}mm, SNR {snr:.0f}) | "
                                  f"현재에 더하면 영역 점수 개선 {eff_l:+.3f} (전체 J {q_Tl:+.3f}) | 창 밖 최대 {far_l:.1f}mm | {why}", flush=True)
                            if use and (cf_best[n] is None or eff_l > cf_best[n]["q"]):
                                cf_best[n] = dict(start=s0, variant=vname, C=C, q=eff_l)
                    gi = lg_group.get(n)
                    if gi is not None:
                        hc, _e = lgrep.get(f"s{gi['s_min']}_control", (None, "없음"))
                        hf, _e2 = lgrep.get(gi["name"], (None, "없음"))
                        if hc is None or hf is None:
                            print(f"    [{n}] 그룹 반사실 재생 실패: control {_e}, 변형 {_e2}", flush=True)
                            continue
                        hc = hc.astype(np.float64)
                        reps = [lgrep[k_][0] for k_ in (f"s{gi['s_min']}_control_rep{r_i}" for r_i in range(2, a.long_replicates + 1)) if lgrep.get(k_, (None, None))[0] is not None]
                        gt = local_gate(hc, h_rec_L, wins[n], reps[0].astype(np.float64) if reps else None, a.local_drift_mm, a.local_p99_mm)
                        C = hf.astype(np.float64) - hc
                        effect = float(np.abs(C)[masks[n]].mean() * 1000)
                        snr = effect / max(gt["noise"], 1e-3)
                        Tn_ = h_cur + wsoft[n] * C
                        _, v_T, _ = _terms(Tn_, cfg)
                        eff_g = region_J(n, v_now_l, cfg) - region_J(n, v_T, cfg)
                        q_Tg = s_cur["J"] - score(Tn_, cfg)["J"]
                        far_g = float(np.abs(C)[~wins[n]].max() * 1000)
                        np.save(a.out / "cf_replay" / f"C_group_{n}_{gi['s_min']}.npy", C.astype(np.float32))
                        use = gt["ok"] and snr >= a.min_snr and eff_g > a.eps
                        cf_table.append(dict(region=n, start=gi["top"], variant="group:" + "+".join(f"{s_}{v[0]}" for s_, v in gi["members"]), kind="long_group", replay_len=K - gi["s_min"] + 1,
                                             members=[s_ for s_, _ in gi["members"]], control_ok=gt["ok"], local_drift_mae_mm=gt["mae"], local_drift_p99_mm=gt["p99"], noise_mm=gt["noise"],
                                             effect_abs_mm=effect, snr=snr, effect_region=eff_g, q_T=q_Tg, q=eff_g, region_mean_change_mm=float(C[masks[n]].mean() * 1000), far_change_mm=far_g))
                        why = "사용" if use else ("국소 게이트 불통과" if not gt["ok"] else (f"SNR {snr:.1f} < {a.min_snr:g}" if snr < a.min_snr else "개선 없음"))
                        print(f"    [{n}] 그룹 {[s_ for s_, _ in gi['members']]}: control 창 평균 {gt['mae']:.3f}mm p99 {gt['p99']:.2f}mm {'통과' if gt['ok'] else '불통과'} | 영역 평균 변화 {float(C[masks[n]].mean() * 1000):+.2f}mm "
                              f"(|C| {effect:.2f}mm, SNR {snr:.0f}) | 현재에 더하면 영역 점수 개선 {eff_g:+.3f} (전체 J {q_Tg:+.3f}) | 창 밖 최대 {far_g:.1f}mm | {why}", flush=True)
                        if use and (cf_best[n] is None or eff_g > cf_best[n]["q"]):
                            cf_best[n] = dict(start=gi["top"], variant="group", C=C, q=eff_g)
            except Exception as e_:      # 이 단계가 실패해도 나머지 탐색은 계속한다
                print(f"    (②L 장거리 반사실 실패, 건너뜀: {e_!r})", flush=True)
        # ---- ②T 계층(트리) 탐색 (cf_tree.py): 관련 행동을 시간 구간 묶음(노드)으로 나눠 묶음째 Genesis 에서 재생하고, 효과가 있는 묶음만 쪼갠다.
        #   묶음 재생 = 묶음 안 모든 행동을 각자 w 부호 방향 변형으로 바꿔 가장 이른 행동부터 현재까지 한 번 재생(그룹 반사실). 효과 C = H_묶음 - H_control.
        #   control: --control record 면 기록된 현재 상태(검증된 길이 이내일 때만), 아니면 같은 구간을 바꾸지 않고 재생. 가지치기·확정은 Genesis 결과로만 한다.
        if a.long_select in ("tree", "both"):                 # both = ②T 트리 + ②S 묶음 탐색을 모두 (②T 가 먼저, 재생 캐시를 ②S 가 이어 쓴다. 더 큰 이득의 반사실이 cf_best 가 된다)
            try:
                import cf_tree as ct
                import hashlib

                def run_jobs(job_list, tag):
                    res, todo = {}, []
                    for j in job_list:
                        cp = cache / f"{j[0]}.npy"
                        if cp.exists():
                            res[j[0]] = (np.load(cp), None)
                        else:
                            todo.append(j)
                    print(f"    [{tag}] 재생 작업 {len(job_list)}개 중 캐시 {len(job_list) - len(todo)}개, 새로 {len(todo)}개 (재생 행동 합 {sum(K - j[1] + 1 for j in todo)}개, 워커 {len(replay_workers)}개)", flush=True)
                    pend, inflight, free, t0 = sorted(todo, key=lambda j: -(K - j[1] + 1)), {}, list(replay_workers), time.time()
                    while pend or inflight:
                        while pend and free:
                            w_ = free.pop(0)
                            jid, s_, e_, ed_ = pend.pop(0)
                            tqs[w_].put(("replay", jid, str(root), s_, e_, ed_))
                            inflight[w_] = jid
                        m = rq.get(timeout=14400)
                        if m[0] != "replay":
                            continue
                        _, w_, jid, h_r, err, sec = m
                        res[jid] = (h_r, err)
                        if h_r is not None:
                            np.save(cache / f"{jid}.npy", h_r)
                        inflight.pop(w_, None)
                        free.append(w_)
                        print(f"      재생 {jid}: {'실패 ' + str(err) if err else '완료'} ({sec:.0f}초, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
                    return res

                def ctrl_evidence(n_):
                    """캐시된 control 재생 중 이 영역 창에서 국소 게이트를 통과한 것의 최대 재생 길이(행동 수)."""
                    best_len = 0
                    for fcp in cache.glob("s*_control.npy"):
                        try:
                            s_ = int(fcp.name[1:].split("_")[0])
                        except ValueError:
                            continue
                        if local_gate(np.load(fcp).astype(np.float64), h_rec_K, wins[n_], None, a.local_drift_mm, a.local_p99_mm)["ok"]:
                            best_len = max(best_len, K - s_ + 1)
                    return best_len

                _, v_now_t, _ = _terms(h_cur, cfg)
                for n in cf_names:
                    rows_all = dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px)
                    if len(rows_all) < 2:
                        print(f"[②T 트리] 영역 '{n}': 관련 행동 {len(rows_all)}개라 건너뜀", flush=True)
                        continue
                    w_of = {t + 1: w for t, w, e in rows_all}
                    # 노드는 w 의 부호가 같은 행동끼리만 묶는다: 묶음 안에 '더'(w>0)와 '덜'(w<0) 변형이 섞이면 서로 상쇄해서 효과 있는 행동이 가려진다
                    # (A 에서 #57 skip + #59/#62/#71/#75 깊이 x1.5 묶음은 영역 평균 변화 -0.02mm, 점수 개선 +0.07 인데 #75 하나는 +0.54 였다)
                    rows_t = [(t + 1, w, e) for t, w, e in rows_all]
                    frontier, pri = [], {}
                    for part_rows in ([r_ for r_ in rows_t if r_[1] > 0], [r_ for r_ in rows_t if r_[1] <= 0]):
                        if not part_rows:
                            continue
                        fr_, pr_ = ct.initial_nodes(part_rows, max(1, int(round(a.tree_init * len(part_rows) / len(rows_t)))))
                        frontier += fr_
                        pri.update(pr_)
                    nid, wave, spent_min, nres, t_tree = 100, 0, 0.0, 0, time.time()
                    ev = max(ctrl_evidence(n), a.ctrl_valid_len) if a.control == "record" else 0
                    print(f"[②T 트리] 영역 '{n}': 관련 행동 {len(rows_all)}개(#{min(w_of)}~#{max(w_of)}) -> 초기 노드 {len(frontier)}개 "
                          + ", ".join(f"#{nd.members[0]}~#{nd.members[-1]}({len(nd.members)}개)" for nd in frontier)
                          + f" | 예산 {a.tree_budget_min:g}분 | control={a.control}" + (f"(검증된 control 최대 {ev}행동 x{a.control_extrap:g} 까지 기록 사용)" if a.control == "record" else ""), flush=True)
                    best_t = None
                    while frontier and spent_min < a.tree_budget_min:
                        wave += 1
                        budget_actions = (a.tree_budget_min - spent_min) * 60.0 * max(len(replay_workers), 1) / a.replay_sec_per_action
                        chosen, frontier = ct.pick_wave(frontier, K, budget_actions, max_single=(a.tree_budget_min - spent_min) * 60.0 / a.replay_sec_per_action)
                        if not chosen:
                            print(f"    (파동 {wave}: 남은 예산으로 재생할 수 있는 노드가 없어 종료)", flush=True)
                            break
                        jobs, info, use_rec_of = {}, {}, {}
                        for nd in chosen:
                            vp = ct.variant_plan(nd, w_of, vars_for)
                            jid = f"s{nd.s_min}_{vp[nd.s_min][0]}" if nd.is_leaf else \
                                f"s{nd.s_min}_g" + hashlib.md5(json.dumps(sorted((m_, v_[0]) for m_, v_ in vp.items())).encode()).hexdigest()[:10]
                            jobs[jid] = (jid, nd.s_min, K, {m_: variant_action(actions[m_ - 1], v_[1], v_[2]) for m_, v_ in vp.items()})
                            info[jid] = (nd, vp)
                            if not nd.is_leaf:
                                (cache / f"{jid}.json").write_text(json.dumps(dict(region=n, start=nd.s_min, last=K, members={str(m_): v_[0] for m_, v_ in vp.items()}), ensure_ascii=False))
                            use_rec = a.control == "record" and nd.cost(K) <= a.control_extrap * ev
                            use_rec_of[jid] = use_rec
                            if a.control == "record" and not use_rec:
                                print(f"    (노드 #{nd.members[0]}~#{nd.members[-1]}: 필요한 재생 {nd.cost(K)}행동이 검증된 control {ev}행동의 {a.control_extrap:g}배를 넘어 control 을 직접 재생)", flush=True)
                            if not use_rec:
                                jobs.setdefault(f"s{nd.s_min}_control", (f"s{nd.s_min}_control", nd.s_min, K, None))
                                for r_i in range(2, a.long_replicates + 1):
                                    jobs.setdefault(f"s{nd.s_min}_control_rep{r_i}", (f"s{nd.s_min}_control_rep{r_i}", nd.s_min, K, None))
                        t_wave = time.time()
                        res = run_jobs(list(jobs.values()), f"파동 {wave}")
                        spent_min += (time.time() - t_wave) / 60.0
                        for jid, (nd, vp) in info.items():
                            hf, err = res.get(jid, (None, "없음"))
                            if hf is None:
                                print(f"    노드 #{nd.members[0]}~#{nd.members[-1]} 재생 실패({err}) -> 가지 버림", flush=True)
                                continue
                            L_ = nd.cost(K)
                            if use_rec_of[jid]:
                                hc, noise, gate_ok = h_rec_K, max(a.noise_floor_mm, a.noise_per_action_mm * L_), True
                                ctl = "기록"
                            else:
                                hc, e_c = res.get(f"s{nd.s_min}_control", (None, "없음"))
                                if hc is None:
                                    print(f"    노드 #{nd.members[0]}~#{nd.members[-1]} control 재생 실패({e_c}) -> 가지 버림", flush=True)
                                    continue
                                hc = hc.astype(np.float64)
                                reps = [res[k_][0] for k_ in (f"s{nd.s_min}_control_rep{r_i}" for r_i in range(2, a.long_replicates + 1)) if res.get(k_, (None, None))[0] is not None]
                                gt = local_gate(hc, h_rec_K, wins[n], reps[0].astype(np.float64) if reps else None, a.local_drift_mm, a.local_p99_mm)
                                noise, gate_ok, ctl = gt["noise"], gt["ok"], f"재생(창 평균 어긋남 {gt['mae']:.3f}mm)"
                            C = hf.astype(np.float64) - hc
                            effect = float(np.abs(C)[masks[n]].mean() * 1000)
                            snr = effect / max(noise, 1e-3)
                            Tn_ = h_cur + wsoft[n] * C
                            _, v_T, _ = _terms(Tn_, cfg)
                            eff_t = region_J(n, v_now_t, cfg) - region_J(n, v_T, cfg)
                            far_t = float(np.abs(C)[~wins[n]].max() * 1000)
                            nres += 1
                            cf_table.append(dict(region=n, start=nd.s_min, variant=f"tree{len(nd.members)}", kind="tree", members=nd.members, depth=nd.depth, prior=nd.prior, replay_len=L_, wave=wave,
                                                 control=("record" if use_rec_of[jid] else "replay"), control_ok=gate_ok, noise_mm=noise, effect_abs_mm=effect, snr=snr, effect_region=eff_t, q=eff_t,
                                                 region_mean_change_mm=float(C[masks[n]].mean() * 1000), far_change_mm=far_t))
                            promising = gate_ok and snr >= a.min_snr and eff_t > a.eps
                            why = "유망 -> " + ("원인 후보(잎)" if nd.is_leaf else "쪼갬") if promising else ("게이트 불통과" if not gate_ok else (f"SNR {snr:.1f}<{a.min_snr:g}" if snr < a.min_snr else "효과 없음 -> 가지 버림"))
                            print(f"    [파동 {wave}] 노드 깊이{nd.depth} #{nd.members[0]}~#{nd.members[-1]}({len(nd.members)}개, 재생 {L_}행동, control {ctl}): 영역 평균 변화 {float(C[masks[n]].mean() * 1000):+.2f}mm "
                                  f"(|C| {effect:.2f}, SNR {snr:.0f}) | 영역 점수 개선 {eff_t:+.3f} | 창 밖 최대 {far_t:.1f}mm | {why}", flush=True)
                            if promising:
                                if not nd.is_leaf:
                                    kids = ct.children(nd, pri, nid, a.tree_min_prior_frac)
                                    nid += len(kids)
                                    frontier += kids
                                if best_t is None or eff_t > best_t["q"]:
                                    top_m = max(nd.members, key=lambda m_: pri[m_])
                                    best_t = dict(start=top_m, variant=f"tree{len(nd.members)}", C=C, q=eff_t, members=nd.members)
                    if best_t is not None and (cf_best[n] is None or best_t["q"] > cf_best[n]["q"]):
                        cf_best[n] = dict(start=best_t["start"], variant=best_t["variant"], C=best_t["C"], q=best_t["q"])
                    print(f"[②T 트리] 영역 '{n}' 종료: 노드 {nres}개 재생({wave}파동, {spent_min:.0f}분)" + (f", 최선: 묶음 {best_t['members']} 영역 점수 개선 {best_t['q']:+.3f}" if best_t else ", 효과 있는 묶음 없음(abstain)")
                          + (f" | 남은 미재생 노드 {len(frontier)}개(예산 소진)" if frontier else ""), flush=True)
            except Exception as e_:      # 이 단계가 실패해도 나머지 탐색은 계속한다
                import traceback
                if a.strict:
                    raise
                print(f"    (②T 트리 탐색 실패, 건너뜀: {e_!r})\n{traceback.format_exc()}", flush=True)
        # ---- ②S 개입 묶음 탐색 (cf_seq_cf.py; Tsirtsis & Gomez-Rodriguez, NeurIPS 2023 [10] 의 구조만 사용): 과거 행동을 '최대 k 개'만 바꿔 영역 점수가 가장 좋아지는 묶음을 고른다.
        #   후보 = 영역에 영향이 큰 과거 행동(전체 #1~#K, 재생 범위 제한 없음) 상위 --seq-pool 개 x w 부호 방향의 첫 변형. 단일 개입 이득 g1 은 단일 재생으로 재고,
        #   묶음 이득은 묶음 전체를 가장 이른 행동부터 현재까지 한 번 재생해(control 대비) 잰다. 최적성 보장은 없다(5차원 연속 푸시). 평가는 항상 Genesis 재생 결과다.
        #   재생은 ②T 와 같은 캐시를 쓴다(단일 개입은 같은 이름이라 공유). control 은 --control record 면 기록된 현재 상태, 아니면 같은 구간을 바꾸지 않고 재생한다.
        if a.long_select in ("seq", "both"):
            try:
                import cf_seq_cf as sq
                import hashlib

                def seq_jobs(job_list):
                    res, todo = {}, []
                    for j in job_list:
                        cp = cache / f"{j[0]}.npy"
                        if cp.exists():
                            res[j[0]] = (np.load(cp), None)
                        else:
                            todo.append(j)
                    print(f"    [②S] 재생 작업 {len(job_list)}개 중 캐시 {len(job_list) - len(todo)}개, 새로 {len(todo)}개 (재생 행동 합 {sum(K - j[1] + 1 for j in todo)}개, 워커 {len(replay_workers)}개)", flush=True)
                    pend, inflight, free, t0 = sorted(todo, key=lambda j: -(K - j[1] + 1)), {}, list(replay_workers), time.time()
                    while pend or inflight:
                        while pend and free:
                            w_ = free.pop(0)
                            jid, s_, e_, ed_ = pend.pop(0)
                            tqs[w_].put(("replay", jid, str(root), s_, e_, ed_))
                            inflight[w_] = jid
                        m = rq.get(timeout=14400)
                        if m[0] != "replay":
                            continue
                        _, w_, jid, h_r, err, sec = m
                        res[jid] = (h_r, err)
                        if h_r is not None:
                            np.save(cache / f"{jid}.npy", h_r)
                        inflight.pop(w_, None)
                        free.append(w_)
                        print(f"      재생 {jid}: {'실패 ' + str(err) if err else '완료'} ({sec:.0f}초, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
                    return res

                def seq_ctrl_len(n_):
                    """캐시된 control 재생 중 이 영역 창에서 국소 게이트를 통과한 것의 최대 재생 길이(행동 수)."""
                    best_len = 0
                    for fcp in cache.glob("s*_control.npy"):
                        try:
                            s_ = int(fcp.name[1:].split("_")[0])
                        except ValueError:
                            continue
                        if local_gate(np.load(fcp).astype(np.float64), h_rec_K, wins[n_], None, a.local_drift_mm, a.local_p99_mm)["ok"]:
                            best_len = max(best_len, K - s_ + 1)
                    return best_len

                _, v_now_s, _ = _terms(h_cur, cfg)
                for n in cf_names:
                    rows_all = sorted(dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px), key=lambda r_: -abs(r_[1]) * r_[2])
                    if len(rows_all) < 2:
                        print(f"[②S 묶음] 영역 '{n}': 관련 행동 {len(rows_all)}개라 건너뜀", flush=True)
                        continue
                    w_of = {t_ + 1: w_ for t_, w_, e_ in rows_all}
                    cand_var = {}
                    for t_, w_, e_ in rows_all[:a.seq_pool]:
                        vv = vars_for(w_)[0]                                  # w 의 부호 방향의 첫 변형 하나: (이름, 길이배율, 깊이배율)
                        cand_var[(t_ + 1, vv[0])] = (vv[1], vv[2])
                    ev = max(seq_ctrl_len(n), a.ctrl_valid_len) if a.control == "record" else 0
                    t_seq = time.time()

                    def jid_of(members):
                        s_min = members[0][0]
                        if len(members) == 1:
                            return f"s{s_min}_{members[0][1]}"                 # ②/②T 의 단일 재생과 같은 이름이라 캐시를 공유한다
                        return f"s{s_min}_g" + hashlib.md5(json.dumps(members).encode()).hexdigest()[:10]

                    def effects(bundles):
                        """묶음(정렬된 (행동 번호, 변형 이름) 목록)마다 재생해 영역 점수 이득 q 를 구한다. 한 번에 여러 묶음을 워커에 나눠 돌린다."""
                        jobs, rec_ctl = {}, {}
                        for members in bundles:
                            jid, s_min = jid_of(members), members[0][0]
                            jobs[jid] = (jid, s_min, K, {m_: variant_action(actions[m_ - 1], *cand_var[(m_, v_)]) for m_, v_ in members})
                            rec_ctl[jid] = a.control == "record" and (K - s_min + 1) <= a.control_extrap * ev
                            if not rec_ctl[jid]:
                                jobs.setdefault(f"s{s_min}_control", (f"s{s_min}_control", s_min, K, None))
                        res = seq_jobs(list(jobs.values()))
                        out = {}
                        for members in bundles:
                            jid, s_min = jid_of(members), members[0][0]
                            hf, err = res.get(jid, (None, "없음"))
                            out[tuple(members)] = None
                            if hf is None:
                                continue
                            L_ = K - s_min + 1
                            if rec_ctl[jid]:
                                hc, noise, gate_ok = h_rec_K, max(a.noise_floor_mm, a.noise_per_action_mm * L_), True
                            else:
                                hc, e_c = res.get(f"s{s_min}_control", (None, "없음"))
                                if hc is None:
                                    continue
                                hc = hc.astype(np.float64)
                                gt = local_gate(hc, h_rec_K, wins[n], None, a.local_drift_mm, a.local_p99_mm)
                                noise, gate_ok = gt["noise"], gt["ok"]
                            C = hf.astype(np.float64) - hc
                            effect = float(np.abs(C)[masks[n]].mean() * 1000)
                            snr = effect / max(noise, 1e-3)
                            _, v_T, _ = _terms(h_cur + wsoft[n] * C, cfg)
                            q_J = region_J(n, v_now_s, cfg) - region_J(n, v_T, cfg)
                            mae_g = float((np.abs(h_cur - tgts[n])[masks[n]].mean() - np.abs(h_cur + wsoft[n] * C - tgts[n])[masks[n]].mean()) * 1000)      # 영역 MAE 이득(mm): 반사실 T 가 영역 목표에 얼마나 더 가까운가
                            out[tuple(members)] = dict(q=q_J, mae_gain=mae_g, obj=(mae_g + a.screen_j_weight * q_J if a.screen_objective == "mae" else q_J), C=C, ok=bool(gate_ok and snr >= a.min_snr), snr=snr, effect=effect, L=L_,
                                                       mean_mm=float(C[masks[n]].mean() * 1000), far_mm=float(np.abs(C)[~wins[n]].max() * 1000))
                        return out

                    if a.long_screen == "model" and model is not None:
                        # ---- 모델 선별 (--long-screen model): Genesis 재생은 후보 하나에 25~40분이라 60분 예산에 5번 남짓밖에 못 잰다(E3_s1: 후보 18개 중 단일 몇 개 + 묶음 1번).
                        #      그래서 영역 기준으로 뽑은 관련 과거 행동(전체 이력에서) x 모든 변형 x 묶음(최대 seq_k 개)을 Param2Deform 롤아웃으로 먼저 훑는다(후보당 몇 초).
                        #      롤아웃 효과 C = 모델(변형) - 모델(원래)라서 모델의 누적 오차(영역 MAE 2.4mm @79푸시)는 상쇄된다. A@80 groove_top 의 Genesis 반사실 44개와 비교하면
                        #      효과 0.3mm 이상 11개 모두 부호 일치, 크기 중앙값 1.09배(0.5~2.4), 허경보 1/29. 모델은 순위만 정하고, 상위만 Genesis 로 재생해 control 게이트·SNR 을 통과해야 쓴다.
                        import itertools
                        import torch as _torch
                        from scipy.stats import spearmanr
                        from vs_planner import predict_next
                        dev_s = next(model.parameters()).device
                        t_scr, ctrl_m = time.time(), {}

                        def m_roll(start_, edits_):
                            """행동 start_ 직전 높이맵에서 #K 까지 모델로 굴린다. edits_: {행동 번호: 대체 행동}"""
                            s_ = _torch.as_tensor(before[start_ - 1], dtype=_torch.float32, device=dev_s)[None, None]
                            with _torch.no_grad():
                                for m_ in range(start_, K + 1):
                                    act_ = edits_[m_] if m_ in edits_ else actions[m_ - 1].tolist()
                                    s_ = predict_next(model, s_, _torch.as_tensor(act_, dtype=_torch.float32, device=dev_s))
                            return s_[0, 0].double().cpu().numpy()

                        def m_gain(members_):
                            """묶음의 모델 반사실 변화 C(m) 로 T = 현재 + 가중창 x C 를 만들었을 때의 영역 점수 이득(Genesis 반사실과 같은 식)"""
                            s0_ = members_[0][0]
                            if s0_ not in ctrl_m:
                                ctrl_m[s0_] = m_roll(s0_, {})
                            ed_ = {m_: variant_action(actions[m_ - 1], *cand_var[(m_, v_)]) for m_, v_ in members_}
                            C_ = m_roll(s0_, ed_) - ctrl_m[s0_]
                            h_n_ = h_cur + wsoft[n] * C_
                            _, v_T_, _ = _terms(h_n_, cfg)
                            q_J_ = float(region_J(n, v_now_s, cfg) - region_J(n, v_T_, cfg))
                            if a.screen_objective != "mae":
                                return q_J_
                            # 2026-11-16 --screen-objective mae: 영역 MAE 이득(mm) + 가중치 x J 이득. J 만 보면 위치·방향 편집이 영역 평균을 거의 못 움직이는 쪽으로 뽑힌다(cf_size_exp2: J 기준 영역 평균 -2퍼센트, MAE 기준 +15퍼센트)
                            mae_g_ = float((np.abs(h_cur - tgts[n])[masks[n]].mean() - np.abs(h_n_ - tgts[n])[masks[n]].mean()) * 1000)
                            return mae_g_ + a.screen_j_weight * q_J_

                        scr_vars = parse_cf_variants(a.screen_variants) if a.screen_variants else (cf_more + cf_less)
                        singles = []
                        for t_, w_, e_ in rows_all[:a.screen_actions]:
                            for vv in scr_vars:
                                cand_var[(t_ + 1, vv[0])] = (vv[1], vv[2])
                                singles.append((t_ + 1, vv[0]))
                        pred_m = {(c_,): m_gain([c_]) for c_ in singles}
                        top_s = sorted(singles, key=lambda c_: -pred_m[(c_,)])[:max(a.screen_pairs, 2)]
                        for k_ in range(2, max(a.seq_k, 2) + 1):
                            for comb in itertools.combinations(top_s, k_):
                                if len({m_[0] for m_ in comb}) == k_:      # 서로 다른 행동끼리만
                                    mem_ = tuple(sorted(comb))
                                    pred_m[mem_] = m_gain(list(mem_))
                        greedy_path = []
                        if a.screen_greedy >= 3:      # 탐욕적 전진 선택: 단일 최선에서 시작해 목적값을 가장 올리는 (다른 행동의) 편집을 하나씩 더한다. 모델 예측만 쓴다(후보당 수 초)
                            cur_m = [max(singles, key=lambda c_: pred_m[(c_,)])]
                            cur_g = pred_m[(cur_m[0],)]
                            greedy_path.append(tuple(cur_m))
                            for k_ in range(2, a.screen_greedy + 1):
                                used_ = {m_[0] for m_ in cur_m}
                                best_o = None
                                for c_ in singles:
                                    if c_[0] in used_:
                                        continue
                                    mem_ = tuple(sorted(cur_m + [c_]))
                                    if mem_ not in pred_m:
                                        pred_m[mem_] = m_gain(list(mem_))
                                    if best_o is None or pred_m[mem_] > best_o[1]:
                                        best_o = (c_, pred_m[mem_])
                                if best_o is None or best_o[1] < cur_g + a.screen_greedy_min:
                                    break
                                cur_m = sorted(cur_m + [best_o[0]])
                                cur_g = best_o[1]
                                greedy_path.append(tuple(cur_m))
                            print(f"    [②S 탐욕 묶음] 목적 {a.screen_objective}: " + " -> ".join(f"k={len(p_)} {pred_m[p_]:+.3f}" for p_ in greedy_path) + " | 최종 묶음 " + str([(m_[0], m_[1]) for m_ in greedy_path[-1]]), flush=True)
                        ranked = sorted(pred_m.items(), key=lambda kv: -kv[1])
                        print(f"[②S 모델 선별] 영역 '{n}': 관련 행동 {len(rows_all)}개(#{min(w_of)}~#{max(w_of)}) 중 상위 {min(a.screen_actions, len(rows_all))}개 x 변형 {len(scr_vars)}종 = 단일 {len(singles)}개 + 묶음 {len(pred_m) - len(singles)}개를 "
                              f"모델로 훑음({time.time() - t_scr:.0f}초). 예측 이득 상위: " + "; ".join(f"{[(m_[0], m_[1]) for m_ in mem]} {g_:+.3f}" for mem, g_ in ranked[:5]), flush=True)
                        if a.screen_model_only:      # 2026-11-17 --screen-model-only: Genesis 재생 없이 모델 반사실만 쓴다. 모델 C = 모델(변형) - 모델(원래)라서 모델의 누적 오차는 상쇄된다(위 주석: A@80 Genesis 반사실 44개와 부호 일치)
                            top_mo = next(((mem_, g_) for mem_, g_ in ranked if g_ > a.eps), None)
                            if top_mo is not None:
                                mem, g_m = top_mo
                                s0_ = mem[0][0]
                                if s0_ not in ctrl_m:
                                    ctrl_m[s0_] = m_roll(s0_, {})
                                ed_ = {m_: variant_action(actions[m_ - 1], *cand_var[(m_, v_)]) for m_, v_ in mem}
                                C_ = m_roll(s0_, ed_) - ctrl_m[s0_]
                                _, v_T_, _ = _terms(h_cur + wsoft[n] * C_, cfg)
                                q_J_ = float(region_J(n, v_now_s, cfg) - region_J(n, v_T_, cfg))
                                top_m = max(mem, key=lambda m_: abs(w_of.get(m_[0], 0.0)))
                                if q_J_ > a.eps and (cf_best[n] is None or q_J_ > cf_best[n]["q"]):
                                    cf_best[n] = dict(start=top_m[0], variant="seq" + str(len(mem)), C=C_, q=q_J_, members=[m_[0] for m_ in mem])
                                print(f"[②S 모델 전용] 영역 '{n}': 모델 반사실 {[(m_[0], m_[1]) for m_ in mem]} 모델 영역 점수 이득 {q_J_:+.3f} (예측 목적값 {g_m:+.3f}, Genesis 재생 없음, |C| 최대 {float(np.abs(C_).max() * 1000):.2f}mm)", flush=True)
                            else:
                                print(f"[②S 모델 전용] 영역 '{n}': 유효한 반사실 없음(모델 예측 이득이 eps 이하)", flush=True)
                            continue
                        verify, spent_s, nw_, acts_used = [], 0.0, max(len(replay_workers), 1), 0
                        budget_s = a.seq_budget_min * 60.0 if a.seq_budget_min > 0 else 1e18
                        for mem, g_ in ranked:
                            if g_ <= a.eps or len(verify) >= a.screen_verify:
                                break
                            n_act = K - mem[0][0] + 1
                            if a.screen_max_len > 0 and n_act > a.screen_max_len:
                                continue      # 재생이 너무 긴 후보는 예측 이득 1위라도 Genesis 로 검증하지 않는다(K=190 하트처럼 한 재생이 1시간 반을 넘을 때)
                            c_cost = n_act * a.replay_sec_per_action / nw_
                            if verify and spent_s + c_cost > budget_s:
                                break
                            if verify and a.screen_budget_actions > 0 and acts_used + n_act > a.screen_budget_actions:
                                continue      # 재생 행동 수 예산을 넘는 긴 재생은 건너뛰고 더 짧은 후보를 찾는다(예측 이득 1위는 위에서 항상 검증된다)
                            verify.append(list(mem))
                            spent_s += c_cost
                            acts_used += n_act
                        if len(greedy_path) >= 2:      # 탐욕 묶음이 있으면 검증은 크기별로: 가장 큰 묶음, 가운데 크기, 앞 3개 묶음(앞쪽 묶음은 뒤쪽 묶음의 부분집합이라 예측 순위로 고르면 같은 계열만 남는다). --screen-verify 가 상한
                            verify, acts_used, seen_g = [], 0, set()
                            for mem in (greedy_path[-1], greedy_path[len(greedy_path) // 2], greedy_path[min(2, len(greedy_path) - 1)]):
                                if mem in seen_g or len(verify) >= max(a.screen_verify, 1):
                                    continue
                                seen_g.add(mem)
                                if pred_m[mem] > a.eps:
                                    verify.append(list(mem))
                                    acts_used += K - mem[0][0] + 1
                        print(f"    [②S 모델 선별] Genesis 검증 후보 {len(verify)}개, 재생 행동 합 {acts_used}" + (f" (예산 {a.screen_budget_actions})" if a.screen_budget_actions > 0 else "") + f" | control={'기록(검증 길이 ' + str(a.ctrl_valid_len) + ' 이어받음)' if a.ctrl_valid_len > 0 else a.control}", flush=True)
                        res_v, best_v, gm_pairs = (effects(verify) if verify else {}), None, []
                        for mem in verify:
                            r_, g_m = res_v.get(tuple(mem)), pred_m[tuple(mem)]
                            if r_ is None:
                                print(f"    검증 {[(m_[0], m_[1]) for m_ in mem]}: 재생 실패", flush=True)
                                continue
                            gm_pairs.append((g_m, r_["obj"]))
                            print(f"    검증 {[(m_[0], m_[1]) for m_ in mem]}: 모델 예측 이득 {g_m:+.3f} -> Genesis 실측 {r_['obj']:+.3f} (J 이득 {r_['q']:+.3f}, MAE 이득 {r_['mae_gain']:+.2f}mm, 영역 평균 변화 {r_['mean_mm']:+.2f}mm, 먼 곳 {r_['far_mm']:.1f}mm, |C| {r_['effect']:.2f}mm, SNR {r_['snr']:.0f}, 재생 {r_['L']}행동) {'사용 가능' if r_['ok'] else '게이트/SNR 불통과'}", flush=True)
                            cf_table.append(dict(region=n, start=mem[0][0], variant="screen" + str(len(mem)), kind="seq", members=[m_[0] for m_ in mem], replay_len=r_["L"], effect_abs_mm=r_["effect"], snr=r_["snr"],
                                                 effect_region=r_["q"], q=r_["q"], control_ok=r_["ok"], region_mean_change_mm=r_["mean_mm"], far_change_mm=r_["far_mm"], predicted=g_m, screened=True))
                            if r_["ok"] and r_["q"] > a.eps and (best_v is None or r_["obj"] > best_v[1]["obj"]):      # 자격은 J 이득 > eps, 여럿이면 --screen-objective 값이 큰 쪽(J 모드에서는 예전과 같다)
                                best_v = (mem, r_)
                        rho_s = float(spearmanr([g for g, _ in gm_pairs], [q for _, q in gm_pairs])[0]) if len(gm_pairs) >= 4 else float("nan")
                        if best_v is not None and (cf_best[n] is None or best_v[1]["q"] > cf_best[n]["q"]):
                            mem, r_ = best_v
                            top_m = max(mem, key=lambda m_: abs(w_of.get(m_[0], 0.0)))
                            cf_best[n] = dict(start=top_m[0], variant="seq" + str(len(mem)), C=r_["C"], q=r_["q"], members=[m_[0] for m_ in mem])
                        print(f"[②S 모델 선별] 영역 '{n}' 종료: Genesis 검증 {len(verify)}개({(time.time() - t_scr) / 60:.0f}분, 모델 훑기 포함)" + (f", 모델-Genesis 순위상관 {rho_s:+.2f}" if rho_s == rho_s else "")
                              + (f" | 최선: {[(m_[0], m_[1]) for m_ in best_v[0]]} 영역 점수 이득 {best_v[1]['q']:+.3f}" if best_v else " | 유효한 반사실 없음"), flush=True)
                        continue

                    print(f"[②S 묶음] 영역 '{n}': 관련 행동 {len(rows_all)}개(#{min(w_of)}~#{max(w_of)}) 중 상위 {len(cand_var)}개 후보 " + ", ".join(f"#{c[0]}({c[1]})" for c in cand_var)
                          + f" | k={a.seq_k}, " + (f"시간 예산 {a.seq_budget_min:g}분" if a.seq_budget_min > 0 else f"묶음 재생 {a.seq_budget}회") + f" | control={a.control}", flush=True)
                    eff1 = effects([[c] for c in cand_var])                    # 단일 개입 이득 g1 (재생으로 잰 값)
                    g1 = {c: (eff1[(c,)]["q"] if eff1.get((c,)) and eff1[(c,)]["ok"] else -1.0) for c in cand_var}
                    Cs = {c: eff1[(c,)]["C"] for c in cand_var if eff1.get((c,))}
                    for c in cand_var:
                        e1 = eff1.get((c,))
                        print(f"    단일 #{c[0]}({c[1]}): " + (f"영역 점수 이득 {e1['q']:+.3f} (|C| {e1['effect']:.2f}mm, SNR {e1['snr']:.0f}, 재생 {e1['L']}행동) {'사용 가능' if e1['ok'] else '게이트/SNR 불통과'}" if e1 else "재생 실패"), flush=True)

                    def overlap(c1, c2):                                       # 두 개입이 같은 곳을 바꾸는 정도 0~1 (단일 반사실 변화 |C| 의 창 안 겹침)
                        if c1 not in Cs or c2 not in Cs:
                            return 0.0
                        x1, x2 = np.abs(Cs[c1])[wins[n]], np.abs(Cs[c2])[wins[n]]
                        return float(np.minimum(x1, x2).sum() / (np.maximum(x1, x2).sum() + 1e-12))

                    got = {}

                    def evaluate(b):
                        members = sorted(b)
                        r_ = effects([members])[tuple(members)]
                        got[frozenset(b)] = r_
                        return float(r_["q"]) if (r_ is not None and r_["ok"]) else -1.0

                    cost = lambda b: (K - min(m_[0] for m_ in b) + 1) * a.replay_sec_per_action
                    out_s = sq.search(list(cand_var), g1, evaluate, overlap, k=a.seq_k, budget=a.seq_budget, cost=cost,
                                      seconds=(a.seq_budget_min * 60.0 if a.seq_budget_min > 0 else None), pool=a.seq_pool)
                    for b_, true_, pred_ in out_s["measured"]:
                        print(f"    묶음 {sorted(b_)}: 실측 이득 {true_:+.3f} (단일 이득 합 기반 예측 {pred_:+.3f})", flush=True)
                        r_ = got.get(b_)
                        if r_ is not None:
                            cf_table.append(dict(region=n, start=min(m_[0] for m_ in b_), variant="seq" + str(len(b_)), kind="seq", members=sorted(m_[0] for m_ in b_), replay_len=r_["L"],
                                                 effect_abs_mm=r_["effect"], snr=r_["snr"], effect_region=r_["q"], q=r_["q"], control_ok=r_["ok"], region_mean_change_mm=r_["mean_mm"], far_change_mm=r_["far_mm"],
                                                 predicted=pred_))
                    best_b, best_gain = out_s["best"]
                    members_b = sorted(best_b)
                    rb = eff1.get(tuple(members_b)) if len(members_b) == 1 else got.get(best_b)
                    print(f"[②S 묶음] 영역 '{n}' 종료: 묶음 재생 {out_s['n_replays']}회({(time.time() - t_seq) / 60:.0f}분) | 최선 {members_b} 이득 {best_gain:+.3f} | 단일 최선 이득 {out_s['single_best'][1]:+.3f}", flush=True)
                    if rb is not None and rb["ok"] and best_gain > a.eps and (cf_best[n] is None or best_gain > cf_best[n]["q"]):
                        top_m = max(members_b, key=lambda m_: abs(w_of.get(m_[0], 0.0)))
                        cf_best[n] = dict(start=top_m[0], variant="seq" + str(len(members_b)), C=rb["C"], q=best_gain, members=[m_[0] for m_ in members_b])
            except Exception as e_:      # 이 단계가 실패해도 나머지 탐색은 계속한다
                import traceback
                if a.strict:
                    raise
                print(f"    (②S 묶음 탐색 실패, 건너뜀: {e_!r})\n{traceback.format_exc()}", flush=True)
        (a.out / "cf_replay.json").write_text(json.dumps(cf_table, ensure_ascii=False, indent=1))
        for n in names:
            b = cf_best[n]
            print(f"    => '{n}': " + (f"과거 #{b['start']}({b['variant']}) 반사실 사용 (q={b['q']:+.3f})" if b else "유효한 반사실 없음(abstain) -> 현재 상태에서의 후보 평가만 한다"), flush=True)

    # ---- ②d 선형(가산) 반사실: 재생 없이, 기록된 높이 변화로 '그 행동을 다르게 했다면'을 근사한다 (cf_best 에 반사실 목표 T 와 CFR 시드로 들어간다)
    if a.lin_cf > 0:
        try:
            _, v_now_l, _ = _terms(h_cur, cfg)
            for n in names:
                dd = dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px)
                dd.sort(key=lambda r: -abs(r[1]) * r[2])
                for t_l, w_l, e_l in dd[:a.lin_cf]:
                    dH = (after[t_l] - before[t_l]).astype(np.float64)
                    facs = (("skip", -1.0), ("half", -0.5)) if w_l < 0 else (("again", 1.0), ("more", 0.5))
                    for vn, fac in facs:
                        C_l = fac * dH
                        T_l = h_cur + wsoft[n] * C_l
                        _, vT_l, _ = _terms(T_l, cfg)
                        eff_l = region_J(n, v_now_l, cfg) - region_J(n, vT_l, cfg)
                        lag = len(actions) - t_l - 1
                        cf_table.append(dict(region=n, start=t_l + 1, variant=f"lin_{vn}", kind="linear", lag=lag, q=eff_l, region_mean_change_mm=float(C_l[masks[n]].mean() * 1000)))
                        print(f"    [선형 반사실] 영역 '{n}' 과거 #{t_l + 1}(현재 {lag}개 전, w={w_l:+.2f}) {vn:<5}: 영역 평균 변화 {float(C_l[masks[n]].mean() * 1000):+.2f}mm | 현재에 더하면 영역 점수 개선 {eff_l:+.3f}", flush=True)
                        # 선형 근사는 Genesis 시뮬레이션이 아니므로 반사실 결과(cf_best)로 쓰지 않는다. 어떤 행동·변형을 Genesis 로 재생할지 고르는 선별 용도로만 출력한다.
            print("    (선형 근사는 선별 전용입니다. 반사실 효과는 Genesis 재생(②, ②L)으로만 확정합니다.)", flush=True)
        except Exception as e_l2:
            print(f"    (선형 반사실 실패, 건너뜀: {e_l2!r})", flush=True)
    replay_sec = time.time() - t_replay0                     # ②·②L·②T·②S 과거 재생에 쓴 벽시계 초(캐시가 있으면 짧다). 시간표용
    log, carry, stuck, commits, visited = [], {n: [] for n in names}, set(), 0, set()
    h_start = h_cur.copy()
    T_fixed, dT_hist = {}, {n_: [] for n_ in names}        # --cf-target fixed: 영역별 고정 T, 확정마다의 T 거리(mm)
    T_blob = {}                                            # --t-metric blob: 영역별 고정 덩어리(T 를 고정한 시점의 |T-현재| > 문턱)
    s_orig0 = score(h_cur, cfg_orig) if cfg_orig is not None else None
    rcount, done_reg = {n: 0 for n in names}, set()           # 영역별 확정 횟수 / 더 확정하지 않을 영역(--per-region-commits, --stop-progress)
    goal_move = {n: float((tgts[n] - h_ref)[masks[n]].mean()) for n in names}

    def progress(h, n_):
        """영역 목표 진행률 = (영역 평균 높이 변화) / (영역 목표가 요구한 변화). 기준 상태 h_ref 에서 잰다. 목표 변화가 0.1mm 미만이면 정의하지 않음(None)."""
        if abs(goal_move[n_]) < 1e-4:
            return None
        return float((h - h_ref)[masks[n_]].mean()) / goal_move[n_]
    skip = {x for x in a.skip_regions.split(",") if x} & set(names)      # 영역 오차는 J 에 계속 들어가지만(손상 감시) 이 영역을 고치는 라운드는 하지 않는다
    fam_stats = {}          # 계열 -> [현재보다 J 가 좋아진 수, 물리 평가 수] (라운드를 넘어 누적, 적응형 쿼터용)
    t_round = 0
    while commits < a.max_commits and len(stuck | skip | done_reg) < len(names) and (time.time() - t_start) / 60 < a.budget_min:
        t_round += 1
        t_round0 = time.time()
        eval_seq[0] = 0
        stage_sec.clear()
        stage_cnt.clear()
        rho_info.clear()
        feat_used.clear()
        pair_budget_r[0] = None
        todo = [n for n in names if n not in stuck and n not in skip and n not in done_reg]
        n = max(todo, key=lambda k: 0.5 * sum(s_cur["per"][k]))
        if a.own_score:                                            # 이 라운드의 점수 = 영역 n 하나의 점수 (워커에도 알린다)
            cfg["active"] = n
            for w_ in tqs:
                tqs[w_].put(("active", n))
            s_cur = score(h_cur, cfg, h_cur)
        if a.credit_side > 0:      # 이 라운드의 '다른 요청 영역' 기준값(현재 상태)과 활성 영역 J
            credit_state["side"] = [k_ for k_ in names if k_ != n and k_ not in skip_cf]
            credit_state["base"] = {k_: float(region_J(k_, s_cur["vis"], cfg)) for k_ in credit_state["side"]}
            credit_state["J0"] = float(s_cur["J"])
            print(f"    [다중 영역] 활성 '{n}' 말고 요청한 영역 {len(credit_state['side'])}개의 이득도 센다(x{a.credit_side:g}, 영역별 악화 한도 {a.side_worsen_tol:g}, 활성 악화 한도 {a.active_slack:g})", flush=True)
        print(f"\n[라운드 {t_round}] 영역 '{n}' [{label[n]}] (현재 MAE/RMSE {s_cur['per'][n][0]:.2f}/{s_cur['per'][n][1]:.2f}mm)  확정 {commits}회, 경과 {(time.time() - t_start) / 60:.0f}분", flush=True)
        # ---- G1 과거 행동 전부 + 사전(제한 최소제곱)
        dic = dictionary(actions, before, after, fps, masks[n], tgts[n], h_cur, near_px, win_px)
        dic.sort(key=lambda r: -abs(r[1]) * r[2])
        if t_round <= len(names):
            (a.out / f"past_analysis_{n}.json").write_text(json.dumps([dict(action=t + 1, w=w, effect_rms_mm=e) for t, w, e in dic], ensure_ascii=False, indent=1))
        print("    과거 행동 반사실 가중치 상위(번호, w, 영역 영향): " + ", ".join(f"#{t + 1}(w={w:+.2f},{e:.1f})" for t, w, e in dic[:6]), flush=True)
        cands, meta = [], {}

        def add(act, fam, base=None, kind=None, t=None):
            act = clampA(act)
            k = akey(act)
            if k not in meta:
                meta[k] = dict(family=fam, base=base, kind=kind, t=t)
                cands.append((k, act))

        first = (n not in visited) or a.mode == "beam"
        visited.add(n)
        for rank, (t, w, e) in enumerate(dic if not a.no_hints else []):         # --no-hints: 과거 행동 변형(G1) 없음
            if rank < a.g1_top:
                kinds = [(0.5, 0), (0.5, 4), (0.5, -4), (1.0, 4), (1.0, -4), ("rev", 0), (1.0, 0)] if first else [(0.5, 0), (0.5, 4), (0.5, -4)]
            elif first:
                kinds = [(1.0, 0), (0.5, 4)] if w > 0.1 else [(0.5, 0), (0.5, 4), (0.5, -4)]
            else:
                kinds = []
            for kd in kinds:
                add(variants_of(actions[t].tolist(), kd), "G1", t + 1, kd, t)
        # ---- G2 / G3
        for act in transport_candidates(geo, h_cur, tgts[n], masks[n]):
            add(act, "G2")
        for act in flatten_candidates(geo, h_cur, tgts[n], masks[n]):
            add(act, "G3")
        sq_pred, sq_seqs = {}, []
        if a.squeeze > 0:      # 짜내기: 구덩이 둘레를 눌러 점토가 구덩이로 밀려 올라오게(여드름 짜듯이). 보호(건너뛴) 영역은 누르지 않는다
            try:
                import cf_transport as ctp
                avoid_m = np.zeros((SIZE, SIZE), bool)
                for k_ in (skip_cf if (a.avoid_protected or a.protect_abs) else []):      # 기본은 사전 필터 없음(2026-11-16: A 에서 짜내기 후보 260개 중 220개가 이 필터로 평가도 못 받고 빠졌다)
                    avoid_m |= masks[k_]
                if model is not None:      # 넓은 격자(8방향 x 길이 3 x 깊이 7)를 모델로 예측해 영역 J 가 가장 좋아질 것으로 보이는 몇 개만 Genesis 평가에 올린다(필요한 세기를 계산으로 정함)
                    sq_c, sq_info = ctp.squeeze_candidates(h_cur, masks[n], tgts[n], geo.g, BOUNDS, clampA, avoid=avoid_m, keep_angles=8, lengths_mm=(5.0, 8.0, 12.0), depths_mm=(2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 22.0, 26.0))
                    if sq_c:
                        pj_sq = model_rank_J(model, h_cur, cfg, n, sq_c)
                        sel_sq = pick_diverse_by_pred(sq_c, pj_sq, a.squeeze)
                        sq_info.update(grid=len(sq_c), pred_J_best=round(float(min(pj_sq)), 3), pred_J_now=round(float(s_cur["J"]), 3))
                        sq_pred.update({akey(sq_c[i_]): pj_sq[i_] for i_ in sel_sq})
                        sq_c = [sq_c[i_] for i_ in sel_sq]
                else:
                    sq_c, sq_info = ctp.squeeze_candidates(h_cur, masks[n], tgts[n], geo.g, BOUNDS, clampA, avoid=avoid_m)
                for act in sq_c:
                    add(act, "SQ")
                for act in sq_c[:max(2, a.squeeze // 2)]:      # 양옆 짝: 구덩이 반대편에서도 눌러(시퀀스 2개 푸시) 가운데로 몰아 솟게 한다. PLANSEQ 로 Genesis 평가
                    ck_ = tuple(round(float(v), 6) for v in act)
                    cen_ = sq_info.get("centers", {}).get(ck_)
                    opp_ = ctp.opposite_press(act, cen_, geo.g, clampA, avoid=avoid_m) if cen_ is not None else None
                    if opp_ is not None:
                        sq_seqs.append([[float(v) for v in act], [float(v) for v in opp_]])
                sq_info["two_sided"] = len(sq_seqs)
                sq_info.pop("centers", None)
                feat_used["sq"] = sq_info
                print(f"    [짜내기] {sq_info}", flush=True)
            except Exception as e_sq:
                if a.strict:
                    raise
                print(f"    (짜내기 실패, 건너뜀: {e_sq!r})", flush=True)
        if a.ot_donors > 0:      # 점토 수송 계획(cf_transport): 필요한 점토를 어디서 가져올지 최적 수송으로 정하고 그 흐름을 따라 미는 후보
            try:
                import cf_transport as ctp
                ot_c, ot_info = ctp.round_candidates(h_cur, n, names, masks, tgts, [k_ for k_ in names if k_ not in skip_cf], geo.g, BOUNDS, clampA, rmask,
                                                     k_flows=a.ot_donors, depth_mm=a.ot_depth_mm, relief_weight=a.ot_relief_weight)
                for act in ot_c:
                    add(act, "OT")
                feat_used["ot"] = ot_info
                print(f"    [수송 계획] {ot_info}", flush=True)
            except Exception as e_ot:
                if a.strict:
                    raise
                print(f"    (수송 계획 실패, 건너뜀: {e_ot!r})", flush=True)
        for act in carry[n]:
            add(act, "carry")
        # ---- CFR: 과거 반사실(C*)이 알려 준 '그때 달랐다면' 상태 T 를 현재 점토에서 만들기 위해, 그 과거 행동을 되돌리는 후보
        tinfo, dT0 = None, 1.0
        cfb = cf_best.get(n)
        if cfb is not None:
            if a.cf_target == "fixed":
                if n not in T_fixed:        # 반사실 세계를 처음 한 번만 고정: 매 라운드 현재 + C 로 다시 만들면 이미 판 만큼 T 가 또 깊어지는 이중 계산이 된다
                    T_fixed[n] = (h_cur + wsoft[n] * cfb["C"]).astype(np.float32)
                Tn = T_fixed[n]
            else:
                Tn = (h_cur + wsoft[n] * cfb["C"]).astype(np.float32)
            win_t = wins[n]
            if a.t_metric == "blob":
                if n not in T_blob:
                    T_blob[n] = (np.abs(Tn - h_cur) * 1000.0 > a.t_blob_mm) & wins[n]
                if T_blob[n].sum() >= 20:
                    win_t = T_blob[n]
            dT0 = max(float(np.abs(h_cur - Tn)[win_t].mean() * 1000), 1e-6)
            tinfo = dict(T=Tn, win=win_t)
            # 사용한 반사실을 파일로 남긴다: T = 영역 창에서만 바뀐 목표, H_cf = 과거를 바꿔 끝까지 재생한 세계(현재 + C). 전/후 그림에 '과거를 바꾼 세계'로 함께 비교한다.
            (a.out / "cf_replay").mkdir(exist_ok=True)
            np.save(a.out / "cf_replay" / f"T_{n}.npy", Tn)
            np.save(a.out / "cf_replay" / f"H_cf_{n}.npy", ((h_start if a.cf_target == "fixed" else h_cur) + cfb["C"]).astype(np.float32))
            print(f"    반사실 목표 T: 과거 #{cfb['start']}({cfb['variant']})가 달랐다면 J {s_cur['J']:.3f} -> {score(Tn.astype(np.float64), cfg)['J']:.3f}. 현재-T 창 안 평균 거리 {dT0:.2f}mm", flush=True)
            for act in reverse_candidates(actions[cfb["start"] - 1].tolist()):
                add(act, "CFR", cfb["start"])
            if a.cf_target == "fixed":      # T 방향 후보: T 가 현재보다 낮은 곳(더 파야 하는 곳)은 누르고, 높은 곳은 점토를 옮긴다 (요청 영역 목표가 아니라 반사실 세계가 기준)
                for act in transport_candidates(geo, h_cur, Tn.astype(np.float64), wins[n], n_blobs=a.trk_blobs, thr=a.trk_thr_mm * 1e-3):
                    add(act, "TRK")
                for act in flatten_candidates(geo, h_cur, Tn.astype(np.float64), wins[n], n_blobs=a.trk_blobs, thr=a.trk_thr_mm * 1e-3):
                    add(act, "TRK")
            if a.hint_match > 0 and model is not None:      # 힌트 맞춤(HM): D = T - 현재 를 모델 예측으로 가장 잘 맞추는 푸시(세기까지)를 골라 올린다
                try:
                    import cf_transport as ctp
                    avoid_h = np.zeros((SIZE, SIZE), bool)
                    for k_ in (skip_cf if (a.avoid_protected or a.protect_abs) else []):
                        avoid_h |= masks[k_]
                    hm_c, hm_info = ctp.hint_match_candidates(h_cur, Tn.astype(np.float64), wins[n], geo.g, BOUNDS, clampA, avoid=avoid_h)
                    if hm_c:
                        pj_hm = model_rank_match(model, h_cur, Tn, wins[n], hm_c, blob=(tinfo["win"] if a.t_metric == "blob" else None))
                        sel_hm = pick_diverse_by_pred(hm_c, pj_hm, a.hint_match)
                        hm_info.update(grid=len(hm_c), pred_dT_best=round(float(min(pj_hm)), 3), pred_dT_picked=round(float(min(pj_hm[i_] for i_ in sel_hm)), 3), dT_now=round(float(dT0), 3), picked=len(sel_hm))
                        for i_ in sel_hm:
                            add(hm_c[i_], "HM")
                    feat_used["hm"] = hm_info
                    print(f"    [힌트 맞춤] D=T-현재 덩어리와 모델 예측(T 까지 거리 mm): {hm_info}", flush=True)
                except Exception as e_hm:
                    if a.strict:
                        raise
                    print(f"    (힌트 맞춤 실패, 건너뜀: {e_hm!r})", flush=True)
        if a.deep and not a.no_hints:                        # 깊게 누르는 후보: 영향 큰 과거 행동을 깊이 x1.5, x2 로 (clampA 가 플래너 안전 상한 30 mm 로 자른다)
            for t_d, w_d, e_d in dic[:4]:
                x_d, y_d, th_d, L_d, d_d = [float(v) for v in actions[t_d]]
                for df_ in (1.5, 2.0):
                    add([x_d, y_d, th_d, L_d, d_d * df_], "DEEP", t_d + 1)
        plan_seqs = list(sq_seqs)                            # 양옆 짝 시퀀스(짜내기)도 PLANSEQ 평가에 포함
        # PLANSEQ: 계획기가 짠 계획의 앞 K개 행동 묶음 (--plan-seq)
        if model is not None and a.plan_cands > 0:           # 계획기가 요청 영역의 목표를 향해 제안하는 첫 행동 (영역에서 멀리 떨어진 곳도 가능)
            try:
                import torch as _torch
                from vs_planner import greedy_plan_gpu
                dev_p = next(model.parameters()).device
                cur_p = _torch.as_tensor(h_cur, dtype=_torch.float32, device=dev_p)[None, None]
                if a.cf_target == "fixed" and tinfo is not None:      # T-추적: 계획기 목표 = 현재에서 창 안만 T 로 바꾼 맵
                    tmap_p = h_cur.copy()
                    tmap_p[wins[n]] = tinfo["T"].astype(np.float64)[wins[n]]
                    tgt_p = _torch.as_tensor(tmap_p, dtype=_torch.float32, device=dev_p)[None, None]
                elif a.plan_refine > 0:      # 반사실 목표 T 가 없을 때도 모델이 고치는 영역 하나만 목표로 삼는다(나머지는 현재 그대로 = 부수 변화는 모델 손실에서 불리). 기본(--plan-refine 0)은 예전 그대로 전체 목표 맵
                    tmap_p = h_cur.copy()
                    tmap_p[masks[n]] = np.asarray(tgts[n], dtype=np.float64)[masks[n]]
                    tgt_p = _torch.as_tensor(tmap_p, dtype=_torch.float32, device=dev_p)[None, None]
                else:
                    tgt_p = _torch.as_tensor(hstar_all, dtype=_torch.float32, device=dev_p)[None, None]
                n_plan = 0
                n_ref_p, ref_fail_p = 0, 0
                horizon_p = max(3, a.plan_seq)
                for _ in range(a.plan_cands):
                    plan_p = greedy_plan_gpu(model, cur_p, tgt_p, horizon=horizon_p, candidates=64, w3d=1.0, wviz=1.0, recent_actions=None, candidate_batch_size=1)
                    add([float(v) for v in plan_p[0].detach().cpu()], "PLAN")
                    if a.plan_refine > 0:      # 모델이 계획 전체(여러 푸시)를 한 롤아웃으로 보고 목표 쪽으로 다듬는다: 직접 파는 푸시인지, 근처를 눌러 올리는 푸시인지, 연쇄인지를 모델이 정한다
                        try:
                            from vs_planner import refine_plan_gpu
                            plan_p = refine_plan_gpu(model, cur_p, tgt_p, plan_p, w3d=1.0, wviz=1.0, recent_actions=None, max_refine_actions=horizon_p, refine_steps=a.plan_refine)
                            add([float(v) for v in plan_p[0].detach().cpu()], "PLAN")
                            n_ref_p += 1
                        except Exception as e_ref:      # 다듬기 실패는 greedy 계획으로 계속한다(기록은 남긴다)
                            ref_fail_p += 1
                            print(f"    (계획 다듬기 실패, greedy 계획으로 계속: {e_ref!r})", flush=True)
                    if a.plan_seq >= 2:
                        seq_p = [clampA([float(v) for v in plan_p[i_].detach().cpu()]) for i_ in range(min(a.plan_seq, plan_p.shape[0]))]
                        if len(seq_p) >= 2 and all(akey(seq_p[i_]) != akey(seq_p[i_ + 1]) for i_ in range(len(seq_p) - 1)):
                            plan_seqs.append(seq_p)
                    n_plan += 1
                if a.plan_refine > 0:
                    feat_used["plan_refine"] = dict(steps=a.plan_refine, refined=n_ref_p, failed=ref_fail_p, horizon=horizon_p)
                print(f"    계획기 제안(PLAN) {n_plan}회 시도" + (f", 모델 기울기로 다듬음 {n_ref_p}회(실패 {ref_fail_p})" if a.plan_refine > 0 else "") + (f", 시퀀스(PLANSEQ, 앞 {a.plan_seq}개 행동 묶음) {len(plan_seqs)}개" if a.plan_seq >= 2 else ""), flush=True)
            except Exception as e_p:
                print(f"    (계획기 제안 실패, 건너뜀: {e_p!r})", flush=True)
        mdl_pred = {}
        if model is not None:
            t_m = time.time()
            for act, pj in model_candidates(model, h_cur, cfg, n, tgts[n], masks[n], geo, top=a.mdl_top, deep=a.deep):
                add(act, "MDL")
                mdl_pred[akey(clampA(act))] = pj
            print(f"    모델 기반 제안(MDL) {len(mdl_pred)}개 ({time.time() - t_m:.0f}초, 예측 J 최저 {min(mdl_pred.values()) if mdl_pred else float('nan'):.3f} / 영역 현재 J 기준 1.0 x 항 수)", flush=True)
        fams = {f: sum(1 for v in meta.values() if v["family"] == f) for f in ("G1", "G2", "G3", "carry", "CFR", "MDL", "PLAN", "DEEP", "TRK", "HM", "OT", "SQ")}
        print(f"    1단계 후보 {len(cands)}개 {fams}", flush=True)
        n_all = len(cands)
        # ---- πBO 사전: 과거 반사실 변화 C* 와 지금 필요한 변화(목표-현재)가 같은 방향인 곳을 지나는 푸시일수록 먼저 시험한다 (반사실이 없으면 만들지 않는다)
        prior_raw = None
        if a.pibo and cfb is not None:
            A_s = E.smooth_for_tool(E.alignment_map(cfb["C"], tgts[n] - h_cur, wsoft[n]), geo.g["tool_r_px"])

            def prior_raw(act_):
                rows_, cols_ = fb.path_points(geo.g, act_)
                return E.prior_values(A_s, rows_, cols_)
            print(f"    πBO 사전: 반사실 #{cfb['start']}({cfb['variant']}) 변화와 지금 필요한 변화가 같은 방향인 곳 {int((A_s > 0.05).sum())}화소, 초기 지수 {E.pibo_gamma(a.pibo_beta, 0):.2f}", flush=True)
        if a.mode == "beam":
            # 대리 점수: 과거 기록 dH_t(그 행동이 실제로 만든 높이 변화)를 현재 상태에 (옆으로 밀어서, 깊이 비율만큼) 더한 높이의 J
            mm = geo.mm_px()
            for k, act in cands:
                mt = meta[k]
                mt["proxy"] = None
                if mt["family"] == "G1" and mt["kind"][0] != "rev":
                    ds, sh = mt["kind"]
                    fac = ds if sh != 0 or ds != 1.0 else 0.5
                    th = actions[mt["t"]][2]
                    dH = (after[mt["t"]] - before[mt["t"]]).astype(np.float64)
                    if sh:
                        dH = ndimage.shift(dH, (sh * mm * math.cos(th), -sh * mm * math.sin(th)), order=1, mode="constant")
                    mt["proxy"] = score(h_cur + fac * dH, cfg)["J"]
            quota = {q.split(":")[0]: int(q.split(":")[1]) for q in a.quota.split(",")}
            if a.adaptive_quota and fam_stats:
                quota, pf = adaptive_quota(quota, fam_stats)
                print("    계열별 성공률(개선/평가) -> 추정 성공률 -> 이번 쿼터: " + ", ".join(
                    f"{f} {fam_stats.get(f, [0, 0])[0]}/{fam_stats.get(f, [0, 0])[1]}->{pf[f]:.2f}->{quota[f]}" for f in quota), flush=True)
            sel, per_base = [], {}
            use_pr = prior_raw is not None
            gam0 = E.pibo_gamma(a.pibo_beta, 0) if use_pr else 0.0
            pri = {}
            if use_pr:
                pri = {c[0]: float(p_) for c, p_ in zip(cands, E.normalize_prior([prior_raw(c[1]) for c in cands]))}

            def g1_key(c):                      # 기본: 대리 점수 J 가 낮은 순. πBO: (예상 개선) x 사전^gamma 가 큰 순
                m_ = meta[c[0]]
                if use_pr:
                    return -(max((s_cur["J"] - m_["proxy"]) if m_["proxy"] is not None else 0.0, 0.0) + 0.005) * pri[c[0]] ** gam0
                return (m_["proxy"] is None, m_["proxy"] or 0)
            g1 = sorted([c for c in cands if meta[c[0]]["family"] == "G1"], key=g1_key)
            for c in g1:
                b = meta[c[0]]["base"]
                if sum(1 for x in sel if meta[x[0]]["family"] == "G1") >= quota.get("G1", 0):
                    break
                if per_base.get(b, 0) < 2:
                    per_base[b] = per_base.get(b, 0) + 1
                    sel.append(c)
            for fam in ("G2", "G3", "carry", "CFR", "MDL", "PLAN", "DEEP", "TRK", "HM", "OT", "SQ"):
                pool = [c for c in cands if meta[c[0]]["family"] == fam]
                q = quota.get(fam, 0)
                if (use_pr or a.diverse) and len(pool) > q > 0:        # 사전 순위 + 지역 벌점(몰리지 않게)
                    F_ = np.array([E.act_feat(c[1]) for c in pool])
                    sc_ = np.array([pri[c[0]] ** gam0 if use_pr else 1.0 for c in pool])
                    sel += [pool[i_] for i_ in E.select_batch(F_, sc_, q, a.penalize_ell)]
                else:
                    step = max(len(pool) // max(q, 1), 1)
                    sel += pool[::step][:q]
            print(f"    빔(다양성 쿼터 {a.quota}): 후보 {n_all}개를 순위 매겨 {len(sel)}개만 물리 평가", flush=True)
            cands = sel
        res = run_eval(cands, tgt=tinfo, patience=a.patience, base_J=s_cur["J"], stage="beam")
        allres = {akey(r["action"]): r for r in res}
        if a.mode == "beam":
            pp = [(meta[k]["proxy"], r["J"]) for k, r in allres.items() if meta.get(k, {}).get("proxy") is not None]
            if len(pp) >= 4:
                from scipy.stats import spearmanr
                rho_info.update(g1_rho=float(spearmanr([x for x, _ in pp], [y for _, y in pp])[0]), g1_n=len(pp)); print(f"    대리 점수-물리 J 순위 상관(G1 {len(pp)}개): {spearmanr([x for x, _ in pp], [y for _, y in pp])[0]:+.2f}", flush=True)
        if sq_pred:
            pp3 = [(sq_pred[k_], r_["J"]) for k_, r_ in allres.items() if k_ in sq_pred and r_["J"] < 1e8]
            if len(pp3) >= 3:
                from scipy.stats import spearmanr
                rho_info.update(sq_rho=float(spearmanr([x for x, _ in pp3], [y for _, y in pp3])[0]), sq_n=len(pp3))
                print(f"    모델 예측 J - 물리 J 순위 상관(SQ {len(pp3)}개): {rho_info['sq_rho']:+.2f}", flush=True)
        if mdl_pred:
            pp2 = [(mdl_pred[k], r["J"]) for k, r in allres.items() if k in mdl_pred and r["J"] < 1e8]
            if len(pp2) >= 4:
                from scipy.stats import spearmanr
                rho_info.update(mdl_rho=float(spearmanr([x for x, _ in pp2], [y for _, y in pp2])[0]), mdl_n=len(pp2)); print(f"    모델 예측 J - 물리 J 순위 상관(MDL {len(pp2)}개): {spearmanr([x for x, _ in pp2], [y for _, y in pp2])[0]:+.2f}  (모델이 후보 순위를 얼마나 맞히나)", flush=True)
        # ---- 3단계' 적응형 정제(--refine-mode bads/spsa): 1단계에서 이미 눌러 본 후보(위치·방향·길이·깊이 -> 영역 J)로 대리모델을 맞춰 다음에 시험할 소수를 고른다.
        #      bads: 국소 가우시안 과정의 기대 개선 x πBO 사전, 묶음 안은 지역 벌점. spsa: 동시 섭동 두 방향. 실패하면 아래 패턴 탐색으로 대신한다.
        adaptive_failed = False
        if a.refine_mode in ("bads", "spsa"):
            try:
                rng_r = np.random.default_rng(20261116 + t_round + 1000 * a.seed)
                # 2026-11-16 --refine-t-weight: 정제 목적값 = J + 가중치 x dT(후보 푸시 뒤 높이맵과 반사실 T 사이 창 평균 거리 mm). T 가 없으면 J 그대로(예전 동작).
                #   대리모델 맞춤, 기대 개선의 기준값, 시작점 순위, 신뢰영역 성공 판정, SPSA 비교가 모두 이 목적값을 쓴다. 확정 규칙(J 띠 + T 접근)은 정제 뒤에 그대로 적용된다.
                ov_on = bool(a.refine_t_weight > 0 and tinfo is not None)
                ov = (lambda r_: r_["J"] + a.refine_t_weight * float(r_.get("dT", dT0))) if ov_on else (lambda r_: r_["J"])
                if ov_on:
                    feat_used["refine_t_weight"] = float(a.refine_t_weight)
                is_single = lambda r: not isinstance(r["action"][0], (list, tuple))
                relax_r = [False]      # 완화 모드: 한도 안 후보가 모자랄 때만 True 가 된다. 한도 안 후보가 충분하면 False 로 남아 예전과 똑같이 동작한다
                single = lambda: [r for r in allres.values() if r["J"] < 1e8 and (relax_r[0] or within(r)) and is_single(r)]
                n_feas_r = len([r for r in allres.values() if r["J"] < 1e8 and within(r) and is_single(r)])
                if n_feas_r < 4:
                    n_valid_r = len([r for r in allres.values() if r["J"] < 1e8 and is_single(r)])
                    if a.cbo and not a.no_fewobs_relax and n_valid_r >= 4:
                        # 한도 안 후보가 4개 미만이어도 패턴 탐색으로 대체하지 않는다. 한도를 넘은 후보의 J 도 물리로 잰 값이라 목적 대리모델에 쓸 수 있고
                        # (제약 BO, Gardner 2014: 목적 모델은 모든 관측으로 맞추고 제약은 별도 모델), 한도를 넘을 만한 제안은 아래 '한도를 지킬 확률'(--cbo)이 깎는다.
                        # 시작점·기준값은 한도 안 후보를 우선한다(rk, best_r). 2026-11-16: groove_top 에서 left_leg 한도 때문에 빔 후보 22개 중 21개가 한도를 넘어 BADS/SPSA/TuRBO 가 한 번도 안 돌았다.
                        relax_r[0] = True
                        feat_used["bads_relaxed"] = dict(n_feasible=n_feas_r, n_valid=n_valid_r)
                        print(f"    (한도 안 단일 후보 {n_feas_r}개 < 4: 한도를 넘은 {n_valid_r - n_feas_r}개도 대리모델에 쓰고 '한도를 지킬 확률'로 거른다)", flush=True)
                    else:
                        raise FewObservations("1단계에서 한도 안으로 측정된 후보가 4개 미만이라 대리모델을 맞출 수 없다")
                # rk: 시작점 순위 키. 한도 안이면 J, 넘으면 J + 1 + (위반 배율 - 1) = 한도 안 후보가 먼저, 없으면 덜 넘은 후보가 먼저. 완화 모드가 아니면 단일 후보가 전부 한도 안이라 J 와 같다
                rk = lambda r: ov(r) + (0.0 if within(r) else 1.0 + max(violation(r) - 1.0, 0.0))
                batch_r = a.bads_batch if a.bads_batch > 0 else max(len(eval_workers), 1)
                # 기준값(EI 의 incumbent) = 지금 확정할 수 있는 최선 = 한도 안 최선(완화 모드에서는 없으면 현재 J)
                cur_o_ = s_cur["J"] + (a.refine_t_weight * dT0 if ov_on else 0.0)       # 현재 상태의 목적값(현재-T 거리 dT0)
                best_r = min([cur_o_] + [ov(r) for r in single() if within(r)]) if relax_r[0] else min(ov(r) for r in single())
                J_beam, n_obs_r, used_r, since_r, it_r, ls_used, spsa_x = best_r, len(allres), 0, 0, 0, None, None
                tr_r, sp_k, n_sp_r, n_bads_r = E.TrustRegions(), 0, 0, 0
                cap_r = a.bads_evals + (a.spsa_after if a.refine_mode == "bads" else 0)       # --spsa-after: BADS 뒤에 SPSA [7] 를 이어서 돌린다. 총 평가 = bads_evals + spsa_after
                while used_r < cap_r:
                    mode_r = "spsa" if (a.refine_mode == "bads" and a.spsa_after > 0 and used_r >= a.bads_evals) else a.refine_mode
                    S_r = sorted(single(), key=rk)
                    items_r, pts_r, dirs_r = [], [], []
                    if mode_r == "bads":
                        F_r, J_r = np.array([E.act_feat(r["action"]) for r in S_r]), np.array([ov(r) for r in S_r])
                        Jk_r = np.array([rk(r) for r in S_r])       # 대리모델은 목적값(기본 J, --refine-t-weight 이면 J + 가중치 x dT)으로 맞추고, 시작점 고르기에만 순위 키를 쓴다
                        pred_r = E.fit_gp(F_r, J_r)
                        ls_used = pred_r.ls
                        props_r, seen_r, owner_r, own_i_r = [], set(), [], []
                        if a.turbo:      # [8] TuRBO: 서로 떨어진 시작점. 첫째 = 실측 최선, 둘째 = 반사실 사전이 가장 높은 관측(힌트가 가리킨 곳. 사전이 있을 때만)
                            hint_i_ = int(np.argmax([prior_raw(r["action"]) for r in S_r])) if prior_raw is not None else None
                            st_idx_ = E.pick_starts(F_r, Jk_r, max(a.refine_top, 1), a.penalize_ell, force_first=0, force_second=hint_i_)
                        else:
                            hint_i_, st_idx_ = None, list(range(min(max(a.refine_top, 1), len(S_r))))
                        for s_i_, i_ in enumerate(st_idx_):
                            r = S_r[i_]
                            scale_r = tr_r.length(s_i_) if a.turbo else 0.8 ** it_r        # 신뢰영역은 시작점 번호(슬롯)에 묶는다: 중심이 새 최선점으로 옮겨가도 길이를 유지(원본 TuRBO 와 같다)
                            for q_ in E.sample_around(rng_r, r["action"], scale_r, 150, clampA) + [clampA(v_) for v_ in neighbors(r["action"])]:
                                k_ = akey(q_)
                                if k_ not in allres and k_ not in meta and k_ not in seen_r:
                                    seen_r.add(k_)
                                    props_r.append((k_, q_))
                                    owner_r.append(s_i_)
                        if not props_r:
                            break
                        Xp_r = np.array([E.act_feat(q_) for _, q_ in props_r])
                        if a.turbo:      # 시작점마다 그 주변(신뢰영역) 관측으로만 맞춘 국소 모델
                            mu_r, sd_r, nloc_r = E.turbo_predict(F_r, J_r, F_r[st_idx_], Xp_r, owner_r, {s_: tr_r.length(s_) for s_ in range(len(st_idx_))})
                            feat_used["turbo"] = dict(starts=len(st_idx_), local_n=nloc_r, tr_len=[round(tr_r.length(s_), 2) for s_ in range(len(st_idx_))],
                                                      hint_start=bool(hint_i_ is not None and hint_i_ in st_idx_))
                        else:
                            mu_r, sd_r = pred_r(Xp_r)
                        acq_r = E.expected_improvement(mu_r, sd_r, best_r)
                        if prior_raw is not None:           # πBO: 사전^(beta/(1+이 라운드의 물리 관측 수)) — 관측이 쌓일수록 사전의 영향이 줄어든다
                            acq_r = acq_r * E.normalize_prior([prior_raw(q_) for _, q_ in props_r]) ** E.pibo_gamma(a.pibo_beta, n_obs_r)
                        pof_r = 1.0
                        if a.cbo:      # 제약 베이지안 최적화: 획득 점수 = EI x P(먼 곳 최대 변화 <= 한도). 한도를 넘은 후보까지 포함한 모든 단일 관측으로 제약 모델을 맞춘다
                            try:
                                g_obs = [r_ for r_ in allres.values() if r_["J"] < 1e8 and not isinstance(r_["action"][0], (list, tuple))]
                                if len(g_obs) >= 4:
                                    pg_ = E.fit_gp(np.array([E.act_feat(r_["action"]) for r_ in g_obs]), np.log1p(np.array([violation(r_) for r_ in g_obs], dtype=np.float64)))
                                    mg_, sg_ = pg_(Xp_r)
                                    pof_r = E.prob_feasible(mg_, sg_, np.log1p(1.0))
                                    feat_used["cbo"] = dict(n_obs=len(g_obs), n_over=int(sum(violation(r_) > 1.0 for r_ in g_obs)), pof_mean=round(float(np.mean(pof_r)), 3), limit_mm=round(a.far_limit_mm, 2), n_protected=len(prot_lim))
                            except Exception:
                                if a.strict:
                                    raise
                            acq_r = acq_r * pof_r
                        if float(acq_r.max()) <= 1e-12:     # 기대 개선이 전부 0 이면 예측 평균이 낮은 순으로
                            acq_r = np.maximum(float(mu_r.max()) - mu_r, 1e-9) * pof_r
                        for i_ in E.select_batch(Xp_r, acq_r, batch_r, a.penalize_ell):
                            k_, q_ = props_r[i_]
                            meta[k_] = dict(family="bads", base=None)
                            items_r.append((k_, q_))
                            own_i_r.append((k_, owner_r[i_]))
                    else:
                        if spsa_x is None:
                            spsa_x = S_r[0]["action"]
                        dirs_r = E.spsa_directions(rng_r, 2)
                        pts_r = E.spsa_points(spsa_x, dirs_r, sp_k, clampA)
                        for p_, m_ in pts_r:
                            for q_ in (p_, m_):
                                k_ = akey(q_)
                                if k_ not in allres and k_ not in meta:
                                    meta[k_] = dict(family="spsa", base=None)
                                    items_r.append((k_, q_))
                    if not items_r:
                        break
                    res_r = run_eval(items_r, chunk=1, tgt=tinfo, stage="bads")
                    for r_ in res_r:
                        allres[akey(r_["action"])] = r_
                    used_r += len(items_r)
                    n_obs_r += len(items_r)
                    got_r = [ov(r_) for r_ in res_r if r_["J"] < 1e8 and within(r_)]
                    if mode_r == "spsa":
                        sp_k += 1
                        n_sp_r += len(items_r)
                    else:
                        n_bads_r += len(items_r)
                    if a.turbo and mode_r == "bads":      # 신뢰영역 갱신: 시작점마다, 이번 배치에서 자기 제안 중 시작점 J 보다 좋아진 것이 있으면 성공
                        by_s_ = {}
                        for k_, s_ in own_i_r:
                            rr_ = allres.get(k_)
                            if rr_ is not None and rr_["J"] < 1e8 and within(rr_):
                                by_s_.setdefault(s_, []).append(ov(rr_))
                        for s_, j_ in enumerate(st_idx_):
                            if s_ in by_s_:
                                tr_r.update(s_, min(by_s_[s_]) < rk(S_r[j_]) - 0.002)
                    if mode_r == "spsa" and pts_r:                # 기울기 추정 -> 한 걸음 -> 그 점도 평가
                        jp_r = []
                        for p_, m_ in pts_r:
                            ovj_ = lambda k_: (ov(allres[k_]) if (k_ in allres and allres[k_]["J"] < 1e8) else 1e9)
                            jp, jm = ovj_(akey(p_)), ovj_(akey(m_))
                            jp_r.append((jp if jp < 1e8 else jm, jm if jm < 1e8 else jp))
                        x_new = E.spsa_update(spsa_x, dirs_r, jp_r, sp_k - 1, clampA)
                        k_ = akey(x_new)
                        if k_ not in allres and k_ not in meta:
                            meta[k_] = dict(family="spsa", base=None)
                            rn_ = run_eval([(k_, x_new)], chunk=1, tgt=tinfo, stage="bads")
                            for q_ in rn_:
                                allres[akey(q_["action"])] = q_
                            used_r += len(rn_)
                            n_obs_r += len(rn_)
                            got_r += [ov(q_) for q_ in rn_ if q_["J"] < 1e8 and within(q_)]
                            if rn_ and rn_[0]["J"] < 1e8 and within(rn_[0]) and ov(rn_[0]) < rk(allres[akey(spsa_x)]):
                                spsa_x = rn_[0]["action"]
                    nb_r = min(got_r) if got_r else 1e9
                    if nb_r < best_r - 0.002:
                        best_r, since_r = nb_r, 0
                    else:
                        since_r += len(items_r)
                    it_r += 1
                    if a.refine_patience > 0 and since_r >= a.refine_patience:
                        break
                feat_used.update(bads_evals=n_bads_r, spsa_evals=n_sp_r); print(f"    적응형 정제({a.refine_mode}{'+spsa' if (a.refine_mode == 'bads' and a.spsa_after > 0) else ''}{'+turbo' if a.turbo else ''}): 물리 평가 {used_r}개({it_r}회차), 최선 J {J_beam:.3f} -> {best_r:.3f}" + (f", 가우시안 과정 길이 척도 {ls_used}" if ls_used else ""), flush=True)
            except Exception as e_r:    # 이 단계가 실패해도 기존 패턴 탐색으로 계속한다 (--strict 면 예상 밖의 오류만 실패시킨다. 후보 부족은 대체 사실을 features 에 기록)
                if a.strict and not isinstance(e_r, FewObservations):
                    raise
                feat_used["bads_fallback"] = str(e_r)[:120]
                adaptive_failed = True
                print(f"    (적응형 정제 실패, 기존 패턴 탐색으로 대체: {e_r!r})", flush=True)
        # ---- 3단계 상위 후보 주변 패턴 탐색
        for it in range(a.refine_iters if (a.refine_mode == "pattern" or adaptive_failed) else 0):
            ok = sorted([r for r in allres.values() if within(r)], key=lambda r: r["J"])
            top = ok[:max(a.refine_top // (it + 1), 1)]
            nb = []
            for r in top:
                for act in neighbors(r["action"]):
                    act = clampA(act)
                    k = akey(act)
                    if k not in allres and k not in meta:
                        meta[k] = dict(family="refine", base=None)
                        nb.append((k, act))
            if not nb:
                break
            print(f"    {it + 2}단계(패턴 탐색 {it + 1}회차): 상위 {len(top)}개 주변 {len(nb)}개", flush=True)
            ref_kw = dict(patience=a.refine_patience, base_J=s_cur["J"], gain=0.002, need=-1e9, best0=ok[0]["J"]) if (a.refine_patience > 0 and ok) else {}
            for r in run_eval(nb, tgt=tinfo, stage="refine", **ref_kw):                 # --refine-patience: 최선이 0.002 이상 좋아지지 않은 채 연속 P개를 평가하면 패턴 탐색을 멈춘다
                allres[akey(r["action"])] = r
        # ---- 3b단계 자코비안 + Adam + Broyden (cf_jacobian.py): 모델 미분으로 방향을 정하고, 물리로 측정한 이웃 결과로 기울기를 고친다
        if model is not None and a.jac_top > 0:
            try:
                import cf_jacobian as jac
                score_fn = lambda m_: score(m_, cfg)["J"]
                prior = (tinfo["T"], tinfo["win"], dT0, a.w_cf) if tinfo is not None else None
                jst = {}
                for it in range(a.jac_iters):
                    single = sorted([r for r in allres.values() if r["J"] < 1e8 and within(r) and not isinstance(r["action"][0], (list, tuple))],
                                    key=lambda r: r["J"])
                    phys = [(r["action"], r["J"]) for r in single]
                    jb = []
                    for r in single[:a.jac_bases]:
                        bk = akey(r["action"])
                        props, jst[bk], dg = jac.propose(model, h_cur, r["action"], r["J"], phys, score_fn, jst.get(bk), clampA, _ok_end, seen=set(allres) | set(meta),
                                                         prior=prior, top=a.jac_top, radius=a.jac_radius, key=akey)
                        fmtg = lambda v: "[" + " ".join(f"{nm}{x:+.3f}" for nm, x in zip(jac.NAMES, v)) + "]"
                        print(f"    {it + 1}회차 자코비안: 기준 J={r['J']:.3f} | 모델 기울기(J/단위) {fmtg(dg['g0'])} -> 물리 이웃 {dg['n_nb']}개로 보정 {fmtg(dg['g'])} (변화 {dg['dg']:.3f}) -> 후보 {len(props)}개", flush=True)
                        for act, jhat, tag in props:
                            k = akey(act)
                            if k not in allres and k not in meta:
                                meta[k] = dict(family="JAC", base=None, pred=jhat, ref=r["J"], tag=tag)
                                jb.append((k, act))
                    if not jb:
                        break
                    for r_ in run_eval(jb, tgt=tinfo, stage="jac"):
                        allres[akey(r_["action"])] = r_
                    got = [allres[k] for k, _ in jb if k in allres and allres[k]["J"] < 1e8]
                    if got:
                        imp = sum(g_["J"] < meta[akey(g_["action"])]["ref"] - a.eps for g_ in got)
                        pr = [(meta[akey(g_["action"])]["pred"], g_["J"]) for g_ in got]
                        rho = ""
                        if len(pr) >= 4:
                            from scipy.stats import spearmanr
                            rho = f" | 예측 J_hat-물리 J 순위 상관 {spearmanr([x for x, _ in pr], [y for _, y in pr])[0]:+.2f}"
                        print(f"      JAC 결과: {len(got)}개 중 기준보다 J 가 좋아진 것 {imp}개, 최선 J={min(g_['J'] for g_ in got):.3f} (예측 평균 {np.mean([x for x, _ in pr]):.3f}, 실제 평균 {np.mean([y for _, y in pr]):.3f}){rho}", flush=True)
            except Exception as e:      # 이 단계가 실패해도 나머지 탐색은 계속한다
                print(f"    (자코비안 단계 실패, 건너뜀: {e!r})", flush=True)
        for k_, r_ in allres.items():
            if r_["J"] < 1e8 and k_ in meta:
                st_ = fam_stats.setdefault(meta[k_]["family"], [0, 0])
                st_[1] += 1
                st_[0] += int(r_["J"] < s_cur["J"] - a.eps)
        ok = sorted([r for r in allres.values() if within(r)], key=lambda r: r["J"])
        if not ok:
            print("    모든 후보가 영역 밖 변화 제한을 넘어 이 영역은 건너뜀", flush=True)
            stuck.add(n)
            log.append(dict(round=t_round, region=n, committed=False))
            continue
        # ---- 2-step 앞내다보기: 단일 후보 하나로는 J 가 안 좋아져도 두 행동을 이어서 하면 좋아지는 경우
        pair_best = None
        if a.pairs > 0 and len(ok) >= 2:
            try:
                n_pairs_r, n_sec_r = a.pairs, a.pair_seconds
                if a.pair_adapt:      # 메모 3쪽: 단일 개선이 작을수록 순서쌍 예산을 크게. 단일 개선이 좋아도 최소 1x2 는 시험해 놓치는 조합을 추정한다
                    g_s = (s_cur["J"] - ok[0]["J"]) / max(s_cur["J"], 1e-9)
                    n_pairs_r, n_sec_r = (1, 2) if g_s >= 0.25 else ((2, 3) if g_s >= 0.10 else (a.pairs, a.pair_seconds))
                if not a.pair_branch:      # 분기를 끄면 쌍은 상위 후보끼리만 만든다: 1개로는 쌍이 0개라 최소 2개
                    n_pairs_r = max(n_pairs_r, 2)
                pair_budget_r[0] = [n_pairs_r, n_sec_r]
                topk = ok[:n_pairs_r]
                seconds = ok[:max(n_pairs_r, n_sec_r)] if a.pair_branch else topk       # --pair-branch: 두 번째 푸시는 더 많이 볼 수 있다(첫 푸시는 한 번만 실행)
                items = [(("pair", akey(x["action"]), akey(y["action"])), [x["action"], y["action"]]) for x in topk for y in seconds if x is not y]
                pres = run_eval(items, chunk=(max(len(seconds) - 1, 1) if a.pair_branch else 2), tgt=tinfo, stage="pair")   # 같은 첫 푸시의 쌍들이 한 워커로 가야 상태 재사용이 된다
                pres = [r for r in pres if within(r) and r["J"] < 1e8]
                if pres:
                    pair_best = min(pres, key=lambda r: r["J"])
                    print(f"    2-step 앞내다보기: 상위 {len(topk)}개 순서쌍 {len(items)}개 평가 -> 최선 쌍 J={pair_best['J']:.3f} (단일 최선 {ok[0]['J']:.3f}, 현재 {s_cur['J']:.3f})", flush=True)
            except Exception as e:      # 앞내다보기가 실패해도 단일 후보 탐색은 계속한다
                print(f"    (2-step 앞내다보기 실패, 건너뜀: {e!r})", flush=True)
                pair_best = None
        # ---- PLANSEQ: 계획기가 짠 여러 행동을 한 묶음으로 Genesis 에서 실행해 본다 (묶음 전체의 J 로만 판단, 모델 예측은 쓰지 않는다)
        pair_fam = "PAIR"
        if plan_seqs:
            try:
                raw_sres = run_eval([(("seq", i_), s_) for i_, s_ in enumerate(plan_seqs)], tgt=tinfo, stage="planseq")
                for r_ in raw_sres:      # 묶음마다 J 와 어느 한도에 걸렸는지 남긴다(2026-11-16: 모델이 계획한 3푸시 연쇄 4개가 전부 한도 불통과였는데 이유가 로그에 없었다)
                    viol_ = (["far"] if r_.get("far_max", 0.0) > a.far_limit_mm else []) + [n_ for n_, d_ in (r_.get("prot") or {}).items() if d_ > prot_lim.get(n_, a.protect_limit_mm)]
                    print(f"      묶음 J={r_['J']:.3f} 먼 곳 {r_.get('far_max', 0.0):.1f}mm(한도 {a.far_limit_mm:.1f}) 보호 최대 {r_.get('prot_max', 0.0):.2f}mm 위반 {viol_ or '없음(충돌 한도일 수 있음)'}", flush=True)
                sres = [r for r in raw_sres if within(r) and r["J"] < 1e8]
                allj = sorted(r["J"] for r in sres)
                print(f"    PLANSEQ {len(plan_seqs)}개 묶음({a.plan_seq}행동씩) Genesis 평가: 영역 밖 한도 통과 {len(sres)}개, 현재 J {s_cur['J']:.3f} 대비 개선 {sum(j < s_cur['J'] - a.eps for j in allj)}개"
                      + (f", 최선 J={allj[0]:.3f}" if allj else ""), flush=True)
                if sres:
                    sb = min(sres, key=lambda r: r["J"])
                    if pair_best is None or sb["J"] < pair_best["J"]:
                        pair_best, pair_fam = sb, "PLANSEQ"
            except Exception as e:      # 묶음 평가가 실패해도 나머지는 계속한다
                print(f"    (PLANSEQ 평가 실패, 건너뜀: {e!r})", flush=True)
        # J 가 실제로 좋아진 후보만 확정 대상. 그중에서는 반사실 목표 T 에 가장 가까운 것(없으면 J 가 가장 낮은 것).
        tot = lambda r: r["J"] + (a.w_cf * r.get("dT", dT0) / dT0 if tinfo is not None else 0.0)
        good = [r for r in ok if r["J"] < s_cur["J"] - max(a.eps, a.min_commit_gain)]
        t_mode = a.cf_target == "fixed" and tinfo is not None and not a.commit_by_j      # --commit-by-j: T 는 후보·사전에만, 확정은 J 로만(아래 else 분기)
        if t_mode:      # T-추적: 고정 T 까지 거리가 줄어드는(그리고 요청 기준 J 가 허용치 안인) 후보 중 T 에 가장 가까운 것
            elig = [r for r in ok if r.get("dT") is not None and r["dT"] < dT0 - a.eps_t and r["J"] <= s_cur["J"] + a.t_slack and r["J"] <= s_cur["J"] - a.min_commit_gain]
            n_elig0, jmin_t, jfirst = len(elig), min((r["J"] for r in elig), default=None), False
            if a.t_jband > 0 and elig:      # J 밴드(--t-jband): T 에 가까워지고 한도를 지키는 후보 중 J 가 최선에서 t_jband 이내인 것만 남기고, 그중 T 에 가장 가까운 것을 확정한다. T 는 방향 힌트, J 는 실제 목표 접근
                elig = [r for r in elig if r["J"] <= jmin_t + a.t_jband]
            elif a.t_jband > 0 and good:    # T 쪽으로 다가가는 후보가 하나도 없으면 J 가 가장 좋은 후보를 확정한다(예전에는 확정 없음으로 끝났다)
                elig, jfirst = good, True
                print("    [T-추적] T 에 가까워지는 후보가 없어 J 가 가장 좋은 후보를 확정 대상으로 한다(--t-jband)", flush=True)
            best = (min(elig, key=lambda r: r["J"]) if jfirst else min(elig, key=lambda r: r["dT"])) if elig else ok[0]
            n_closer = sum(1 for r in allres.values() if r.get("dT") is not None and r["dT"] < dT0 - a.eps_t)
            print(f"    [T-추적] 고정 T 까지 거리 현재 {dT0:.2f}mm | T 에 가까워진 후보 {n_closer}/{len(allres)}개, 요청 J 허용치({s_cur['J'] + a.t_slack:.3f}) 안 {n_elig0}개"
                  + ((f" | 확정 후보 dT {best.get('dT', float('nan')):.2f}mm (J {best['J']:.3f})" + (f", J 밴드 {a.t_jband:g} 이내(밴드 안 최선 J {jmin_t:.3f})" if (a.t_jband > 0 and jmin_t is not None) else "")) if elig else ""), flush=True)
        else:
            best = min(good, key=tot) if good else ok[0]
        if tinfo is not None and not t_mode:
            print(f"    반사실 목표 T 까지 거리: 현재 {dT0:.2f}mm -> 최선 후보 {best.get('dT', float('nan')):.2f}mm (J {best['J']:.3f})", flush=True)
        (a.out / f"eval_round_{t_round:02d}.json").write_text(json.dumps([dict(region=n, J=r["J"], far_max=r["far_max"], action=r["action"], family=meta.get(akey(r["action"]), {}).get("family"),
                                                                              dT=r.get("dT"), base=meta.get(akey(r["action"]), {}).get("base"), proxy=meta.get(akey(r["action"]), {}).get("proxy"), seq=r.get("seq"), stage=r.get("stage"), prot_max=r.get("prot_max"), prot=r.get("prot"), Ja=r.get("J_active"), side_gain=r.get("side_gain"),
                                                                              per={k: list(v) for k, v in r["per"].items()},
                                                                              Jr={k_: float(region_J(k_, r["vis"], cfg)) for k_ in r["per"]} if r.get("vis") else None) for r in allres.values()], ensure_ascii=False))   # Jr: 같은 높이맵에서 다섯 영역을 각각 채점한 점수
        is_pair = False
        if t_mode:
            d_best = best.get("dT") if (elig and best.get("dT") is not None) else dT0
            if (pair_best is not None and pair_best.get("dT") is not None and pair_best["dT"] < min(d_best, dT0 - a.eps_t) and pair_best["J"] <= s_cur["J"] + a.t_slack
                    and (a.t_jband <= 0 or jmin_t is None or pair_best["J"] <= jmin_t + a.t_jband) and pair_best["J"] <= s_cur["J"] - a.min_commit_gain):
                best, is_pair, elig = pair_best, True, [pair_best]
        elif pair_best is not None and pair_best["J"] < min(best["J"], s_cur["J"]) - a.eps:
            best, is_pair = pair_best, True
        fam_best = dict(family=pair_fam, base=None) if is_pair else meta.get(akey(best["action"]), {})
        print("    상위 3: " + "; ".join(f"J={r['J']:.3f}({meta.get(akey(r['action']), {}).get('family')}{'#' + str(meta.get(akey(r['action']), {}).get('base')) if meta.get(akey(r['action']), {}).get('base') else ''})" for r in ok[:3]) +
              f" | 현재 J={s_cur['J']:.3f} | 평가 {len(allres)}개 중 개선 {sum(r['J'] < s_cur['J'] - a.eps for r in allres.values())}개", flush=True)
        carry[n] = [r["action"] for r in ok[:a.carry]]
        rec = dict(round=t_round, region=n, n_eval=len(allres), best_J=best["J"], cur_J=s_cur["J"], family=fam_best.get("family"), base=fam_best.get("base"),
                   seed=a.seed, round_sec=time.time() - t_round0, stage_sec=dict(stage_sec), stage_n=dict(stage_cnt),
                   rho=dict(rho_info), features=dict(feat_used), pair_budget=pair_budget_r[0],
                   credit=dict(side=list(credit_state["side"]), w=a.credit_side, best_side_gain=best.get("side_gain"), best_J_active=best.get("J_active")))
        if (bool(elig) if t_mode else best["J"] < s_cur["J"] - max(a.eps, a.min_commit_gain)):
            commits += 1
            sp = a.out / "states" / f"iter_{commits:03d}.pt"
            tqs[0].put(("commit", best["action"], str(sp), commits))
            while True:
                m = rq.get(timeout=3600)
                if m[0] == "committed":
                    h_new = m[2].astype(np.float64)
                    break
            os.replace(sp, sp)  # 존재 확인
            for w in eval_workers:
                if w != 0:
                    tqs[w].put(("load", str(sp), commits))
            got = 0
            while got < len([w for w in eval_workers if w != 0]):
                if rq.get(timeout=3600)[0] == "loaded":
                    got += 1
            s_new = score(h_new, cfg, h_cur)
            acts_c = best["action"] if is_pair else [best["action"]]
            desc = " + ".join(f"x={q[0]:.4f} y={q[1]:.4f} th={q[2]:+.2f} L={q[3] * 1000:.0f}mm d={q[4] * 1000:.0f}mm" for q in acts_c)
            print(f"    확정 {commits}: {fam_best.get('family')}{'#' + str(fam_best.get('base')) if fam_best.get('base') else ''} {desc} "
                  f"-> J {s_cur['J']:.3f} -> {s_new['J']:.3f} (예측 {best['J']:.3f}) 밖 {s_new['far_max']:.1f}mm | {fmt(s_new)}", flush=True)
            h_cur, s_cur = h_new, score(h_new, cfg, h_new)
            np.save(a.out / "states" / f"iter_{commits:03d}.npy", h_cur.astype(np.float32))
            rcount[n] += 1
            if tinfo is not None:
                dT_new = float(np.abs(h_cur - Tn)[tinfo["win"]].mean() * 1000)
                dT_hist[n].append(dT_new)
                print(f"    반사실 목표 T 까지 거리: {dT0:.2f} -> {dT_new:.2f} mm ({100 * (1 - dT_new / dT0):+.0f}%)", flush=True)
                rec["dT_before"], rec["dT_after"] = dT0, dT_new
            prog = {k_: progress(h_cur, k_) for k_ in names}
            print("    영역 목표 진행률(영역 평균 변화 / 목표 변화, 100%=목표 도달): " + ", ".join(f"{k_} " + ("정의 안 됨" if v_ is None else f"{100 * v_:.0f}%") for k_, v_ in prog.items()), flush=True)
            if a.per_region_commits > 0 and rcount[n] >= a.per_region_commits:
                done_reg.add(n)
                print(f"    '{n}' 은(는) 영역당 확정 한도({a.per_region_commits}회)에 도달해 더 확정하지 않음", flush=True)
            elif a.stop_progress > 0 and prog[n] is not None and prog[n] >= a.stop_progress:
                done_reg.add(n)
                print(f"    '{n}' 은(는) 목표 진행률 {100 * prog[n]:.0f}% >= {100 * a.stop_progress:.0f}% 라 더 확정하지 않음", flush=True)
            make_frame(fin / f"frame_{commits:03d}.png", h_cur, cfg, hstar_all, bg, crop, f"after compensation action {commits} ({rec['family']}) J={s_cur['J']:.3f}")
            rec.update(committed=True, action=best["action"], J_after=s_cur["J"], per={k: list(v) for k, v in s_cur["per"].items()}, past_base=rec["base"])
            stuck.clear()
        else:
            print(f"    개선되는 후보가 없어 '{n}' 은(는) 잠시 멈춤 (다른 영역이 바뀌면 다시 시도)", flush=True)
            stuck.add(n)
            rec["committed"] = False
        log.append(rec)
        (a.out / "rounds.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    # ---- 정리
    np.save(a.out / "final_m.npy", h_cur.astype(np.float32))
    np.save(a.out / "ref_m.npy", h_ref.astype(np.float32))
    summ = dict(commits=commits, minutes=(time.time() - t_start) / 60, replay_sec=replay_sec, seed=a.seed, strict=bool(a.strict), protect_limits=dict(prot_lim), protect_mode=("abs" if a.protect_abs else "harm"), level_adjust=dict(level_info), far_limit_mm=float(a.far_limit_mm), credit_side=float(a.credit_side),
                cf_kinds={str(k_): sum(1 for r_ in cf_table if r_.get("kind") == k_) for k_ in {r_.get("kind") for r_ in cf_table}}, start_ref=dict(J=s_ref["J"], focus=s_ref.get("focus"), per={k: list(v) for k, v in s_ref["per"].items()}, vis={k: list(v) for k, v in s_ref["vis"].items()}),
                final=dict(J=s_cur["J"], focus=s_cur.get("focus"), per={k: list(v) for k, v in s_cur["per"].items()}, vis={k: list(v) for k, v in s_cur["vis"].items()}, coll=s_cur["coll"]),
                weights=dict(height=a.w_height, grad=a.w_grad, edge=a.w_edge, tail=a.w_tail, mu=a.mu, cf=a.w_cf, blur_tools=a.blur_tools), cf_replay=cf_table,
                cf_used={n: (None if cf_best[n] is None else dict(start=cf_best[n]["start"], variant=cf_best[n]["variant"], q=cf_best[n]["q"])) for n in names})
    s_orig1 = score(h_cur, cfg_orig, h_start) if cfg_orig is not None else None
    summ["progress"] = {n: progress(h_cur, n) for n in names}
    summ["dT_history"] = dT_hist
    summ["cf_target"] = a.cf_target
    summ["commits_per_region"] = rcount
    if cfg_orig is not None:
        s_req0 = score(h_start, cfg)
        summ["dual_criteria"] = dict(request=dict(J_start=s_req0["J"], J_final=s_cur["J"], e_start=s_req0["e"], e_final=s_cur["e"]),
                                     original_target=dict(e_start=s_orig0["e"], e_final=s_orig1["e"], vis_start={k: list(v) for k, v in s_orig0["vis"].items()}, vis_final={k: list(v) for k, v in s_orig1["vis"].items()}))
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print("\n완료. 기준 상태 -> 최종 (MAE/RMSE mm):", flush=True)
    for n in names:
        print(f"  {n:<12} 형태 {s_ref['vis'][n][2]:.2f}->{s_cur['vis'][n][2]:.2f} | 기울기 {s_ref['vis'][n][0]:.2f}->{s_cur['vis'][n][0]:.2f} | 테두리 {s_ref['vis'][n][1]:.2f}->{s_cur['vis'][n][1]:.2f} | (원본 높이 MAE {s_ref['per'][n][0]:.2f}->{s_cur['per'][n][0]:.2f}mm)", flush=True)
    print(f"  J {s_ref['J']:.3f} -> {s_cur['J']:.3f}, 확정 행동 {commits}개, {summ['minutes']:.0f}분", flush=True)
    if cfg_orig is not None:
        dc = summ["dual_criteria"]
        print(f"  [두 기준] 요청 기준 영역 점수 {dc['request']['e_start']:.3f} -> {dc['request']['e_final']:.3f} | 원래 목표 기준 영역 점수 {dc['original_target']['e_start']:.3f} -> {dc['original_target']['e_final']:.3f} "
              f"(낮을수록 좋음. 두 기준이 반대로 움직이면 요청과 원래 목표가 서로 반대 방향인 영역이 있다는 뜻)", flush=True)
        for n in names:
            vs, vf = s_orig0["vis"][n], s_orig1["vis"][n]
            print(f"    {n:<12} 원래 목표 기준 형태 {vs[2]:.2f}->{vf[2]:.2f} | 기울기 {vs[0]:.2f}->{vf[0]:.2f} | 테두리 {vs[1]:.2f}->{vf[1]:.2f} | 요청 목표 진행률 "
                  + ("정의 안 됨" if summ["progress"][n] is None else f"{100 * summ['progress'][n]:.0f}%"), flush=True)
    for w in tqs:
        tqs[w].put(("stop",))
    time.sleep(3)


if __name__ == "__main__":
    main()
