"""설계 §7 + 10~12단계: 현재 상태에서 국소 자코비안 보정 (beam -> 자코비안+Adam -> 1행동 실행 -> 재관측 -> Broyden).

한 번 반복:
  1) beam   : 편집 영역 주변의 (x,y,theta) 후보 x 공칭 (l,d) 를 Param2Deform 으로 평가해 상위 B개 선택
  2) 자코비안: 후보마다 J = d(dH)/d(l,d) 를 중앙차분으로 구하고, 아래 최소제곱으로 (l,d) 초기해를 얻음
        min |M_e (D0 + J dz - G)|^2 + lam_p |M_p (D0 + J dz)|^2 [+ lam_c |M_e (D0 + J dz - C*)|^2] + lam_a |dz|^2
        (G = H* - H_now 는 남은 목표 변형, D0 는 (l0,d0) 에서의 예측 변형)
  3) Adam   : 위 해에서 시작해 실제(비선형) 모델로 (l,d) 를 미세조정 (설계 손실 L + lam_c 항을 직접 최소화)
  4) 실행   : 예측 이득이 --min-gain 이하이면 실행하지 않고 종료(아무것도 안 하는 선택지). 아니면 1행동만 Genesis 실행
  5) 재관측 + Broyden: 모델 오차 e = dH_real - f(H, a) 를 관측해 보정항을 갱신
        예측 변형 = f(H, a) + g(a) * [ B + Je (u(a) - u_prev) ],   g = exp(-|xy - xy_prev| / sigma)
        Je <- Je + (e_n - g B_prev - Je du) du^T / (du^T du + eps)     (rank-1, 정규화한 행동 u 공간)
     설계의 식(J += (dH_real - J da) da^T/(da^T da))은 상태가 매번 바뀌는 상황에서 dH_real 이 모호하므로,
     같은 형태의 갱신을 '모델 오차 e'에 적용했다. --no-broyden 이면 보정항을 쓰지 않는다.

이력 폴더는 읽기만 한다. 실행한 보정 행동은 새 ActionHistory 로 기록해서 다시 재생·분석할 수 있다.
사용: python local_correct.py --history-dir H --spec-dir S --checkpoint CK --out-dir DIR [--cstar C.npy] [--no-broyden]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

U_SCALE = (0.01, 0.01, 0.2, 0.005, 0.005)   # x[m] y[m] theta[rad] l[m] d[m] 를 비슷한 크기로 맞추는 척도


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--restore-action", type=int)
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--cstar", type=Path, help="유효한 C* (512x512, m). 없으면 과거 반사실 항을 쓰지 않는다(abstain).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--pos-samples", type=int, default=48)
    ap.add_argument("--reach-px", type=int, default=12, help="후보 시작점은 M_e 를 이만큼 부풀린 영역 안에서 뽑는다")
    ap.add_argument("--n-theta", type=int, default=12)
    ap.add_argument("--nominal", default="8:6,14:12", help="공칭 길이:깊이(mm) 쌍. beam 평가용")
    ap.add_argument("--l-range", default="5,20", help="보정 행동 길이 범위(mm)")
    ap.add_argument("--d-range", default="3,25", help="보정 행동 깊이 범위(mm). planner 의 하한(10mm)보다 낮다")
    ap.add_argument("--lam-p", type=float, default=1.0)
    ap.add_argument("--lam-a", type=float, default=0.05)
    ap.add_argument("--lam-c", type=float, default=0.5)
    ap.add_argument("--adam-steps", type=int, default=30)
    ap.add_argument("--adam-lr", type=float, default=0.2)
    ap.add_argument("--min-gain", type=float, default=0.005, help="모델이 예측한 L 감소가 이보다 작으면 실행하지 않고 종료")
    ap.add_argument("--success-frac", type=float, default=0.5, help="L 이 시작값의 이 비율 이하이면 성공 종료")
    ap.add_argument("--sigma", type=float, default=0.015, help="Broyden 보정항 게이트 폭[m]")
    ap.add_argument("--no-broyden", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="실행 전에 상위 후보를 시뮬레이터에서 실제로 시도해 보고(롤백), 눈에 보이는 오차가 줄어드는 것만 확정한다")
    ap.add_argument("--verify-k", type=int, default=4, help="실제로 시도해 볼 후보 수")
    ap.add_argument("--verify-eps", type=float, default=0.01, help="눈에 보이는 오차 점수가 이 값 이상 줄어야 확정")
    ap.add_argument("--viewer", action="store_true", help="Genesis 뷰어 창을 켜고 점토가 눌리는 것을 실시간으로 본다 (화면 필요)")
    a = ap.parse_args()

    out = a.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"새 폴더를 지정하세요: {out}")
    (out / "states").mkdir(parents=True, exist_ok=True)
    root = a.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if a.viewer else "0"

    import vs_simulator as sim
    import vs_config as cfg
    from edit_spec import load_spec
    from vs_dynamics import Param2Deform
    from vs_history import ActionHistory, load_snapshot, physics_signature
    from vs_planner import predict_next_batch, wrap_angle

    # ---- 복원: 이력의 마지막 상태에서 이어서 -----------------------------------
    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    n0 = a.restore_action or len(dirs)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다. 이 이력은 복원할 수 없습니다.")
    spec_probe = np.load(a.spec_dir / "h_star.npy")
    h_star_t = torch.from_numpy(spec_probe)
    roi_path = a.spec_dir / "m_edit.npy"
    hist = ActionHistory(sim, out / "history", target=h_star_t, raw_target=h_star_t, roi_path=roi_path,
                         checkpoint_path=a.checkpoint, target_transform=None,
                         run_args={k: str(v) for k, v in vars(a).items()})
    hist.adapter.restore(load_snapshot(root / f"action_{n0:06d}" / "post_state.pt"))
    h0 = sim.get_height_gpu()
    device, dtype = h0.device, h0.dtype

    def state():
        h = torch.as_tensor(sim.get_height_gpu(), device=device, dtype=dtype)
        while h.ndim > 2:
            h = h[0]
        return h[None, None]

    start = state()[0, 0].cpu().numpy()
    recorded = np.load(root / f"action_{n0:06d}" / "transition.npz")["after_m"]
    restore_err = float(np.abs(start - recorded).max()) * 1000
    print(f"[복원] 행동 {n0} 직후, 기록 대비 최대 차이 {restore_err:.4f}mm")
    if restore_err > 1e-3:
        raise RuntimeError("복원한 높이맵이 기록과 다릅니다.")

    spec = load_spec(a.spec_dir, device, dtype)
    me_flat = spec.m_edit.flatten().nonzero().squeeze(1)
    mp_flat = spec.m_preserve.flatten().nonzero().squeeze(1)
    me2d = spec.m_edit[0, 0]
    n_e, n_p = float(me_flat.numel()), float(mp_flat.numel())
    cstar = None
    if a.cstar is not None:
        cstar = torch.from_numpy(np.load(a.cstar)).to(device=device, dtype=dtype)[None, None]
        print(f"[C*] 과거 반사실 효과 사용: 편집영역 평균 |C*|={float(cstar[spec.m_edit].abs().mean()) * 1000:.3f}mm, lam_c={a.lam_c}")
    else:
        print("[C*] 없음 (abstain) -> 과거 반사실 항 없이 현재 상태 보정만 수행")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = Param2Deform()
    model.load_state_dict(ck["model"])
    del ck
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    l_lo, l_hi = (float(v) / 1000 for v in a.l_range.split(","))
    d_lo, d_hi = (float(v) / 1000 for v in a.d_range.split(","))
    nominal = [tuple(float(v) / 1000 for v in s.split(":")) for s in a.nominal.split(",")]
    u_scale = torch.tensor(U_SCALE, device=device, dtype=dtype)
    rng = np.random.default_rng(a.seed)

    # ---- Broyden 상태 ---------------------------------------------------------
    corr = dict(B=None, Je=torch.zeros(5, 1, 512, 512, device=device, dtype=dtype), a_prev=None)

    def u_diff(acts, a_prev):
        d = acts - a_prev[None]
        d = torch.cat([d[:, :2], wrap_angle(d[:, 2:3]), d[:, 3:]], dim=1)   # in-place 대입은 autograd 를 깨뜨린다
        return d / u_scale

    def cdelta(H, acts):
        """보정된 예측 변형 f(H,a) + g(a)[B + Je du].  acts: (N,5) -> (N,1,512,512)"""
        d = predict_next_batch(model, H, acts) - H
        if a.no_broyden or corr["B"] is None:
            return d
        ap_ = corr["a_prev"]
        g = torch.exp(-torch.linalg.vector_norm(acts[:, :2] - ap_[None, :2], dim=1) / a.sigma)
        du = u_diff(acts, ap_)
        return d + g[:, None, None, None] * (corr["B"] + torch.einsum("nj,jchw->nchw", du, corr["Je"]))

    def objective(H, delta):
        L, t = spec.loss(H + delta, return_terms=True)
        obj = L
        if cstar is not None:
            obj = obj + a.lam_c * ((delta - cstar).abs() * spec.m_edit).sum(dim=(1, 2, 3)) / n_e * 1000.0
        return obj, L, t

    x0w = cfg.CLAY_CENTER_X - cfg.CLAY_SIZE_X / 2
    y0w = cfg.CLAY_CENTER_Y - cfg.CLAY_SIZE_Y / 2
    pix = cfg.CLAY_SIZE_X / (cfg.HEIGHT_MAP_RES - 1)
    from scipy import ndimage
    reach = ndimage.binary_dilation(me2d.cpu().numpy(), iterations=a.reach_px)
    reach_idx = np.flatnonzero(reach.ravel())

    def gen_candidates():
        sel = rng.choice(reach_idx, size=min(a.pos_samples, reach_idx.size), replace=False)
        rows, cols = np.divmod(sel, 512)
        xs, ys = x0w + cols * pix, y0w + rows * pix
        th = np.linspace(-math.pi, math.pi, a.n_theta, endpoint=False)
        c = [[x, y, t, l, d] for x, y in zip(xs, ys) for t in th for (l, d) in nominal]
        return torch.tensor(c, device=device, dtype=dtype)

    def refine(H, G_mm, cand):
        """후보 하나의 (l,d) 를 자코비안 최소제곱 + Adam 으로 다듬는다. 반환: 행동, 목적함수, L."""
        x, y, th, l0, d0 = (float(v) for v in cand)
        h = 0.0015
        acts = torch.tensor([[x, y, th, l0, d0], [x, y, th, l0 + h, d0], [x, y, th, l0 - h, d0],
                             [x, y, th, l0, d0 + h], [x, y, th, l0, d0 - h]], device=device, dtype=dtype)
        with torch.no_grad():
            dl = cdelta(H, acts)
            D0 = dl[0].flatten() * 1000.0
            Jl = (dl[1] - dl[2]).flatten() * 1000.0 / (2 * h) * U_SCALE[3]    # mm per (l 정규화 단위)
            Jd = (dl[3] - dl[4]).flatten() * 1000.0 / (2 * h) * U_SCALE[4]
            Gf = G_mm.flatten()
            we, wp = 1.0 / math.sqrt(n_e), math.sqrt(a.lam_p / n_p)
            A = [torch.stack([Jl[me_flat], Jd[me_flat]], 1) * we, torch.stack([Jl[mp_flat], Jd[mp_flat]], 1) * wp]
            b = [(Gf[me_flat] - D0[me_flat]) * we, (0.0 - D0[mp_flat]) * wp]
            if cstar is not None:
                wc = math.sqrt(a.lam_c / n_e)
                Cf = cstar.flatten() * 1000.0
                A.append(torch.stack([Jl[me_flat], Jd[me_flat]], 1) * wc)
                b.append((Cf[me_flat] - D0[me_flat]) * wc)
            A.append(math.sqrt(a.lam_a) * torch.eye(2, device=device, dtype=dtype))
            b.append(torch.zeros(2, device=device, dtype=dtype))
            A, b = torch.cat(A).double(), torch.cat(b).double()
            z = torch.linalg.lstsq(A, b[:, None]).solution[:, 0].to(dtype)
        lin = torch.tensor([l0 + float(z[0]) * U_SCALE[3], d0 + float(z[1]) * U_SCALE[4]], device=device, dtype=dtype)
        lin = torch.stack([lin[0].clamp(l_lo, l_hi), lin[1].clamp(d_lo, d_hi)])
        p = (lin / U_SCALE[3]).clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=a.adam_lr)
        fixed = torch.tensor([x, y, th], device=device, dtype=dtype)
        best = None
        for step in range(a.adam_steps + 1):
            act = torch.cat([fixed, p * U_SCALE[3]])[None]
            obj, L, _ = objective(H, cdelta(H, act))
            if best is None or float(obj) < best[0]:
                best = (float(obj), float(L), act.detach()[0].clone())
            if step == a.adam_steps:
                break
            opt.zero_grad()
            obj.sum().backward()
            opt.step()
            with torch.no_grad():
                p[0].clamp_(l_lo / U_SCALE[3], l_hi / U_SCALE[3])
                p[1].clamp_(d_lo / U_SCALE[3], d_hi / U_SCALE[3])
        return best

    def metrics(h):
        L, t = spec.loss(h, return_terms=True)
        d = (h - spec.h_t).abs() * 1000.0
        zone = spec.m_edit | spec.m_boundary
        return dict(L=float(L), r=float(t["r"]), g=float(t["g"]), b=float(t["b"]), p=float(t["p"]),
                    preserve_max_mm=float(d[spec.m_preserve].max()),
                    outside_zone_mean_mm=float(d[~zone].mean()), outside_zone_max_mm=float(d[~zone].max()),
                    edit_depth_gain_mm=float(((spec.h_t - h) * 1000.0)[spec.m_edit].mean()))

    from visual_score import VisualScorer
    vs_scorer = VisualScorer(np.load(a.spec_dir / "h_star.npy"), start, np.load(a.spec_dir / "m_edit.npy") > 0)

    def vscore(h):
        """눈에 보이는 오차 점수 S: 창 L3D, 창 Lviz, 테두리, 전체 Lviz 의 평균 비율 (시작 = 1.00, 낮을수록 목표와 비슷)."""
        r = vs_scorer.ratios(h)
        return float((r["l3d_win"] + r["lviz_win"] + r["edge_win"] + r["lviz_full"]) / 4.0)

    m0 = metrics(state())
    print(f"[시작] L={m0['L']:.4f} (홈 깊이오차 r={m0['r']:.3f}mm, 기울기 g={m0['g']:.3f}, 경계 b={m0['b']:.3f}, 보존 p={m0['p']:.4f}mm)")
    rows, actions = [dict(step=0, **m0, pred_gain=0.0, real_gain=0.0, gap_edit_mm=0.0, gap_edit_corrected_mm=0.0, seconds=0.0)], []
    stop_reason = "반복 횟수 도달"
    for n in range(a.iters):
        H = state()
        L_now = float(spec.loss(H))
        G_mm = (spec.h_star - H) * 1000.0
        t0 = time.perf_counter()
        # ---- 1) beam ----
        cands = gen_candidates()
        objs = []
        with torch.no_grad():
            for i in range(0, cands.shape[0], 24):
                objs.append(objective(H, cdelta(H, cands[i:i + 24]))[0])
        objs = torch.cat(objs)
        top = torch.argsort(objs)[:a.beam]
        # ---- 2~3) 자코비안 + Adam ----
        results = [refine(H, G_mm, cands[i]) for i in top]
        obj, L_pred, act = min(results, key=lambda r: r[0])
        pred_gain = L_now - L_pred
        plan_sec = time.perf_counter() - t0
        print(f"[{n + 1:02d}] beam 후보 {cands.shape[0]}개 -> 상위 {a.beam}개 다듬음 ({plan_sec:.0f}s) | "
              f"선택 x={float(act[0]):.4f} y={float(act[1]):.4f} th={float(act[2]):+.2f} l={float(act[3]) * 1000:.1f}mm d={float(act[4]) * 1000:.1f}mm "
              f"| 예측 L {L_now:.4f}->{L_pred:.4f} (이득 {pred_gain:+.4f})")
        if pred_gain < a.min_gain and not a.verify:
            stop_reason = f"모델 예측 이득 {pred_gain:+.4f} < {a.min_gain} -> 실행하지 않고 종료 (no-op)"
            print("    " + stop_reason)
            break
        if a.verify:
            # 모델 예측은 믿지 않는다: 상위 후보를 실제 물리로 시도(캡처 -> 실행 -> 관측 -> 복원)하고 눈에 보이는 오차를 잰다.
            order = sorted(results, key=lambda r: r[0])[:a.verify_k]
            s_now = vscore(H[0, 0].cpu().numpy())
            saved = hist.adapter.capture()
            trials = []
            for (o_, Lp_, ac_) in order:
                hist.adapter.execute(ac_.detach().cpu().float(), n)
                trials.append((vscore(hist.adapter.observe().numpy()), Lp_, ac_))
                hist.adapter.restore(saved)
            best_s, Lp_v, act_v = min(trials, key=lambda t: t[0])
            print(f"    [검증] 실제 물리로 {len(trials)}개 시도 후 되돌림. 눈에 보이는 오차 S: 지금 {s_now:.3f} -> 후보별 " +
                  ", ".join(f"{t[0]:.3f}" for t in trials))
            if best_s > s_now - a.verify_eps:
                stop_reason = f"검증 실패: 시도한 {len(trials)}개 후보 모두 실제로는 개선 없음(최선 S {best_s:.3f} vs 지금 {s_now:.3f}) -> 확정하지 않고 종료 (no-op)"
                print("    " + stop_reason)
                break
            act, pred_gain = act_v, L_now - Lp_v
            print(f"    [확정] S {s_now:.3f} -> {best_s:.3f} 인 후보를 실제로 실행합니다.")
        # ---- 4) 실행 (1행동) ----
        with torch.no_grad():
            delta_model = predict_next_batch(model, H, act[None]) - H
            delta_corr = cdelta(H, act[None])
        before = hist.adapter.observe()
        t1 = time.perf_counter()
        hist.record_push(act.detach().cpu().float(), action_index=n, before=before, predicted=(H + delta_corr)[0, 0].cpu(),
                         planner_context=dict(iteration=n, pred_gain=pred_gain, beam_objectives=[float(r[0]) for r in results],
                                              broyden=not a.no_broyden, cstar=a.cstar is not None))
        sec = time.perf_counter() - t1
        H1 = state()
        d_real = H1 - H
        actions.append([float(v) for v in act.cpu()])
        # ---- 5) 재관측 + Broyden ----
        gap_raw = float((delta_model - d_real).abs()[spec.m_edit].mean()) * 1000
        gap_cor = float((delta_corr - d_real).abs()[spec.m_edit].mean()) * 1000
        e_n = (d_real - delta_model)
        if not a.no_broyden:
            if corr["a_prev"] is not None and corr["B"] is not None:
                du = u_diff(act[None], corr["a_prev"])[0]
                g = torch.exp(-torch.linalg.vector_norm(act[:2] - corr["a_prev"][:2]) / a.sigma)
                resid = e_n - g * corr["B"] - torch.einsum("j,jchw->chw", du, corr["Je"])[None]
                nn = float(du @ du)
                if nn > 1e-8:
                    corr["Je"] += resid[0][None] * du[:, None, None, None] / (nn + 1e-8)
            corr["B"] = e_n.clone()
            corr["a_prev"] = act.clone()
        m = metrics(H1)
        real_gain = L_now - m["L"]
        rows.append(dict(step=n + 1, **m, pred_gain=pred_gain, real_gain=real_gain, gap_edit_mm=gap_raw,
                         gap_edit_corrected_mm=gap_cor, seconds=sec))
        np.save(out / "states" / f"iter_{n + 1:02d}.npy", H1[0, 0].cpu().numpy())
        print(f"    실행 {sec:.0f}s | 실제 L {L_now:.4f}->{m['L']:.4f} (이득 {real_gain:+.4f}, 예측 {pred_gain:+.4f}) | "
              f"홈 깊이 +{m['edit_depth_gain_mm']:.2f}mm, r={m['r']:.3f}, 보존 p={m['p']:.4f} (최대 {m['preserve_max_mm']:.2f}mm), "
              f"영역 밖 평균 {m['outside_zone_mean_mm']:.4f}mm | 모델 오차(편집영역) {gap_raw:.3f}mm -> 보정 예측 {gap_cor:.3f}mm")
        if m["L"] <= a.success_frac * m0["L"]:
            stop_reason = f"성공: L={m['L']:.4f} <= {a.success_frac} x 시작 L"
            print("    " + stop_reason)
            break

    final = state()[0, 0].cpu().numpy()
    np.save(out / "start_m.npy", start)
    np.save(out / "final_m.npy", final)
    with (out / "steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mf = rows[-1]
    (out / "summary.json").write_text(json.dumps(dict(
        history=str(root), restored_action=n0, executed_actions=len(actions), stop_reason=stop_reason,
        start={k: v for k, v in rows[0].items() if k not in ("step", "seconds")},
        final={k: v for k, v in mf.items() if k not in ("step", "seconds")},
        actions=actions, args={k: str(v) for k, v in vars(a).items()}), ensure_ascii=False, indent=2))
    print(f"[완료] 행동 {len(actions)}개 | L {m0['L']:.4f} -> {mf['L']:.4f} | 홈 깊이오차 r {m0['r']:.3f} -> {mf['r']:.3f}mm | "
          f"보존 p {m0['p']:.4f} -> {mf['p']:.4f}mm | 종료: {stop_reason}\n저장: {out}")


if __name__ == "__main__":
    main()
