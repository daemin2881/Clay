"""설계 §4 + §5(7단계): 과거 행동 기여 점수 -> 계층 beam -> Param2Deform 반사실 스크리닝.

  D_t = H_{t+1} - H_t,  G = H*_edit - H_T
  s_t = w1*cos(M_e*D_t, M_e*G) + w2*|M_e*D_t|_1/|D_t|_1 + w3*IoU(P_t, M_e)
        - w4*|M_p*D_t|_1/|D_t|_1 - w5*e_t
    (w4 항은 설계의 |M_p*D_t|_1 을 |D_t|_1 로 나눠 비율로 썼다: 항 크기를 다른 항과 맞추기 위함)
    e_t = clip(model_gap / deformation, 0, 2) / 2   (Param2Deform 이 그 행동에서 얼마나 틀렸나)
  계층 beam: 5행동 묶음(겹치지 않음) 점수 -> 상위 3개 묶음 -> 그 안의 개별 행동 순위 -> 상위 N개
  스크리닝:  H_cf ~= H_T + [f(H_k, a'_k) - f(H_k, a_k)],  a' = (l*lf, d*df), lf,df in {0.7,1,1.3}, (1,1) 제외
             q_k = L(H_T) - L(H_cf)   (>0 이면 그 행동을 다르게 했을 때 요청한 홈 모양에 가까워짐)
  이 값은 후속 행동의 영향을 무시하는 1차 근사다. 실제 효과는 rollback_cf.py(Genesis)로 확인한다.
  q_k 가 --min-q 를 넘는 후보가 없으면 abstain: 과거 원인을 찾지 못한 것이므로 C* 없이 현재 상태 보정만 한다.

사용: python attribution.py --history-dir H --spec-dir S --checkpoint CK --out DIR
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import first_beam as fb
from edit_spec import load_spec

W = dict(w1=1.0, w2=1.0, w3=1.0, w4=1.0, w5=0.5)


def load_history_full(root: Path):
    manifest, actions, before, after = fb.load_history(root)
    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    recs = [json.loads((d / "record.json").read_text()) for d in dirs]
    gap = np.array([r["metrics"]["model_gap_mae_mm"] for r in recs])
    deform = np.array([r["metrics"]["deformation_mae_mm"] for r in recs])
    e = np.clip(gap / np.maximum(deform, 1e-6), 0.0, 2.0) / 2.0
    return manifest, actions, before, after, e


def item_score(D_mm, P, e, me, mp, G_mm, w):
    eps = 1e-9
    d, g = D_mm[me], G_mm[me]
    cos = float(d @ g / (np.linalg.norm(d) * np.linalg.norm(g) + eps))
    l1 = float(np.abs(D_mm).sum()) + eps
    conc = float(np.abs(D_mm[me]).sum()) / l1
    iou = float((P & me).sum() / max((P | me).sum(), 1))
    pres = float(np.abs(D_mm[mp]).sum()) / l1
    s = w["w1"] * cos + w["w2"] * conc + w["w3"] * iou - w["w4"] * pres - w["w5"] * e
    return dict(score=float(s), cos=cos, conc=conc, iou=iou, pres=pres, e=float(e))


@torch.no_grad()
def screen(model, spec, predict_next_batch, before_k, a_k, h_T, grid, ranges, device):
    (lmin, lmax), (dmin, dmax) = ranges
    pairs = [p for p in itertools.product(grid, grid) if p != (1.0, 1.0)]
    acts = [torch.as_tensor(a_k, dtype=torch.float32)]
    for lf, df in pairs:
        a = torch.as_tensor(a_k, dtype=torch.float32).clone()
        a[3] = float(np.clip(a[3] * lf, lmin, lmax))
        a[4] = float(np.clip(a[4] * df, dmin, dmax))
        acts.append(a)
    state = torch.from_numpy(before_k)[None, None].to(device)
    pred = predict_next_batch(model, state, torch.stack(acts).to(device))
    delta = pred[1:] - pred[0:1]
    h_cf = h_T + delta
    L_cf, t_cf = spec.loss(h_cf, return_terms=True)
    L0 = float(spec.loss(h_T))
    out = []
    for i, (lf, df) in enumerate(pairs):
        out.append(dict(lf=lf, df=df, q=L0 - float(L_cf[i]),
                        dr=float(spec.terms(h_T)["r"] - t_cf["r"][i]), dp=float(t_cf["p"][i]),
                        effect_edit_mm=float(delta[i][0][spec.m_edit[0, 0]].abs().mean()) * 1000))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--spec-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--lookback", type=int, default=0, help="최근 N개 행동만 후보로 (0이면 전부)")
    ap.add_argument("--group", type=int, default=5)
    ap.add_argument("--top-groups", type=int, default=3)
    ap.add_argument("--top-actions", type=int, default=3)
    ap.add_argument("--grid", default="0.7,1.0,1.3")
    ap.add_argument("--min-q", type=float, default=0.0, help="이 값 이하의 q 는 후보로 인정하지 않는다(abstain)")
    for k, v in W.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from vs_config import LENGTH_MIN, LENGTH_MAX, DEPTH_MIN, DEPTH_MAX
    from vs_dynamics import Param2Deform
    from vs_planner import predict_next_batch

    root = a.history_dir.expanduser().resolve()
    manifest, actions, before, after, e = load_history_full(root)
    T = len(actions)
    g = fb.map_geometry(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_spec(a.spec_dir, device)
    me, mp = (m[0, 0].cpu().numpy() for m in (spec.m_edit, spec.m_preserve))
    h_T = spec.h_t
    if float(np.abs(after[-1] - spec.h_t[0, 0].cpu().numpy()).max()) > 1e-6:
        raise RuntimeError("spec 의 H_T 가 이력의 마지막 상태와 다릅니다.")
    G = (spec.h_star[0, 0] - spec.h_t[0, 0]).cpu().numpy() * 1000.0
    w = {k: getattr(a, k) for k in W}
    first = max(0, T - a.lookback) if a.lookback else 0

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = Param2Deform()
    model.load_state_dict(ck["model"])
    model = model.to(device=device, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    del ck

    # 1) 개별 행동 점수 + 묶음 점수
    fp = {t: fb.footprint(g, actions[t]) for t in range(first, T)}
    singles = {t: dict(t=t, number=t + 1, action=actions[t].tolist(),
                       **item_score((after[t] - before[t]) * 1000.0, fp[t], e[t], me, mp, G, w))
               for t in range(first, T)}
    groups = []
    for s in range(first, T - a.group + 1, a.group):
        ee = s + a.group - 1
        P = np.any([fp[t] for t in range(s, ee + 1)], axis=0)
        groups.append(dict(start=s + 1, end=ee + 1,
                           **item_score((after[ee] - before[s]) * 1000.0, P, e[s:ee + 1].mean(), me, mp, G, w)))
    top_groups = sorted(groups, key=lambda x: -x["score"])[:a.top_groups]
    members = sorted((singles[k - 1] for gp in top_groups for k in range(gp["start"], gp["end"] + 1)),
                     key=lambda x: -x["score"])
    picked = members[:a.top_actions]

    # 2) 스크리닝: 선택된 개별 행동 + (검증용) 모든 행동
    grid = [float(x) for x in a.grid.split(",")]
    ranges = ((LENGTH_MIN, LENGTH_MAX), (DEPTH_MIN, DEPTH_MAX))
    L0 = float(spec.loss(h_T))
    allq = {}
    for t in range(first, T):
        allq[t] = screen(model, spec, predict_next_batch, before[t], actions[t], h_T, grid, ranges, device)
    for it in picked:
        it["screen"] = sorted(allq[it["t"]], key=lambda r: -r["q"])
        it["best_q"] = it["screen"][0]["q"]
    best_over_all = sorted(((max(r["q"] for r in allq[t]), t + 1) for t in allq), reverse=True)[:5]
    s_arr = np.array([singles[t]["score"] for t in range(first, T)])
    q_arr = np.array([max(r["q"] for r in allq[t]) for t in range(first, T)])
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(s_arr)), np.argsort(np.argsort(q_arr)))[0, 1])

    cands = [(it, r) for it in picked for r in it["screen"] if r["q"] > a.min_q]
    chosen = None
    if cands:
        it, r = max(cands, key=lambda x: x[1]["q"])
        chosen = dict(action_number=it["number"], t=it["t"], lf=r["lf"], df=r["df"], q_model=r["q"], score=it["score"])
    report = dict(schema=1, n_actions=T, first=first + 1, weights=w, L_H_T=L0,
                  top_groups=top_groups, picked=picked, chosen=chosen, abstain=chosen is None, min_q=a.min_q,
                  validation=dict(spearman_score_vs_best_q=rank_corr,
                                  top5_actions_by_model_q=[dict(action_number=n, q=q) for q, n in best_over_all],
                                  n_actions_with_q_gt_min=int((q_arr > a.min_q).sum())))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "attribution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (a.out / "all_actions.json").write_text(json.dumps(
        [dict(**{k: v for k, v in singles[t].items() if k not in ("action", "screen", "best_q")},
              best_q=max(r["q"] for r in allq[t])) for t in range(first, T)], indent=1))

    print(f"L(H_T)={L0:.4f}  행동 {T}개 중 {T - first}개 검토, 5행동 묶음 {len(groups)}개")
    print("상위 묶음:", ", ".join(f"{x['start']}-{x['end']} s={x['score']:+.3f} (cos {x['cos']:+.2f} 집중 {x['conc']:.3f} IoU {x['iou']:.3f})" for x in top_groups))
    print("선택 행동 (개별 s_t 순):")
    for it in picked:
        b = it["screen"][0]
        print(f"  #{it['number']:3d} s={it['score']:+.3f} cos={it['cos']:+.2f} 집중={it['conc']:.3f} IoU={it['iou']:.3f} "
              f"보존침범={it['pres']:.2f} e={it['e']:.2f} | 최선 CF l×{b['lf']:g} d×{b['df']:g}: q={b['q']:+.4f}")
    print(f"검증: 모든 행동 중 모델 q 최대 5개 = {[(n, round(q, 4)) for q, n in best_over_all]}, "
          f"점수-q 순위상관 = {rank_corr:+.2f}, q>{a.min_q} 인 행동 {int((q_arr > a.min_q).sum())}/{len(q_arr)}")
    print("=> abstain (과거 원인 후보 없음)" if chosen is None else
          f"=> 후보: 행동 {chosen['action_number']} l×{chosen['lf']:g} d×{chosen['df']:g} (모델 q={chosen['q_model']:+.4f}) -> rollback_cf.py 로 검증")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
