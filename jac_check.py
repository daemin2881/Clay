"""cf_jacobian(JAC) 가 쓸모 있는지 Genesis 없이 저장된 후보 결과로 검증한다 (--save-transitions 폴더 + 확정 상태).

기준 행동 a0(물리 J 가 가장 낮은 후보) 주변에서, 물리로 실행해 본 이웃 a_k 들의 실제 J 변화 dJ_k = J_phys(a_k) - J_phys(a0) 를 다음이 얼마나 맞히나:
  model   모델 예측 J 의 변화  J_model(a_k) - J_model(a0)
  lin0    모델 기울기(유한 차분) 선형 예측  g0.(a_k-a0)
  broyden 하나를 빼고(leave-one-out) 나머지 이웃으로 Broyden 보정한 뒤의  [J_model(a_k)-J_model(a0)] + (g-g0).(a_k-a0)
지표: 부호 일치율(좋아질지 나빠질지), 순위 상관, 평균 절대 오차(J 단위).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy import stats

import cf_common as cc
import cf_compensate as cfc
import cf_jacobian as jac
from region_set import RegionSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--transitions", required=True, type=Path)
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--tag", type=int, default=0)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/param2deform_medium_paper_planner_range_deep150.pt"))
    ap.add_argument("--bases", type=int, default=3)
    a = ap.parse_args()
    import torch
    from vs_dynamics import Param2Deform
    root = a.history_dir.resolve()
    R = cc.geometry(root)["tool_r_px"]
    cfc.BOUNDS = cc.action_bounds(2.0)
    h0 = np.load(a.states / f"iter_{a.tag:03d}.npy").astype(np.float64)
    rs = RegionSet.load(a.regions, ref_default=h0)
    cfg = cfc.build_cfg(rs, h0, 1.0, int(round(0.94 * R)), int(round(1.17 * R)), R)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = Param2Deform()
    model.load_state_dict(ck["model"])
    model = model.eval()
    score_fn = lambda m: cfc.score(m, cfg)["J"]
    cands = []
    for f in sorted(glob.glob(str(a.transitions / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if int(z["state_tag"]) == a.tag:
            cands.append((np.asarray(z["action"], float), score_fn(h0 + z["delta"].astype(np.float64))))
    cands.sort(key=lambda t: t[1])
    print(f"상태 iter_{a.tag:03d}: 물리 평가 후보 {len(cands)}개 (J {cands[0][1]:.3f} ~ {cands[-1][1]:.3f}), 현재 J {score_fn(h0):.3f}")
    allrows = []
    for bi in range(a.bases):
        a0, J0 = cands[bi]
        g0 = jac.fd_gradient(model, h0, a0, score_fn, cfc.clampA)
        Jm0 = jac.model_J(model, h0, [list(a0)], score_fn)[0]
        nb = [(x, j) for x, j in cands if 1e-6 < np.linalg.norm(jac.delta_units(x, a0)) <= a.radius]
        print(f"\n기준 #{bi} J_phys={J0:.3f} 모델 J={Jm0:.3f} | 모델 기울기 " + " ".join(f"{n}{v:+.3f}" for n, v in zip(jac.NAMES, g0)) + f" | 반경 {a.radius:g} 이내 물리 이웃 {len(nb)}개")
        if len(nb) < 3:
            continue
        Jm = jac.model_J(model, h0, [list(x) for x, _ in nb], score_fn)
        for k, (x, jp) in enumerate(nb):
            du = jac.delta_units(x, a0)
            others = [(y, jy) for i, (y, jy) in enumerate(nb) if i != k]
            g, n_used = jac.broyden_correct(g0, a0, J0, others, a.radius)
            allrows.append(dict(dJ=jp - J0, model=Jm[k] - Jm0, lin0=float(g0 @ du), broyden=float((Jm[k] - Jm0) + (g - g0) @ du), n=n_used))
    if not allrows:
        print("이웃이 모자라 검증 불가")
        return
    dJ = np.array([r["dJ"] for r in allrows])
    print(f"\n이웃 총 {len(dJ)}개(기준 {a.bases}개 합침) | 실제 J 변화 평균 {dJ.mean():+.3f}, 표준편차 {dJ.std():.3f}")
    print(f"{'예측 방식':<10}{'부호 일치':>10}{'순위 상관':>10}{'평균 절대 오차':>14}")
    for key, nm in (("model", "모델 예측"), ("lin0", "모델 선형"), ("broyden", "Broyden 보정(LOO)")):
        p = np.array([r[key] for r in allrows])
        sg = float(np.mean(np.sign(p) == np.sign(dJ)))
        rho = stats.spearmanr(p, dJ)[0] if p.std() > 0 else float("nan")
        print(f"{nm:<20}{sg:>8.0%}{rho:>+10.2f}{np.abs(p - dJ).mean():>14.3f}")
    print(f"(영 예측 = 변화 없음의 평균 절대 오차 {np.abs(dJ).mean():.3f})")


if __name__ == "__main__":
    main()
