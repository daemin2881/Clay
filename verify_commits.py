"""확정한 보상을 새 프로세스에서 Genesis 로 독립 재검증한다 (물리 검증).

cf_compensate.py 가 확정한 행동들은 실행 중 워커의 상태에서 이어서 만들어졌다. 여기서는 그 상태에 의존하지 않고
  1) 현재(K번째 행동 직후) 저장 상태를 새 Genesis 프로세스에서 복원해 기록된 높이맵과 맞는지(복원 오차) 확인하고
  2) rounds.json 의 확정 행동을 순서대로 다시 실행해, 확정 때 저장한 상태/최종 높이맵(final_m.npy)과 맞는지(재현 오차) 확인하고
  3) 같은 점수 J (영역 형태/기울기/테두리)를 '다시 실행한 결과'에서 재서 보고서의 값과 비교한다.
점토를 건드리는 저장 파일은 바꾸지 않는다. GPU/Genesis 가 필요하다(하트 조형 등 다른 Genesis 작업과 동시에 돌리면 느려질 수 있음).

사용: python verify_commits.py --history-dir H [--upto K] --cf-dir RUN/cf --regions DIR [--names a,b] [--out verify.json]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def drift(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) * 1000.0
    return dict(mae_mm=float(d.mean()), p99_9_mm=float(np.quantile(d, 0.999)), max_mm=float(d.max()), px_over_0_5mm=int((d > 0.5).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--cf-dir", required=True, type=Path, help="cf_compensate 출력 폴더 (rounds.json, states/, final_m.npy, summary.json)")
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--names", default="")
    ap.add_argument("--start-state", type=Path, help="이어서 시작한 보상이면 그 시작 상태(.pt). 기본: 이력의 K번째 post_state")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    import cf_common as cc
    import first_beam as fb
    from region_set import RegionSet
    root = cc.make_view(a.history_dir, a.upto, a.cf_dir / "verify_history_view")
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    _, actions, before, after = fb.load_history(root)
    K = len(actions)
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature
    if physics_signature(sim) != man["signature"]:
        raise SystemExit("physics signature 불일치: 이력을 만든 코드/설정/버전과 다릅니다.")
    adapter = GenesisStateAdapter(sim)
    start = a.start_state or (root / f"action_{K:06d}" / "post_state.pt")
    adapter.restore(load_snapshot(start))
    h0 = adapter.observe().numpy().astype(np.float64)
    rec = {"restore_error_vs_record": drift(h0, after[K - 1])} if a.start_state is None else {}
    print("복원 오차(기록된 현재 대비):", rec.get("restore_error_vs_record", "시작 상태를 직접 줌: 기록과 비교 안 함"))
    rounds = [r for r in json.loads((a.cf_dir / "rounds.json").read_text()) if r.get("committed")]
    print(f"확정 행동 {len(rounds)}개를 새 프로세스에서 다시 실행합니다.")
    steps, h = [], h0
    for i, r in enumerate(rounds, 1):
        act = r["action"]
        for a_i in (act if isinstance(act[0], (list, tuple)) else [act]):
            adapter.execute(torch.tensor(a_i, dtype=torch.float32), 0)
        h = adapter.observe().numpy().astype(np.float64)
        row = dict(commit=i, region=r.get("region"), family=r.get("family"), action=act)
        if (a.cf_dir / "states" / f"iter_{i:03d}.npy").exists():
            row["repro_vs_saved_state"] = drift(h, np.load(a.cf_dir / "states" / f"iter_{i:03d}.npy"))
        steps.append(row)
        print(f"  확정 {i} [{r.get('region')}, {r.get('family')}]: 저장된 상태 대비 재현 오차 {row.get('repro_vs_saved_state')}")
    final = a.cf_dir / "final_m.npy"
    out = dict(history=str(root), K=K, restore=rec, steps=steps)
    if final.exists():
        out["final_repro_vs_final_m"] = drift(h, np.load(final))
        print("최종 재현 오차(final_m.npy 대비):", out["final_repro_vs_final_m"])
    # 점수 J 를 다시 실행한 결과에서 잰다 (보고서 값과 비교)
    import cf_compensate as cfc
    summ = json.loads((a.cf_dir / "summary.json").read_text())
    w = summ.get("weights", {})
    rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_default=h0)
    # --level-adjust 로 돌린 실행은 '조정된 목표'(cf/level_adjust/<영역>_target_adjusted.npy)로 점수를 냈다. 같은 목표로 다시 채점해야 보고 값과 비교된다
    # (2026-11-16: 이걸 빼먹어서 hf3 fill3 의 보고 J 2.621 과 재채점 J 2.797 이 어긋났다. 조정 목표로 채점하면 소수 셋째 자리까지 일치한다)
    adj_names = []
    for n_ in rs.names:
        p_adj = a.cf_dir / "level_adjust" / f"{n_}_target_adjusted.npy"
        if p_adj.exists():
            rs.targets[n_] = np.load(p_adj).astype(np.float64)
            adj_names.append(n_)
    out["level_adjust_targets"] = adj_names
    g = cc.geometry(root)
    R = g["tool_r_px"]
    ref = np.load(a.cf_dir / "ref_m.npy").astype(np.float64) if (a.cf_dir / "ref_m.npy").exists() else h0
    cfg = cfc.build_cfg(rs, ref, w.get("mu", 1.0), int(round(0.94 * R)), int(round(1.17 * R)), R,
                        (w.get("height", 1.0), w.get("grad", 1.0), w.get("edge", 1.0)), w.get("blur_tools", 0.4), w_tail=w.get("tail", 0.0))
    s0, s1 = cfc.score(h0, cfg), cfc.score(h, cfg, h0)
    out["J_before"], out["J_after_rerun"] = s0["J"], s1["J"]
    # 같은 점수식을 '원래 목표' 기준으로도 (요청 기준과 반대로 움직이는 영역을 드러낸다)
    try:
        goals_o = cc.original_target_goals(rs, root, ref)
        cfg_o = cfc.build_cfg(RegionSet(rs.names, rs.masks, goals_o, rs.labels, ref, rs.meta), ref, 0.0, int(round(0.94 * R)), int(round(1.17 * R)), R,
                              (w.get("height", 1.0), w.get("grad", 1.0), w.get("edge", 1.0)), w.get("blur_tools", 0.4))
        o0, o1 = cfc.score(h0, cfg_o), cfc.score(h, cfg_o, h0)
        out["original_target"] = dict(e_before=o0["e"], e_after_rerun=o1["e"], regions={n: dict(before=list(o0["vis"][n]), after_rerun=list(o1["vis"][n])) for n in rs.names})
        print(f"[두 기준] 요청 기준 영역 점수 {s0['e']:.3f} -> {s1['e']:.3f} | 원래 목표 기준 {o0['e']:.3f} -> {o1['e']:.3f} (낮을수록 좋음)")
    except Exception as e_o:
        print(f"(원래 목표 기준 점수는 계산하지 못함: {e_o!r})")
    out["J_after_reported"] = summ.get("final", {}).get("J")
    out["regions"] = {n: dict(before=list(s0["vis"][n]), after_rerun=list(s1["vis"][n])) for n in rs.names}
    print(f"J(영역 평균): 시작 {s0['J']:.3f} -> 다시 실행한 결과 {s1['J']:.3f} (보고된 값 {out['J_after_reported']}; --own-score 실행이면 보고된 값은 마지막 라운드 영역 하나의 점수라 이 평균과 직접 비교할 수 없다)")
    out["region_J"] = {n: dict(before=float(cfc.region_J(n, s0["vis"], cfg)), after_rerun=float(cfc.region_J(n, s1["vis"], cfg))) for n in rs.names}
    for n in rs.names:
        v0, v1 = s0["vis"][n], s1["vis"][n]
        print(f"  {n:<14} 영역 점수 J {out['region_J'][n]['before']:.3f}->{out['region_J'][n]['after_rerun']:.3f} | 형태 {v0[2]:.2f}->{v1[2]:.2f}  기울기 {v0[0]:.2f}->{v1[0]:.2f}  테두리 {v0[1]:.2f}->{v1[1]:.2f}")
    np.save(a.cf_dir / "verify_final_m.npy", h.astype(np.float32))
    (a.out or a.cf_dir / "verify_commits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("저장:", a.out or a.cf_dir / "verify_commits.json")


if __name__ == "__main__":
    main()
