"""과거 시퀀스를 고쳐 다시 재생한 '수정된 타임라인'을 만든다 (반사실적 과거 국소 보상의 직접 결과).

  cf_compensate 의 트리 탐색(--long-select tree)이 찾은 원인 행동들(예: 홈을 판 #2, #42~#55, #71~#75)을 한꺼번에 고쳐서(깊이 x1.5 등) 가장 이른 행동 직전의 저장 스냅샷에서
  현재(K 번째)까지 Genesis 로 다시 실행한다. 끝난 상태가 '그때 달랐다면 지금 이랬을 점토'이고, 이 상태가 보상 결과 자체다(현재에서 다시 누를 필요 없음).
  점토 입자 상태(post_state.pt)를 함께 저장해서 이어서 조형하거나 새 프로세스에서 다시 재생해 재현성을 확인할 수 있다.

  - control 은 기록된 현재 상태를 쓴다(--control record 와 같은 근거: A 에서 52개 쌍 검증, 효과 차이 최대 0.022, 부호 100% 일치).
  - 영역별 점수(요청 기준)와 세계 전체 변화를 현재 대비로 출력한다.
  - 편집은 '행동번호:길이배율:깊이배율[:옆이동mm]' (cf_compensate 의 변형과 같은 표기).

사용: python revise_timeline.py --history-dir H --upto 80 --edits 2:1:1.5,42:1:1.5,47:1:1.5 --regions DIR --names a,b --out OUT
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--edits", required=True, help="쉼표로 구분: 행동번호:길이배율:깊이배율[:옆이동mm]")
    ap.add_argument("--regions", type=Path, help="영역 점수를 낼 영역 폴더(선택)")
    ap.add_argument("--names", default="")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-save-state", action="store_true")
    a = ap.parse_args()
    import cf_common as cc
    import cf_compensate as cfc
    import first_beam as fb
    a.out.mkdir(parents=True, exist_ok=True)
    root = cc.make_view(a.history_dir, a.upto, a.out / "history_view")
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    _, actions, before, after = fb.load_history(root)
    K = len(actions)
    cfc.BOUNDS = cc.action_bounds(2.0)
    edits = {}
    for tok in [t for t in a.edits.split(",") if t.strip()]:
        v = [float(x) for x in tok.split(":")]
        m, lf, df = int(v[0]), v[1], v[2]
        edits[m] = cfc.variant_action(actions[m - 1], lf, (df, v[3]) if len(v) > 3 else df)
    s0 = min(edits)
    print(f"[수정된 타임라인] 이력 {a.history_dir.name} K={K} | 고친 행동 {len(edits)}개: " + ", ".join(f"#{m}" for m in sorted(edits)) + f" | #{s0} 직전 스냅샷에서 #{K} 까지 {K - s0 + 1}행동 재생", flush=True)
    for m in sorted(edits):
        o, n = [float(x) for x in actions[m - 1]], edits[m]
        print(f"    #{m}: 길이 {o[3] * 1000:.1f}->{n[3] * 1000:.1f}mm 깊이 {o[4] * 1000:.1f}->{n[4] * 1000:.1f}mm 위치 ({o[0]:.4f},{o[1]:.4f})->({n[0]:.4f},{n[1]:.4f})", flush=True)

    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import torch
    import vs_simulator as sim
    import counterfactual_replay as cfr
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature, tensor_save
    if physics_signature(sim) != man["signature"]:
        raise SystemExit("physics signature 불일치: 이력을 만든 코드/설정/버전과 다릅니다.")
    adapter = GenesisStateAdapter(sim)
    adapter.restore(load_snapshot(root / f"action_{K:06d}" / "post_state.pt"))
    pre, records, rngs = cfr.load_plan(Path(root), s0)
    t0 = time.time()
    h_rep, _ = cfr.run_replay(adapter, pre, records, rngs, s0, K, {m: torch.tensor(v, dtype=torch.float32) for m, v in edits.items()})
    h_rep = np.asarray(h_rep, dtype=np.float64)
    print(f"재생 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    np.save(a.out / "h_revised.npy", h_rep.astype(np.float32))
    if not a.no_save_state:
        tensor_save(a.out / "post_state.pt", adapter.capture())
    rec = after[K - 1].astype(np.float64)
    (a.out / "edits.json").write_text(json.dumps(dict(history=str(a.history_dir), K=K, start=s0, edits={str(m): v for m, v in edits.items()}), ensure_ascii=False, indent=1))
    d = (h_rep - rec) * 1000
    print(f"세계 전체 변화: 평균 |Δ| {np.abs(d).mean():.3f} mm, 최대 {np.abs(d).max():.1f} mm | 저장: {a.out / 'h_revised.npy'}" + ("" if a.no_save_state else f", {a.out / 'post_state.pt'}"), flush=True)
    if a.regions is not None:
        from region_set import RegionSet
        g = cc.geometry(root)
        R = g["tool_r_px"]
        rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_default=rec)
        cfg = cfc.build_cfg(rs, rec, 0.0, int(round(0.94 * R)), int(round(1.17 * R)), R, w_tail=1.0)
        s_a, s_b = cfc.score(rec, cfg), cfc.score(h_rep, cfg, rec)
        print(f"\n{'영역':<14}{'현재':>10}{'수정된 타임라인':>16}{'변화':>9}   (형태, 기울기, 테두리, 꼬리)  영역 평균 높이 변화")
        for n in rs.names:
            x, y = cfc.region_J(n, s_a["vis"], cfg), cfc.region_J(n, s_b["vis"], cfg)
            print(f"{n:<14}{x:>10.3f}{y:>16.3f}{y - x:>+9.3f}   {[round(v, 2) for v in s_a['vis'][n]]} -> {[round(v, 2) for v in s_b['vis'][n]]}   {((h_rep - rec)[rs.masks[n]].mean() * 1000):+.2f} mm")
        print(f"{'영역 평균 e':<14}{s_a['e']:>10.3f}{s_b['e']:>16.3f}{s_b['e'] - s_a['e']:>+9.3f}")
        m0 = rs.masks[rs.names[0]]
        print(f"{rs.names[0]}: 최저점 변화 {((h_rep - rec)[m0].min() * 1000):+.2f} mm (현재 푸시 방식 v16 은 -6.06 mm 구덩이)")


if __name__ == "__main__":
    main()
