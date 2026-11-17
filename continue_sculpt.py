"""완성된 조형 상태를 점토 초기화 없이 복원해서, 주어진 목표로 이어서 계획·실행한다.

비교 실험의 두 팔을 같은 코드로 돌리기 위한 실행기다.
  - 기준(baseline): 수정 목표 H_edit 를 그대로 --target 으로 준다 (과거 보상 없이 재계획)
  - 과거 국소보상  : 반사실 재생으로 얻은 가상의 현재(H_cf)를 투영한 목표를 --target 으로 준다

두 팔 모두 --eval-target(최종 평가용 H_edit)에 대한 ROI 오차로 비교한다.
새 폴더만 쓰고 이력 폴더는 읽기만 한다. 복원 상태의 시작 높이맵이 기록과 일치하는지 먼저 확인한다.

사용 예
  python continue_sculpt.py --history-dir HIST --target T.npy --eval-target H_edit.npy --roi roi.npy \
      --checkpoint checkpoints/xxx.pt --num-actions 20 --out-dir results/case_A/arm_baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch


def load_map(path):
    a = np.load(Path(path).expanduser(), allow_pickle=False)
    if a.shape != (512, 512) or not np.isfinite(a).all():
        raise ValueError(f"512×512 유한 배열이 필요합니다: {path}")
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--restore-action", type=int, help="이 번호 행동 직후의 상태를 복원 (기본: 마지막)")
    ap.add_argument("--start-state", type=Path, help="이력 행동 대신 이 상태 스냅샷(.pt, 예: cf_compensate 의 states/iter_XXX.pt)에서 시작한다. 최근 행동 이력(반복 억제용)은 --restore-action 의 것을 쓴다")
    ap.add_argument("--target", required=True, type=Path, help="계획이 따라갈 목표 (512x512, 단위 m, 절대 높이)")
    ap.add_argument("--eval-target", type=Path, help="최종 평가용 목표. 기본: --target")
    ap.add_argument("--roi", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--num-actions", type=int, default=20)
    ap.add_argument("--plan-horizon", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--candidate-batch-size", type=int, default=1)
    ap.add_argument("--refine-actions", type=int, default=20)
    ap.add_argument("--refine-steps", type=int, default=5)
    ap.add_argument("--mpc-execute", type=int, default=5)
    ap.add_argument("--w3d", type=float, default=1.0)
    ap.add_argument("--wviz", type=float, default=1.0)
    ap.add_argument("--seed", type=int, help="계획기의 무작위 후보 추출 시드. 상태 복원이 난수 상태까지 되돌리므로 시드를 주지 않으면 같은 상태에서 시작한 반복이 거의 같은 계획을 만든다. "
                                             "반복 실행마다 다른 시드를 주면 계획기 변동을 잴 수 있고, 같은 시드는 재현에 쓴다")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"새 폴더를 지정하세요: {out}")
    out.mkdir(parents=True, exist_ok=True)
    root = args.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"

    import vs_simulator as sim
    from vs_config import RECENT_ACTION_HISTORY
    from vs_dynamics import Param2Deform
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature
    from vs_planner import greedy_plan_gpu, refine_plan_gpu

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다. 이 이력은 복원할 수 없습니다.")

    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    n0 = args.restore_action or len(dirs)
    if not 1 <= n0 <= len(dirs):
        raise ValueError(f"--restore-action은 1~{len(dirs)} 이어야 합니다.")
    adapter.restore(load_snapshot(args.start_state if args.start_state else root / f"action_{n0:06d}" / "post_state.pt"))

    if args.seed is not None:                       # 복원(restore_rng) 뒤에 건다: 복원이 저장된 난수 상태로 덮어쓰기 때문
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[시드] {args.seed} (상태 복원 뒤에 설정)")
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
    if args.start_state:
        print(f"[복원] 스냅샷 {args.start_state} (행동 {n0} 직후 이력의 최근 행동을 반복 억제에 사용). 기록 대비 차이 {restore_err:.4f}mm는 보상 때문에 생긴 변화")
    else:
        print(f"[복원] 행동 {n0} 직후 상태, 기록 대비 최대 차이 {restore_err:.4f}mm")
        if restore_err > 1e-3:
            raise RuntimeError("복원한 높이맵이 기록과 다릅니다.")

    target_np = load_map(args.target)
    eval_np = load_map(args.eval_target) if args.eval_target else target_np
    roi = np.load(args.roi.expanduser()).astype(bool)
    target = torch.from_numpy(target_np).to(device=device, dtype=dtype)[None, None]

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Param2Deform()
    model.load_state_dict(ck["model"])
    del ck
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 최근 행동 이력(반복 억제용)은 기록된 마지막 행동들로 채운다.
    recent = []
    for d in dirs[max(0, n0 - RECENT_ACTION_HISTORY):n0]:
        recent.append(torch.tensor(json.loads((d / "record.json").read_text())["action"], device=device, dtype=dtype))

    def metrics(h):
        d = np.abs(h - eval_np)
        return dict(roi_mae_mm=float(d[roi].mean()) * 1000, global_mae_mm=float(d.mean()) * 1000,
                    outside_change_mm=float(np.abs(h - start)[~roi].mean()) * 1000,
                    outside_change_max_mm=float(np.abs(h - start)[~roi].max()) * 1000)

    rows, actions = [], []
    m0 = metrics(start)
    rows.append(dict(step=0, **m0, seconds=0.0))
    print(f"[시작] ROI MAE(평가목표 대비) {m0['roi_mae_mm']:.4f}mm | 전체 {m0['global_mae_mm']:.4f}mm")
    done = 0
    while done < args.num_actions:
        cur = state()
        horizon = min(args.plan_horizon, args.num_actions - done)
        rec = torch.stack(recent, 0) if recent else None
        plan = greedy_plan_gpu(model, cur, target, horizon=horizon, candidates=args.candidates,
                               w3d=args.w3d, wviz=args.wviz, recent_actions=rec,
                               candidate_batch_size=args.candidate_batch_size).to(device=device, dtype=dtype)
        if args.refine_steps > 0 and args.refine_actions > 0:
            plan = refine_plan_gpu(model, cur, target, plan, w3d=args.w3d, wviz=args.wviz, recent_actions=rec,
                                   max_refine_actions=args.refine_actions, refine_steps=args.refine_steps)
            plan = plan.to(device=device, dtype=dtype)
        for i in range(min(args.mpc_execute, plan.shape[0], args.num_actions - done)):
            a = plan[i].detach()
            t0 = time.perf_counter()
            sim.linear_push(a, action_index=n0 + done)
            sec = time.perf_counter() - t0
            done += 1
            recent = (recent + [a.clone()])[-RECENT_ACTION_HISTORY:]
            h = state()[0, 0].cpu().numpy()
            m = metrics(h)
            rows.append(dict(step=done, **m, seconds=sec))
            (out / "states").mkdir(exist_ok=True)
            np.save(out / "states" / f"step_{done:03d}.npy", h.astype(np.float32))       # 단계별 높이맵(영역 점수·그림 비교용)
            actions.append([float(v) for v in a.cpu()])
            print(f"[{done:03d}] ROI MAE {m['roi_mae_mm']:.4f}mm | 전체 {m['global_mae_mm']:.4f}mm | "
                  f"ROI 밖 변화 평균 {m['outside_change_mm']:.4f}mm 최대 {m['outside_change_max_mm']:.2f}mm | "
                  f"행동 길이 {float(a[3]) * 1000:.1f}mm 깊이 {float(a[4]) * 1000:.1f}mm")

    final = state()[0, 0].cpu().numpy()
    np.save(out / "start_m.npy", start)
    np.save(out / "final_m.npy", final)
    with (out / "steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mf = rows[-1]
    (out / "summary.json").write_text(json.dumps(dict(
        history=str(root), restored_action=n0, num_actions=args.num_actions, target=str(args.target),
        eval_target=str(args.eval_target or args.target), checkpoint=str(args.checkpoint),
        restore_max_diff_mm=restore_err, start=m0, final={k: v for k, v in mf.items() if k not in ("step", "seconds")},
        roi_mae_change_mm=mf["roi_mae_mm"] - m0["roi_mae_mm"], actions=actions,
        args={k: str(v) for k, v in vars(args).items()}), ensure_ascii=False, indent=2))
    print(f"[완료] ROI MAE {m0['roi_mae_mm']:.4f} → {mf['roi_mae_mm']:.4f}mm "
          f"({mf['roi_mae_mm'] - m0['roi_mae_mm']:+.4f}) | 저장: {out}")


if __name__ == "__main__":
    main()
