"""기록된 이력의 한 행동을 '실수'로 바꿔 재생하고, 그 결과를 새 이력 폴더로 저장한다.

목적: 원인이 최근 과거에 있는, 정답을 아는 오차 상황을 만든다.
  - 깨끗한 최종 상태 H*(원래 이력의 마지막 상태)가 '실수가 없었다면 도달했을 목표'다.
  - 실수 이력의 최종 상태 H_f 가 '현재'이고, 오차 = H_f - H* 의 원인은 --fault-action 이다.

새 이력은 --start 행동 직전 상태에서 시작해 --start..마지막 행동을 순서대로 재생한 결과다.
행동 번호는 1부터 다시 매긴다(새 이력의 1번 = 원래의 --start 번). 이력 형식은 ActionHistory와 같아서
first_beam.py / counterfactual_replay.py / continue_sculpt.py 를 그대로 쓸 수 있다.

사용 예
  python inject_fault.py --history-dir HIST --start 171 --fault-action 171 --depth-factor 1.6 \
      --roi roi.npy --out results/case_A/fault171
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--start", required=True, type=int, help="재생을 시작할 원래 행동 번호 (그 행동 직전 상태에서 시작)")
    ap.add_argument("--fault-action", type=int, help="실수를 넣을 원래 행동 번호 (기본: --start)")
    ap.add_argument("--depth-factor", type=float, default=1.0)
    ap.add_argument("--length-factor", type=float, default=1.0)
    ap.add_argument("--roi", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    root = a.history_dir.expanduser().resolve()
    out = a.out.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"

    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    last = len(dirs)
    fault = a.fault_action or a.start
    if not (1 <= a.start <= fault <= last):
        raise ValueError(f"--start <= --fault-action <= {last} 이어야 합니다.")
    if a.depth_factor == 1.0 and a.length_factor == 1.0:
        raise ValueError("실수를 넣으려면 --depth-factor 또는 --length-factor가 1이 아니어야 합니다.")

    import vs_config as cfg
    import vs_simulator as sim
    from vs_history import ActionHistory, load_snapshot, physics_signature, restore_rng, verify_action_files

    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")

    records = {}
    for n in range(a.start, last + 1):
        verify_action_files(dirs[n - 1])
        records[n] = json.loads((dirs[n - 1] / "record.json").read_text())
    pre = load_snapshot(dirs[a.start - 1] / "pre_state.pt")

    hist = ActionHistory(sim, out, target=torch.from_numpy(np.load(root / "target_used_512_m.npy")),
                         raw_target=torch.from_numpy(np.load(root / "target_raw_512_m.npy")),
                         roi_path=a.roi.expanduser(), checkpoint_path=None, target_transform=None,
                         run_args=dict(fault=dict(source=str(root), start=a.start, fault_action=fault,
                                                  depth_factor=a.depth_factor, length_factor=a.length_factor)))
    hist.adapter.restore(pre)

    orig_fault = torch.tensor(records[fault]["action"], dtype=torch.float64)
    for t, n in enumerate(range(a.start, last + 1)):
        restore_rng(load_snapshot(dirs[n - 1] / "pre_state.pt")["rng"])
        act = torch.tensor(records[n]["action"], dtype=torch.float64)
        if n == fault:
            act[3] = float(np.clip(act[3] * a.length_factor, cfg.LENGTH_MIN, cfg.LENGTH_MAX))
            act[4] = float(np.clip(act[4] * a.depth_factor, cfg.DEPTH_MIN, cfg.DEPTH_MAX))
            print(f"[실수 주입] 원래 행동 {n}: 길이 {orig_fault[3] * 1000:.1f}→{act[3] * 1000:.1f}mm, "
                  f"깊이 {orig_fault[4] * 1000:.1f}→{act[4] * 1000:.1f}mm")
        before = hist.adapter.observe()
        hist.record_push(act.float(), action_index=t, before=before, predicted=before)
        print(f"[재생 {t + 1}/{last - a.start + 1}] 원래 행동 {n}")

    h_clean = np.load(dirs[last - 1] / "transition.npz")["after_m"].astype(np.float32)
    h_fault = np.load(out / f"action_{last - a.start + 1:06d}" / "transition.npz")["after_m"].astype(np.float32)
    np.save(out / "h_clean_final_512_m.npy", h_clean)
    roi = np.load(a.roi.expanduser()).astype(bool)
    d = np.abs(h_fault - h_clean)
    summary = dict(source=str(root), start=a.start, fault_action=fault, depth_factor=a.depth_factor,
                   length_factor=a.length_factor, n_actions=last - a.start + 1,
                   roi_mae_fault_vs_clean_mm=float(d[roi].mean()) * 1000,
                   global_mae_fault_vs_clean_mm=float(d.mean()) * 1000,
                   outside_roi_max_mm=float(d[~roi].max()) * 1000, roi_max_mm=float(d[roi].max()) * 1000)
    (out / "fault_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[결과] 실수 이력 vs 깨끗한 최종: ROI MAE {summary['roi_mae_fault_vs_clean_mm']:.4f}mm, "
          f"전체 MAE {summary['global_mae_fault_vs_clean_mm']:.4f}mm, ROI 밖 최대 {summary['outside_roi_max_mm']:.2f}mm")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
