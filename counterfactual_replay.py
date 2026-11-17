"""First-Beam이 고른 과거 구간에서 길이/깊이를 바꿔 Genesis로 재생하고 ROI 개선을 측정한다.

baseline 폴더(vs_history.py, vs_simulator.py가 있는 곳)에 두고 실행한다.

흐름
  1) 선택 구간 시작 직전 상태 S_s 를 복원
  2) control : 원래 행동 그대로 마지막 행동까지 재생 -> H_ctrl  (재현 drift 확인용 대조군)
  3) 대안     : 구간 [s, e] 의 length/depth 에 배율 f 를 곱하고 이후 행동은 그대로 재생 -> H_cf(f)
  4) 비교     : H_edit 대비 ROI MAE, ROI 밖 보존 오차를 H_ctrl 기준으로 계산

비교 기준을 기록된 H_T가 아니라 같은 엔진에서 다시 돌린 H_ctrl로 두기 때문에,
엔진 미세 drift가 있어도 개선량이 오염되지 않는다. 단 control drift가 기준을 넘으면 경고한다.

사용 예
  python counterfactual_replay.py --history-dir runs/heart_hist --beam-json runs/heart_hist/first_beam.json \
      --h-edit case_heart/edit/h_edit_512_m.npy --beam-rank 1 --factors 0.7,1.3
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from vs_history import (GenesisStateAdapter, json_write, load_snapshot, physics_signature,
                        restore_rng, sha256, verify_action_files)


def scaled(action, factor, mode, length_range, depth_range):
    a = torch.as_tensor(action, dtype=torch.float64).clone()
    if mode in ("both", "length"):
        a[3] = float(np.clip(a[3] * factor, *length_range))
    if mode in ("both", "depth"):
        a[4] = float(np.clip(a[4] * factor, *depth_range))
    return a.float()


def load_plan(root, start):
    """시작 상태 스냅샷과, 모든 행동의 기록 행동/난수 상태를 읽는다."""
    folders = sorted(p for p in root.glob("action_??????") if p.is_dir())
    records, rngs = {}, {}
    for d in folders:
        n = int(d.name.split("_")[1])
        if n < start:
            continue
        verify_action_files(d)
        records[n] = json.loads((d / "record.json").read_text())
        rngs[n] = load_snapshot(d / "pre_state.pt")["rng"]   # 난수 상태만 보관하고 스냅샷은 버린다
    pre = load_snapshot(root / f"action_{start:06d}" / "pre_state.pt")
    return pre, records, rngs


def run_replay(adapter, pre, records, rngs, start, end_of_history, edits):
    """edits: {행동번호: 대체 action tensor}. 마지막 행동 후 높이맵(512x512)을 반환."""
    adapter.restore(pre)
    t0 = time.perf_counter()
    for n in range(start, end_of_history + 1):
        restore_rng(rngs[n])
        action = edits.get(n, torch.tensor(records[n]["action"]))
        adapter.execute(action, records[n]["action_index"])
    return adapter.observe().numpy(), time.perf_counter() - t0


def run_replay_subset(adapter, pre, records, rngs, numbers, edits):
    """numbers: 실행할 행동 번호들(시간 순서). 그 사이의 다른 행동은 건너뛴다(영역과 무관하다고 보고).
    첫 행동 직전 상태에서 시작해 기록된 난수 상태를 행동마다 복원하며 번호 순서대로 실행한다. 마지막 행동 후 높이맵을 반환."""
    adapter.restore(pre)
    t0 = time.perf_counter()
    for n in sorted(numbers):
        restore_rng(rngs[n])
        action = edits.get(n, torch.tensor(records[n]["action"]))
        adapter.execute(action, records[n]["action_index"])
    return adapter.observe().numpy(), time.perf_counter() - t0


def metrics(h, h_ctrl, h_edit, h_final, roi):
    d_edit = h - h_edit
    out = ~roi
    return dict(
        roi_mae_vs_edit_mm=float(np.abs(d_edit[roi]).mean()) * 1000,
        global_mae_vs_edit_mm=float(np.abs(d_edit).mean()) * 1000,
        outside_mae_vs_ctrl_mm=float(np.abs((h - h_ctrl)[out]).mean()) * 1000,
        outside_max_vs_ctrl_mm=float(np.abs((h - h_ctrl)[out]).max()) * 1000,
        roi_mae_vs_ctrl_mm=float(np.abs((h - h_ctrl)[roi]).mean()) * 1000,
        max_abs_vs_recorded_final_mm=float(np.abs(h - h_final).max()) * 1000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", required=True, type=Path)
    p.add_argument("--beam-json", type=Path, help="first_beam.json (--segment를 쓰면 생략)")
    p.add_argument("--h-edit", type=Path, help="목표 높이맵 (--segment로 H_edit를 만드는 용도면 생략 가능)")
    p.add_argument("--segment", help="S,E : First-Beam 없이 바꿀 행동 구간을 직접 지정(1부터, 양끝 포함)")
    p.add_argument("--beam-rank", type=int, default=1, help="first_beam.json 의 rank (1이 최상위)")
    p.add_argument("--factors", default="0.7,1.3")
    p.add_argument("--mode", choices=["both", "length", "depth"], default="both")
    p.add_argument("--max-drift-mm", type=float, default=0.5)
    p.add_argument("--replicates", type=int, default=0, help="control을 추가로 재생해 ROI 재생 잡음을 측정하는 횟수")
    p.add_argument("--min-snr", type=float, default=5.0, help="효과/잡음이 이 값 이상이어야 valid")
    p.add_argument("--max-bad-pixels", type=int, default=5, help="control 통과: max-drift-mm 초과 픽셀 수 상한")
    p.add_argument("--max-p999-mm", type=float, default=0.2, help="control 통과: 99.9분위수 상한")
    p.add_argument("--roi", type=Path, help="기본값: history-dir/roi_mask_512.npy")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    root = args.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["static_files"].items():
        if Path(name).name != name or sha256(root / name) != expected:
            raise RuntimeError(f"고정 목표/마스크 파일이 변경됐습니다: {name}")
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if args.viewer else "0"
    import vs_simulator as sim
    try:
        from vs_heightmap_fast import apply_fast_height
        apply_fast_height(sim)      # 입자->높이맵 변환의 빠른 버전(결과 동일, 파일은 안 바꿈). VS_FAST_HEIGHT=0 이면 끔
    except Exception as _e_fh:
        print(f"[빠른 높이맵 적용 실패: {_e_fh!r}] 원본 함수를 쓴다", flush=True)
    import vs_config as cfg
    adapter = GenesisStateAdapter(sim)
    current = physics_signature(sim)
    if current != manifest["signature"]:
        changed = [k for k, v in current["details"].items() if v != details.get(k)]
        raise RuntimeError(f"저장 당시와 장면/코드/환경이 다릅니다: {changed}")

    if args.segment:
        start, end = (int(v) for v in args.segment.split(","))
        last = len([d for d in root.glob("action_??????") if d.is_dir()])
        if not 1 <= start <= end <= last:
            raise ValueError(f"--segment는 1 <= S <= E <= {last} 이어야 합니다.")
        beam = dict(rank=0, start_action=start, end_action=end, surrogate_best_factor=None, surrogate_roi_gain_mm=None)
    else:
        if not args.beam_json:
            raise ValueError("--beam-json 또는 --segment가 필요합니다.")
        beam_doc = json.loads(args.beam_json.expanduser().read_text())
        beam = next(b for b in beam_doc["beams"] if b["rank"] == args.beam_rank)
        start, end = beam["start_action"], beam["end_action"]
        last = beam_doc["n_actions"]
    factors = [float(x) for x in args.factors.split(",")]
    length_range = (cfg.LENGTH_MIN, cfg.LENGTH_MAX)
    depth_range = (cfg.DEPTH_MIN, cfg.DEPTH_MAX)

    roi = np.load(args.roi.expanduser() if args.roi else root / "roi_mask_512.npy").astype(bool)
    h_edit = None if args.h_edit is None else np.load(args.h_edit.expanduser(), allow_pickle=False).astype(np.float32)
    h_final = np.load(root / f"action_{last:06d}" / "transition.npz")["after_m"].astype(np.float32)
    if h_edit is None:      # --segment로 H_edit 자체를 만들 때: 비교 지표는 의미 없으니 기록된 최종 상태로 대체
        h_edit = h_final

    pre, records, rngs = load_plan(root, start)
    present = adapter.capture()
    report = dict(schema=1, beam=beam, start=start, end=end, last=last, mode=args.mode,
                  max_drift_mm=args.max_drift_mm, runs=[])
    try:
        h_ctrl, sec = run_replay(adapter, pre, records, rngs, start, last, {})
        dd = np.abs(h_ctrl - h_final) * 1000
        drift = float(dd.max())
        bad_px, p999 = int((dd > args.max_drift_mm).sum()), float(np.quantile(dd, 0.999))
        ctrl = dict(name="control(원래 행동)", factor=1.0, seconds=sec, replay_drift_max_mm=drift,
                    **metrics(h_ctrl, h_ctrl, h_edit, h_final, roi))
        report["control"] = ctrl
        report["control_passed"] = bad_px <= args.max_bad_pixels and p999 <= args.max_p999_mm
        report["control_legacy_passed"] = drift <= args.max_drift_mm
        ctrl.update(replay_bad_pixels=bad_px, replay_p999_mm=p999)
        print(f"[control] 행동 {start}~{last} 재생: 기록 대비 최대 drift={drift:.4f} mm "
              f"({'통과' if report['control_passed'] else '기준 초과 - 이 구간 비교는 신뢰하지 마세요'}), "
              f"ROI MAE vs H_edit={ctrl['roi_mae_vs_edit_mm']:.4f} mm")
        results = {1.0: h_ctrl}
        # 같은 상태에서 control을 다시 재생해 '재생 잡음'을 ROI 기준으로 잰다 (효과의 신뢰도 판정용).
        noise = None
        if args.replicates > 0:
            diffs = []
            for i in range(args.replicates):
                h_rep, sec_r = run_replay(adapter, pre, records, rngs, start, last, {})
                results[f"rep{i + 1}"] = h_rep
                diffs.append(np.abs(h_rep - h_ctrl) * 1000)
                print(f"[replicate {i + 1}] control 재생 {sec_r:.0f}초, ROI 잡음 MAE={diffs[-1][roi].mean():.5f} mm, "
                      f"전체 MAE={diffs[-1].mean():.5f} mm, 최대={diffs[-1].max():.3f} mm")
            noise = dict(n=args.replicates, roi_mae_mm=float(np.mean([d[roi].mean() for d in diffs])),
                         global_mae_mm=float(np.mean([d.mean() for d in diffs])),
                         max_mm=float(max(d.max() for d in diffs)))
        report["noise"] = noise
        report["min_snr"] = args.min_snr
        for f in factors:
            edits = {n: scaled(records[n]["action"], f, args.mode, length_range, depth_range)
                     for n in range(start, end + 1)}
            h_cf, sec = run_replay(adapter, pre, records, rngs, start, last, edits)
            m = metrics(h_cf, h_ctrl, h_edit, h_final, roi)
            m.update(name=f"cf x{f:g} ({args.mode})", factor=f, seconds=sec,
                     roi_gain_vs_ctrl_mm=ctrl["roi_mae_vs_edit_mm"] - m["roi_mae_vs_edit_mm"],
                     global_gain_vs_ctrl_mm=ctrl["global_mae_vs_edit_mm"] - m["global_mae_vs_edit_mm"])
            if noise is not None:
                m["snr"] = m["roi_mae_vs_ctrl_mm"] / max(noise["roi_mae_mm"], 1e-6)
                m["valid"] = bool(m["snr"] >= args.min_snr)
            report["runs"].append(m)
            results[f] = h_cf
            if noise is not None:
                print(f"[cf x{f:g}] 효과(ROI MAE vs control)={m['roi_mae_vs_ctrl_mm']:.4f} mm, 잡음={noise['roi_mae_mm']:.5f} mm, "
                      f"SNR={m['snr']:.1f} -> {'valid' if m['valid'] else 'INVALID(abstain)'}")
            print(f"[cf x{f:g}] ROI MAE vs H_edit={m['roi_mae_vs_edit_mm']:.4f} mm "
                  f"(개선 {m['roi_gain_vs_ctrl_mm']:+.4f}), ROI 밖 최대 변화={m['outside_max_vs_ctrl_mm']:.4f} mm, "
                  f"전체 개선 {m['global_gain_vs_ctrl_mm']:+.4f} mm")
        best = max(report["runs"], key=lambda r: r["roi_gain_vs_ctrl_mm"]) if report["runs"] else None
        report["best"] = None if best is None else best["name"]
        out = (args.out or root / (f"cf_replay_seg{start}_{end}" if args.segment else f"cf_replay_beam{args.beam_rank}")).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "heightmaps.npz", h_final_recorded=h_final, h_edit=h_edit,
                            **{(f"h_x{k:g}" if isinstance(k, float) else f"h_{k}"): v for k, v in results.items()})
        report["surrogate_prediction"] = None if args.segment else dict(best_factor=beam["surrogate_best_factor"],
                                              roi_gain_mm=beam["surrogate_roi_gain_mm"])
        report["passed"] = bool(report["control_passed"])
    finally:
        # 검증 후 원래 물리 상태로 복원
        adapter.restore(present)
    json_write(out / "cf_report.json", report)
    print(f"저장: {out}/cf_report.json, heightmaps.npz")
    if not args.segment:
        print(f"대리 모델 예측(참고): 배율 {beam['surrogate_best_factor']}, ROI 개선 {beam['surrogate_roi_gain_mm']:+.4f} mm")


if __name__ == "__main__":
    main()
