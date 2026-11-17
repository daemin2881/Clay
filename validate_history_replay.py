"""저장 상태를 새 프로세스에서 복원하고 원래 행동을 재실행한다."""
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
                        restore_rng, sha256, verify_action_files, tensor_save)


def height_drift(actual, expected, roi=None):
    d = (torch.as_tensor(actual).cpu().float() - torch.as_tensor(expected).cpu().float()).abs()
    if not bool(torch.isfinite(d).all()):
        raise RuntimeError("재현 높이맵에 NaN 또는 Inf가 있습니다.")
    out = dict(mae_mm=float(d.mean()) * 1000, max_mm=float(d.max()) * 1000)
    if roi is not None:
        r = torch.as_tensor(roi, dtype=torch.bool)
        out.update(roi_mae_mm=float(d[r].mean()) * 1000, roi_max_mm=float(d[r].max()) * 1000)
    return out


def physics_drift(actual, expected):
    out = {}
    if len(actual["physics"]) != len(expected["physics"]):
        raise RuntimeError("solver 수가 달라졌습니다.")
    for i, (a, b) in enumerate(zip(actual["physics"], expected["physics"])):
        if a is None or b is None:
            if a is not b:
                raise RuntimeError("활성 solver 구성이 다릅니다.")
            continue
        for name, value in b["tensors"].items():
            d = (a["tensors"][name].to(torch.float64) - value.to(torch.float64)).abs()
            if not bool(torch.isfinite(d).all()):
                raise RuntimeError(f"재현 물리 상태에 유효하지 않은 값이 있습니다: {name}")
            out[f"{i}:{b['type']}.{name}"] = float(d.max()) if d.numel() else 0.0
    for name in ("q", "velocity", "tip_pos", "tip_quat"):
        d = (actual["robot"][name].double() - expected["robot"][name].double()).abs()
        if not bool(torch.isfinite(d).all()):
            raise RuntimeError(f"재현 로봇 상태에 유효하지 않은 값이 있습니다: {name}")
        out[f"robot.{name}"] = float(d.max())
    return out


def spatial_diagnostics(actual, expected, threshold_mm, roi=None):
    d = (torch.as_tensor(actual).cpu().float()-torch.as_tensor(expected).cpu().float()).abs()*1000
    if d.ndim != 2 or not bool(torch.isfinite(d).all()):
        raise RuntimeError("공간 진단에는 유효한 2차원 높이맵이 필요합니다.")
    row, col = (int(x) for x in torch.unravel_index(d.argmax(), d.shape))
    count = int((d > threshold_mm).sum())
    q = torch.quantile(d.flatten(), torch.tensor([.99, .999, .9999]))
    return dict(max_row=row, max_col=col, above_threshold_pixels=count,
                above_threshold_fraction=count/d.numel(),
                p99_mm=float(q[0]), p999_mm=float(q[1]), p9999_mm=float(q[2]),
                max_is_in_roi=None if roi is None else bool(roi[row, col]))


def validate_replays(adapter, root, *, starts, sequence_length=1, max_drift_mm=0.5,
                     max_bad_pixels=5, max_p999_mm=0.2, gate="robust"):
    root = Path(root)
    if sequence_length < 1 or not math.isfinite(max_drift_mm) or max_drift_mm <= 0:
        raise ValueError("재생 길이와 drift 기준은 양수여야 합니다.")
    available = sorted(p for p in root.glob("action_??????") if p.is_dir())
    numbers = {int(p.name.split("_")[1]): p for p in available}
    if not starts or any(n not in numbers for n in starts):
        raise ValueError("검증할 시작 행동 번호가 없거나 저장 이력에 없습니다.")
    if any(any(n+k not in numbers for k in range(sequence_length)) for n in starts):
        raise ValueError("요청한 길이의 연속 행동 이력이 없습니다.")
    roi_file = root / "roi_mask_512.npy"
    roi = np.load(roi_file, allow_pickle=False) if roi_file.exists() else None
    report = dict(schema=1, passed=False, max_allowed_drift_mm=max_drift_mm,
                  gate=gate, max_bad_pixels=max_bad_pixels, max_p999_mm=max_p999_mm,
                  gate_note="robust: max_drift_mm 초과 픽셀 수<=max_bad_pixels 그리고 p99.9<=max_p999_mm. legacy: 픽셀 최대값<=max_drift_mm",
                  sequence_length=sequence_length, starts=starts,
                  rng_policy="각 행동 직전의 기록된 외생 난수 상태 사용", trials=[])
    present = adapter.capture()
    started = time.perf_counter()
    try:
        for start in starts:
            folder = numbers[start]
            verify_action_files(folder)
            pre = load_snapshot(folder / "pre_state.pt")
            restored = adapter.restore(pre)
            restore_drift = height_drift(restored, pre["height"], roi)
            restored_snapshot = adapter.capture()
            restore_fields = physics_drift(restored_snapshot, pre)
            # 복원 자체는 저장 텐서를 그대로 써야 한다. 0.5 mm 행동 drift와 구별한다.
            restore_ok = all(math.isfinite(v) and v <= 1e-6 for v in restore_fields.values())
            trial = dict(start=start, restore=restore_drift, restore_fields_max_abs=restore_fields,
                         restore_passed=restore_ok, frames=[])
            report["trials"].append(trial)
            if not restore_ok or restore_drift["max_mm"] > max_drift_mm:
                raise RuntimeError(f"행동 {start}의 저장 상태 자체를 정확히 복원하지 못했습니다.")
            for number in range(start, start+sequence_length):
                folder = numbers[number]
                verify_action_files(folder)
                record = json.loads((folder / "record.json").read_text())
                if record["action_number"] != number or record["t"] != number-1 or record["action_index"] != number-1:
                    raise RuntimeError(f"행동 번호 불일치: {folder}")
                # 계획 단계가 소비했던 난수도 동일하게 맞춘다. 물리/제어기 상태는 덮어쓰지 않는다.
                step_pre = pre if number == start else load_snapshot(folder / "pre_state.pt")
                restore_rng(step_pre["rng"])
                adapter.execute(torch.tensor(record["action"]), record["action_index"])
                actual = adapter.capture()
                expected = load_snapshot(folder / "post_state.pt")
                metrics = height_drift(actual["height"], expected["height"], roi)
                fields = physics_drift(actual, expected)
                finite = all(math.isfinite(v) for v in list(metrics.values()) + list(fields.values()))
                spatial = spatial_diagnostics(actual["height"], expected["height"], max_drift_mm, roi)
                legacy_ok = finite and metrics["max_mm"] <= max_drift_mm
                robust_ok = (finite and spatial["above_threshold_pixels"] <= max_bad_pixels
                             and spatial["p999_mm"] <= max_p999_mm)
                ok = robust_ok if gate == "robust" else legacy_ok
                trial["frames"].append(dict(action_number=number, passed=ok, legacy_passed=legacy_ok, height=metrics,
                    fields_max_abs=fields,
                    spatial=spatial))
                print(f"[REPLAY {number:06d}] MAE={metrics['mae_mm']:.6f}mm "
                      f"MAX={metrics['max_mm']:.6f}mm 초과픽셀={spatial['above_threshold_pixels']} "
                      f"p99.9={spatial['p999_mm']:.4f}mm 통과={ok} (기존기준 {legacy_ok})")
                if not ok:
                    stem = f"replay_failure_len{sequence_length}_start{start:06d}_action{number:06d}"
                    tensor_save(root / (stem+"_actual_state.pt"), actual)
                    np.savez_compressed(root / (stem+"_heightmaps.npz"),
                        actual_m=actual["height"].cpu().numpy(),
                        expected_m=expected["height"].cpu().numpy(),
                        delta_m=(actual["height"]-expected["height"]).cpu().numpy())
                    trial["frames"][-1]["debug_files"] = [stem+"_actual_state.pt", stem+"_heightmaps.npz"]
                    raise RuntimeError(f"행동 {number}의 replay drift가 기준을 초과했습니다.")
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            restored_present = adapter.restore(present)
            present_drift = height_drift(restored_present, present["height"])
            fields = physics_drift(adapter.capture(), present)
            report["present_restore"] = dict(height=present_drift, fields_max_abs=fields)
            if any(not math.isfinite(v) or v > 1e-6 for v in fields.values()):
                report["passed"] = False
                report["present_restore_error"] = "검증 후 원래 물리 상태 복원 오차가 큽니다."
        except Exception as exc:
            report["passed"] = False
            report["present_restore_error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--starts", default="1", help="1부터 시작하는 행동 번호. 예: 1,6,11 또는 all")
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--max-drift-mm", type=float, default=0.5)
    parser.add_argument("--max-bad-pixels", type=int, default=5, help="robust 기준: max-drift-mm를 넘는 픽셀 수 상한")
    parser.add_argument("--max-p999-mm", type=float, default=0.2, help="robust 기준: 99.9분위수 상한")
    parser.add_argument("--gate", choices=["robust", "legacy"], default="robust")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    root = args.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != 1:
        raise ValueError("지원하지 않는 이력 형식입니다.")
    for name, expected in manifest["static_files"].items():
        if Path(name).name != name or sha256(root / name) != expected:
            raise RuntimeError(f"고정 목표/마스크 파일이 변경됐습니다: {name}")
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if args.viewer else "0"
    import vs_simulator as sim
    adapter = GenesisStateAdapter(sim)
    current_signature = physics_signature(sim)
    if current_signature != manifest["signature"]:
        changed = [k for k, v in current_signature["details"].items() if v != details.get(k)]
        raise RuntimeError(f"저장 당시와 장면/코드/환경이 다릅니다: {changed}")
    dirs = sorted(p for p in root.glob("action_??????") if p.is_dir())
    numbers = [int(p.name.split("_")[1]) for p in dirs]
    starts = ([n for n in numbers if all(n+k in numbers for k in range(args.sequence_length))]
              if args.starts == "all" else [int(x) for x in args.starts.split(",")])
    report = validate_replays(adapter, root, starts=starts, sequence_length=args.sequence_length,
                              max_drift_mm=args.max_drift_mm, max_bad_pixels=args.max_bad_pixels,
                              max_p999_mm=args.max_p999_mm, gate=args.gate)
    report["signature"] = current_signature["digest"]
    report["manifest_sha256"] = sha256(root / "manifest.json")
    report["scope"] = "검증한 시작점과 재생 길이에 한정. 미검증 과거 후보는 별도 검증 필요."
    output = root / f"replay_validation_len{args.sequence_length}.json"
    json_write(output, report)
    print(f"검증 보고서: {output}")
    if not report["passed"]:
        raise SystemExit("재현 검증 실패. First-Beam 진행을 중단하세요.")


if __name__ == "__main__":
    main()
