#!/usr/bin/env python3
"""이력 확장: 이력 끝에 보상 행동(하나 또는 여러 개)을 Genesis 에서 실제로 실행해 K+1, K+2, ... 번째 행동으로 붙인 새 이력 폴더를 만든다.

왜: cf_compensate 의 모든 반사실(최근 재생, 먼 과거 트리, 개입 묶음)은 '이력 폴더'를 과거로 본다. 확정한 보상 행동을 이력에 붙이면 다음 라운드는
    '원본 K개 + 확정한 행동'을 과거로 하는 새 이력에서 같은 방법을 그대로 다시 돌릴 수 있다. 그러면 방금 한 보상 행동도 반사실의 대상이 된다
    (그때 달리 눌렀다면?). 즉 '계속 과거를 타고 들어가며' 보상하는 구조를 cf_compensate 를 고치지 않고 얻는다.

원본 이력은 읽기만 한다: 새 폴더는 원본의 모든 항목(행동 폴더, 목표 파일, manifest)을 심볼릭 링크로 가리키고, 새 행동 폴더만 실제 파일로 만든다.
새 행동 폴더는 vs_history.ActionHistory.record_push 가 만드는 것과 같은 6개 파일 + 체크섬이다(counterfactual_replay.load_plan 이 그대로 읽는다).
  - pre_state.pt  행동 직전 전체 상태(MPM·로봇·제어기·난수).  post_state.pt  직후 상태.  transition.npz  before/after 높이맵.  record.json  행동.
  - planner_context.pt 는 비어 있다(계획기가 아니라 보상 탐색이 고른 행동이라서). record.json 에 synthetic=true 를 적어 둔다.
  - 행동은 float32 로 바꿔 실행한다(원본 이력의 행동이 float32 라서). action_index 는 번호-1 (로그 출력에만 쓰인다).

여러 자식을 한 번에 만들 수 있다(같은 원본에서 갈라지는 빔 후보들). Genesis 는 한 번만 초기화한다.

사용: python extend_history.py --src ROOT --spec spec.json
  spec.json = [{"out": "새 이력 폴더", "actions": [[x,y,theta,L,d], ...], "expect_npy": "(선택) 실행 결과와 맞춰 볼 높이맵"}, ...]
  각 out 폴더에 extend.json(복원 오차, 재현 오차)과 ext_final_m.npy(마지막 높이맵, float32)가 생긴다.
GPU/Genesis 가 필요하다(워커 하나 분량, 약 3.5 GB).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

_ACTION_DIR = re.compile(r"^action_(\d{6})$")
_OWN_FILES = {"extend.json", "ext_final_m.npy"}


def drift(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) * 1000.0
    return dict(mae_mm=float(d.mean()), p99_9_mm=float(np.quantile(d, 0.999)), max_mm=float(d.max()), px_over_0_5mm=int((d > 0.5).sum()))


def n_actions(root):
    return len([d for d in Path(root).iterdir() if d.is_dir() and _ACTION_DIR.match(d.name)])


def link_tree(src, out):
    """src 의 모든 항목(미완성 행동 폴더 제외)을 out 에 심볼릭 링크로 만든다. out 은 비어 있어야 한다."""
    src, out = Path(src).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"이미 내용이 있는 폴더입니다: {out}")
    for p in sorted(src.iterdir()):
        if p.name.startswith("action_") and not _ACTION_DIR.match(p.name):
            continue
        if p.name in _OWN_FILES:        # 부모가 남긴 결과 파일은 링크하지 않는다(링크하면 자식이 쓸 때 부모 파일을 덮어쓴다)
            continue
        (out / p.name).symlink_to(Path(os.path.realpath(p)))


def write_action_dir(out, number, pre, post, action, seconds, schema, tensor_save, json_write, sha256):
    """record_push 와 같은 6개 파일 + checksums.json 을 쓴다. 미완성 폴더에 다 쓰고 마지막에 이름을 바꾼다."""
    prefix = f"action_{number:06d}"
    pending = Path(out) / (prefix + ".incomplete")
    pending.mkdir()
    tensor_save(pending / "pre_state.pt", pre)
    tensor_save(pending / "planner_context.pt", {})
    json_write(pending / "pending_action.json", dict(t=number - 1, action=action, action_index=number - 1))
    tensor_save(pending / "post_state.pt", post)
    before, after = pre["height"].numpy(), post["height"].numpy()
    np.savez_compressed(pending / "transition.npz", action=np.asarray(action, dtype=np.float32), before_m=before, after_m=after, delta_m=after - before,
                        predicted_after_m=after, predicted_delta_m=after - before)
    json_write(pending / "record.json", dict(schema=schema, t=number - 1, action_number=number, action_index=number - 1, action=action, execution_seconds=seconds,
                                             metrics=dict(deformation_mae_mm=float(np.abs(after - before).mean()) * 1000.0), synthetic=True, source="extend_history.py"))
    json_write(pending / "checksums.json", {p.name: sha256(p) for p in pending.iterdir() if p.is_file()})
    pending.rename(Path(out) / prefix)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="확장할 이력 폴더(행동 K개). 원본의 일부만 쓰려면 cf_common.make_view 로 만든 폴더를 준다")
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--viewer", action="store_true", help="Genesis 창을 띄워 확정한 푸시를 새 프로세스에서 실행하는 모습을 보여 준다(DISPLAY 필요)")
    a = ap.parse_args()
    src = a.src.resolve()
    spec = json.loads(a.spec.read_text())
    K = n_actions(src)
    man = json.loads((src / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if a.viewer else "0"
    import torch
    import vs_simulator as sim
    try:
        from vs_heightmap_fast import apply_fast_height
        apply_fast_height(sim)      # 입자->높이맵 변환의 빠른 버전(결과 동일, 파일은 안 바꿈). VS_FAST_HEIGHT=0 이면 끔
    except Exception as _e_fh:
        print(f"[빠른 높이맵 적용 실패: {_e_fh!r}] 원본 함수를 쓴다", flush=True)
    from vs_history import GenesisStateAdapter, SCHEMA, json_write, load_snapshot, physics_signature, sha256, tensor_save
    if physics_signature(sim) != man["signature"]:
        raise SystemExit("physics signature 불일치: 이력을 만든 코드/설정/버전과 다릅니다.")
    adapter = GenesisStateAdapter(sim)
    last_post = src / f"action_{K:06d}" / "post_state.pt"
    rec_after = np.load(src / f"action_{K:06d}" / "transition.npz")["after_m"]
    for item in spec:
        out = Path(item["out"])
        t0 = time.time()
        link_tree(src, out)
        adapter.restore(load_snapshot(last_post))
        restore_err = drift(adapter.observe().numpy(), rec_after)
        info = dict(src=str(src), n_before=K, restore_error_vs_record=restore_err, actions=[])
        h = None
        for i, act in enumerate(item["actions"], 1):
            number = K + i
            a32 = torch.tensor(act, dtype=torch.float32)
            action = a32.tolist()
            pre = adapter.capture()
            t_a = time.perf_counter()
            adapter.execute(a32, number - 1)
            sec = time.perf_counter() - t_a
            post = adapter.capture()
            write_action_dir(out, number, pre, post, action, sec, SCHEMA, tensor_save, json_write, sha256)
            h = post["height"].numpy()
            info["actions"].append(dict(number=number, action=action, seconds=sec))
        info["n_after"] = K + len(item["actions"])
        np.save(out / "ext_final_m.npy", h.astype(np.float32))      # 링크가 아닌 실제 파일(이름이 action_ 로 시작하지 않아 이력 항목과 섞이지 않는다)
        if item.get("expect_npy"):
            info["repro_vs_expect"] = drift(h, np.load(item["expect_npy"]))
        info["seconds"] = time.time() - t0
        (out / "extend.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        print(f"[확장] {out.name}: 행동 {K} -> {info['n_after']} | 복원 오차 MAE {restore_err['mae_mm']:.5f}mm" + (f" | 기대 결과 대비 MAE {info['repro_vs_expect']['mae_mm']:.5f}mm" if "repro_vs_expect" in info else "") +
              f" | {info['seconds']:.0f}초", flush=True)


if __name__ == "__main__":
    main()
