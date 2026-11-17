"""ActionHistory 이력(action_XXXXXX/transition.npz)을 train_dynamics.py용 chunk 데이터셋으로 바꾼다.

- 기존 데이터셋은 심볼릭 링크로만 포함한다(원본 불변).
- 각 이력은 연속 `--rollout-len`개 행동씩 하나의 rollout으로 묶어 rollout 단위 train/val 분할에 맞춘다.
- 새 폴더가 이미 있으면 중단한다.

사용 예
  python history_to_chunks.py --out datasets/onpolicy_A_mix \
      --base datasets/medium_corrected_press700 \
      --history results/A_history200_xxx/history --history results/exp_A_len8_xxx/history
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

ROLLOUT_ID_OFFSET = 1000   # 기존 데이터의 rollout_id(0~)와 겹치지 않게


def load_history(root: Path):
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"action_\d{6}", p.name))
    if not dirs:
        raise FileNotFoundError(f"완료된 action_XXXXXX 폴더가 없습니다: {root}")
    before, after, actions = [], [], []
    for expected, d in enumerate(dirs, start=1):
        if int(d.name.split("_")[1]) != expected:
            raise RuntimeError(f"행동 번호가 연속적이지 않습니다: {d}")
        z = np.load(d / "transition.npz")
        rec = json.loads((d / "record.json").read_text())
        before.append(z["before_m"].astype(np.float32))
        after.append(z["after_m"].astype(np.float32))
        actions.append(np.asarray(rec["action"], dtype=np.float32))
    for k in range(len(dirs) - 1):   # 이력은 S_k+1 == 다음 행동의 S 여야 한다
        if not np.allclose(after[k], before[k + 1], atol=1e-6, rtol=0):
            raise RuntimeError(f"{root}: 행동 {k+1}의 결과와 {k+2}의 시작 상태가 다릅니다.")
    return np.stack(before), np.stack(after), np.stack(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base", type=Path, help="포함할 기존 chunk 데이터셋(심볼릭 링크)")
    ap.add_argument("--history", action="append", required=True, type=Path)
    ap.add_argument("--rollout-len", type=int, default=10)
    ap.add_argument("--first", action="append", default=[],
                    help="NAME=N 형식이 아니라 --history 순서대로 쓸 최대 행동 수(생략하면 전부)", type=int)
    a = ap.parse_args()
    out = a.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"새 폴더를 지정하세요: {out}")
    out.mkdir(parents=True)
    n_base = 0
    if a.base:
        for f in sorted(a.base.expanduser().resolve().glob("chunk_*.pt")):
            (out / f.name).symlink_to(f)
            n_base += 1
    rid, chunk_no, total = ROLLOUT_ID_OFFSET, 10000, 0
    for i, h in enumerate(a.history):
        b, af, ac = load_history(h.expanduser().resolve())
        if i < len(a.first):
            b, af, ac = b[:a.first[i]], af[:a.first[i]], ac[:a.first[i]]
        for s in range(0, len(ac), a.rollout_len):
            e = min(s + a.rollout_len, len(ac))
            if e - s < 2:     # 너무 짧은 꼬리는 버린다
                continue
            n = e - s
            torch.save(dict(
                state=torch.from_numpy(b[s:e]), action=torch.from_numpy(ac[s:e]),
                next_state=torch.from_numpy(af[s:e]),
                rollout_id=torch.full((n,), rid, dtype=torch.long),
                rollout_step=torch.arange(n, dtype=torch.long),
                rollout_length=n, action_type="visual_sculpting_linear_push", action_dim=5,
                action_definition=["x", "y", "theta", "length", "depth"],
                action_sampler="onpolicy_planner_history", material="medium",
                source=str(h)), out / f"chunk_{chunk_no:05d}.pt")
            rid += 1
            chunk_no += 1
            total += n
        print(f"[{h.name if h.name != 'history' else h.parent.name}] 사용 행동 {len(ac) if i >= len(a.first) else a.first[i]}개")
    print(f"저장: {out}  (기존 링크 {n_base}개, 새 전이 {total}개, 새 chunk {chunk_no - 10000}개)")


if __name__ == "__main__":
    main()
