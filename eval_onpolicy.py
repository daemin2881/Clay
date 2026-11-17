"""체크포인트들을 '학습에 쓰지 않은' 이력의 1-step 전이로 비교한다.

각 전이 (before, action, after)마다 모델이 예측한 after를 실제 after와 비교한다.
  gap_mae_mm        : mean|pred - after|              (모델 정확도)
  corr_dMAE         : 예측 ΔMAE(vs 목표)와 실제 ΔMAE의 상관 (플래너가 의존하는 신호)
  bias_dMAE_mm      : mean(실제 ΔMAE - 예측 ΔMAE)     (0보다 크면 모델이 낙관적)
  frac_optimistic   : 예측은 개선인데 실제는 악화인 비율

사용: python -I eval_onpolicy.py --history HIST [--history HIST2] --ckpt A.pt --ckpt B.pt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from vs_dynamics import Param2Deform
from vs_planner import predict_next_batch


def load_transitions(root: Path):
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"action_\d{6}", p.name))
    tgt = np.load(root / "target_used_512_m.npy").astype(np.float32)
    B, A, F = [], [], []
    for d in dirs:
        z = np.load(d / "transition.npz")
        rec = json.loads((d / "record.json").read_text())
        B.append(z["before_m"].astype(np.float32))
        F.append(z["after_m"].astype(np.float32))
        A.append(np.asarray(rec["action"], np.float32))
    return np.stack(B), np.stack(A), np.stack(F), tgt


@torch.no_grad()
def predict(model, before, actions, device, bs=8):
    out = []
    for i in range(0, len(actions), bs):
        for j in range(i, min(i + bs, len(actions))):
            s = torch.from_numpy(before[j])[None, None].to(device)
            a = torch.from_numpy(actions[j])[None].to(device)
            p = predict_next_batch(model, s, a)
            out.append(p[0, 0].float().cpu().numpy())
    return np.stack(out)


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Param2Deform()
    m.load_state_dict(ck["model"])
    m = m.to(device=device, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def score(pred, before, after, tgt):
    mae = lambda h: np.abs(h - tgt).mean(axis=(1, 2)) * 1000
    dp, da = mae(pred) - mae(before), mae(after) - mae(before)
    gap = np.abs(pred - after).mean(axis=(1, 2)) * 1000
    return dict(n=len(pred), gap_mae_mm=float(gap.mean()),
                corr_dMAE=float(np.corrcoef(dp, da)[0, 1]),
                bias_dMAE_mm=float((da - dp).mean()),
                frac_optimistic=float(((dp < 0) & (da > 0)).mean()),
                pred_dMAE_mm=float(dp.mean()), act_dMAE_mm=float(da.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="append", required=True, type=Path)
    ap.add_argument("--ckpt", action="append", required=True, type=Path)
    ap.add_argument("--max", type=int, help="이력마다 앞에서 N개만")
    a = ap.parse_args()
    dev = torch.device("cuda")
    models = {c.stem: load_model(c, dev) for c in a.ckpt}
    for h in a.history:
        B, A, F, tgt = load_transitions(h.expanduser())
        if a.max:
            B, A, F = B[:a.max], A[:a.max], F[:a.max]
        print(f"\n=== {h.parent.name} ({len(A)}개 전이)")
        for name, m in models.items():
            r = score(predict(m, B, A, dev), B, F, tgt)
            print(f"  {name[-34:]:>34} | gap {r['gap_mae_mm']:.4f}mm | corr {r['corr_dMAE']:+.3f} | "
                  f"bias {r['bias_dMAE_mm']:+.4f}mm | 낙관 {r['frac_optimistic']:.0%} | "
                  f"예측ΔMAE {r['pred_dMAE_mm']:+.4f} 실제 {r['act_dMAE_mm']:+.4f}")


if __name__ == "__main__":
    main()
