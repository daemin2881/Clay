"""보상 중 Genesis 에서 실제로 실행해 본 (상태, 행동, 결과)로 Param2Deform 을 미세조정하고, 학습에 쓰지 않은 실행 결과로 효과를 잰다.

원본 체크포인트는 절대 덮어쓰지 않는다(새 --tag 로 새 파일을 만든다, 데이터셋 폴더도 새로 만든다).
데이터: cf_compensate.py --save-transitions 로 저장된 <run>/transitions/*.npz (행동, 변화량 delta, state_tag) 와 그 상태 <run>/cf/states/iter_<tag>.npy
  전이 = (before = 그 상태, 행동, after = before + delta).  기존 on-policy 데이터셋(--base)을 심볼릭 링크로 포함하고 새 전이를 --oversample 배 반복해 비중을 준다.
평가 (--holdout 실행의 전이, 학습에 안 씀): 모델이 후보 순위를 얼마나 맞히나
  gap   모델 예측 vs 실제 결과의 평균 절대 오차(mm, 행동 주변 창)
  rho   같은 상태의 후보들에서 모델이 예측한 높이맵으로 잰 J 와 실제 J 의 순위 상관(상태별 평균)
  sign  모델이 '현재보다 좋아진다'고 예측한 것이 실제로 그런 비율(정밀도)과 실제 개선 후보 중 모델이 그렇게 예측한 비율(재현율)

사용:
  python finetune_from_cf.py --history-dir H --regions DIR --runs results/case_A/cf_edit_A80_v5 [--runs ...] --holdout results/case_A/cf_edit_A80_v8 --tag cf_ft1 [--epochs 6 --lr 2e-5]
  python finetune_from_cf.py ... --eval-only checkpoints/A.pt checkpoints/B.pt      (학습 없이 체크포인트 비교)
  --dry-run  데이터셋만 만들고 학습은 하지 않는다
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import cf_common as cc
import cf_compensate as cfc
import cf_jacobian as jac
from region_set import RegionSet

ROOT = Path(__file__).resolve().parent


def load_run(run):
    """-> {tag: (before_map, [(행동, delta)])}"""
    run = Path(run)
    out = {}
    for f in sorted(glob.glob(str(run / "transitions" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        tag = int(z["state_tag"])
        sp = run / "cf" / "states" / f"iter_{tag:03d}.npy"
        if not sp.exists():
            continue
        if tag not in out:
            out[tag] = (np.load(sp).astype(np.float32), [])
        out[tag][1].append((np.asarray(z["action"], np.float32), z["delta"].astype(np.float32)))
    return out


def build_dataset(runs, out_dir, base, oversample, ref_chunk):
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"새 폴더를 지정하세요: {out_dir}")
    out_dir.mkdir(parents=True)
    nb = 0
    if base:
        for f in sorted(Path(base).resolve().glob("chunk_*.pt")):
            (out_dir / f.name).symlink_to(f)
            nb += 1
    B, A, F = [], [], []
    for r in runs:
        for tag, (before, items) in load_run(r).items():
            for act, d in items:
                B.append(before)
                A.append(act)
                F.append(before + d)
    if not A:
        raise RuntimeError("학습에 쓸 전이가 없습니다")
    ref = torch.load(ref_chunk, map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ref.items() if k not in ("state", "action", "next_state", "states", "actions", "next_states", "rollout_id", "rollout_step", "rollout_length")}
    meta.update(action_sampler="cf_compensate_candidates", source=",".join(str(r) for r in runs))
    rid, chunk_no, n_chunks = 2000, 20000, 0
    idx = np.arange(len(A))
    for rep in range(oversample):
        perm = np.random.RandomState(rep).permutation(idx)
        for s in range(0, len(perm), 10):
            sel = perm[s:s + 10]
            if len(sel) < 2:
                continue
            st, ac, nx = (torch.from_numpy(np.stack([x[i] for i in sel])) for x in (B, A, F))
            ch = dict(state=st, action=ac, next_state=nx, states=st, actions=ac, next_states=nx, rollout_id=torch.full((len(sel),), rid, dtype=torch.long),
                      rollout_step=torch.arange(len(sel), dtype=torch.long), rollout_length=len(sel), **meta)
            torch.save(ch, out_dir / f"chunk_{chunk_no:05d}.pt")
            rid += 1
            chunk_no += 1
            n_chunks += 1
    print(f"데이터셋 {out_dir}: 기존 chunk {nb}개(링크) + 새 전이 {len(A)}개 x {oversample}회 -> chunk {n_chunks}개")
    return out_dir


def load_model(path, device):
    from vs_dynamics import Param2Deform
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Param2Deform()
    m.load_state_dict(ck["model"])
    m = m.to(device=device, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def predict(model, before, actions, device, bs=12):
    from vs_planner import predict_next_batch
    cur = torch.as_tensor(before, dtype=torch.float32, device=device)[None, None]
    out = []
    for i in range(0, len(actions), bs):
        a = torch.tensor(np.asarray(actions[i:i + bs], dtype=np.float32), device=device)
        out.append(predict_next_batch(model, cur, a)[:, 0].double().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, hold_runs, cfg_for, device, tool_r_px, geo, eps=0.01, exclude=frozenset()):
    gaps, rhos, tp, fp, fn, n_imp, n_all = [], [], 0, 0, 0, 0, 0
    for run in hold_runs:
        for tag, (before, items) in load_run(run).items():
            items = [(a_, d_) for a_, d_ in items if cfc.akey(list(np.asarray(a_, float))) not in exclude]      # 학습에 쓴 행동과 같은 행동은 뺀다(암기 방지)
            if len(items) < 6:
                continue
            cfg = cfg_for(run)
            sf = lambda m: cfc.score(m, cfg)["J"]
            acts = [a for a, _ in items]
            pred = predict(model, before, acts, device)
            Jp = np.array([sf(p) for p in pred])
            Ja = np.array([sf(before.astype(np.float64) + d.astype(np.float64)) for _, d in items])
            J0 = sf(before.astype(np.float64))
            for (a, d), p in zip(items, pred):
                r = max(int(round(0.6 * tool_r_px)), 6)
                x0, y0 = (a[0] - geo['x0']) * geo['px_per_m_x'], (a[1] - geo['y0']) * geo['px_per_m_y']      # 이력 manifest 의 좌표 변환
                c0, r0 = int(np.clip(x0, r, 511 - r)), int(np.clip(y0, r, 511 - r))
                sl = (slice(r0 - 3 * r, r0 + 3 * r), slice(c0 - 3 * r, c0 + 3 * r))
                gaps.append(float(np.abs(p[sl] - (before + d)[sl]).mean() * 1000))
            if np.std(Jp) > 0 and np.std(Ja) > 0:
                rhos.append(stats.spearmanr(Jp, Ja)[0])
            pi, ai = Jp < J0 - eps, Ja < J0 - eps
            tp += int((pi & ai).sum()); fp += int((pi & ~ai).sum()); fn += int((~pi & ai).sum())
            n_imp += int(ai.sum()); n_all += len(Ja)
    return dict(n=n_all, gap_mm=float(np.mean(gaps)) if gaps else float("nan"), rho=float(np.mean(rhos)) if rhos else float("nan"), n_states=len(rhos),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), pred_imp=tp + fp, real_imp=n_imp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--runs", action="append", default=[], type=Path, help="학습에 쓸 cf_edit 출력 폴더들")
    ap.add_argument("--holdout", action="append", default=[], type=Path, help="평가에만 쓸 cf_edit 출력 폴더들(학습 제외)")
    ap.add_argument("--base", type=Path, default=Path("datasets/onpolicy_A_mix_20261114"))
    ap.add_argument("--init-checkpoint", type=Path, default=Path("checkpoints/param2deform_medium_paper_planner_range_deep150.pt"))
    ap.add_argument("--tag", default=None, help="새 체크포인트 이름 태그. 이미 있으면 중단")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--exclude-runs", action="append", default=[], type=Path, help="이 실행들의 전이와 같은 행동은 평가에서 뺀다(학습에 쓴 실행을 주면 암기를 걸러낸다). --runs 를 주면 자동으로 포함")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-only", nargs="+", type=Path, default=None)
    a = ap.parse_args()
    root = a.history_dir.resolve()
    geom = cc.geometry(root)
    tool_r = geom["tool_r_px"]
    cfc.BOUNDS = cc.action_bounds(2.0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    excl = set()
    for rr in list(a.exclude_runs) + list(a.runs):
        for _, (_, items_) in load_run(rr).items():
            excl |= {cfc.akey(list(np.asarray(x, float))) for x, _ in items_}
    if excl:
        print(f"(평가에서 학습/제외 실행과 같은 행동 {len(excl)}종은 뺀다)")

    def cfg_for(run):       # 평가용 J 의 기준(=시작 상태)은 그 실행의 처음 상태 iter_000
        if str(run) not in cache:
            h0 = np.load(Path(run) / "cf" / "states" / "iter_000.npy").astype(np.float64)
            rs = RegionSet.load(a.regions, ref_default=h0)
            cache[str(run)] = cfc.build_cfg(rs, h0, 1.0, int(round(0.94 * tool_r)), int(round(1.17 * tool_r)), tool_r)
        return cache[str(run)]

    def report(name, path):
        m = load_model(path, dev)
        r = evaluate(m, a.holdout, cfg_for, dev, tool_r, geom, exclude=excl)
        print(f"  {name:<34} 평가 전이 {r['n']}개({r['n_states']}개 상태) | 행동 주변 gap {r['gap_mm']:.4f}mm | 상태별 순위 상관 {r['rho']:+.2f} | "
              f"'개선' 예측 {r['pred_imp']}개 중 정답 {r['precision']:.0%}, 실제 개선 {r['real_imp']}개 중 포착 {r['recall']:.0%}")
        return r

    if a.eval_only:
        for c in a.eval_only:
            report(c.stem[-34:], c)
        return
    if not a.runs or not a.tag:
        raise SystemExit("--runs 와 --tag 가 필요합니다(평가만 하려면 --eval-only).")
    new_ck = ROOT / "checkpoints" / f"param2deform_medium_paper_{a.tag}.pt"
    if new_ck.exists():
        raise SystemExit(f"이미 있는 체크포인트입니다(덮어쓰지 않음): {new_ck}")
    ref_chunk = sorted(Path(a.base).glob("chunk_1*.pt"))[0] if sorted(Path(a.base).glob("chunk_1*.pt")) else sorted(Path(a.base).glob("chunk_*.pt"))[0]
    ds = build_dataset(a.runs, ROOT / "datasets" / f"cf_onpolicy_{a.tag}", a.base, a.oversample, ref_chunk)
    if a.dry_run:
        print("(--dry-run) 학습은 하지 않았습니다:", ds)
        return
    if a.holdout:
        print("미세조정 전 평가:")
        report("init " + a.init_checkpoint.stem[-26:], a.init_checkpoint)
    cmd = [sys.executable, "train_dynamics.py", "--material", "medium", "--dataset-dir", str(ds), "--init-checkpoint", str(a.init_checkpoint),
           "--learning-rate", str(a.lr), "--epochs", str(a.epochs), "--loss-mode", "paper", "--tag", a.tag]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise SystemExit(f"학습 실패(종료 코드 {rc})")
    if not new_ck.exists():
        cand = sorted((ROOT / "checkpoints").glob(f"*{a.tag}*.pt"))
        if not cand:
            raise SystemExit("새 체크포인트를 찾지 못했습니다")
        new_ck = cand[-1]
    if a.holdout:
        print("미세조정 후 평가:")
        report("fine-tuned " + a.tag, new_ck)
    print("새 체크포인트:", new_ck, "(원본은 그대로)")


if __name__ == "__main__":
    main()
