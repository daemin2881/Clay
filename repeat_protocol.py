"""보상 있음/없음 이어 조형 반복 프로토콜: 같은 조건을 R번(시드만 다르게) 돌려 분산까지 보고 비교한다. 대상(A, 하트, ...)에 의존하지 않는다.

왜: 한 번의 이어 조형은 계획기의 무작위 후보 추출과 GPU 비결정성 때문에 같은 팔의 두 번 사이 차이(0.32)가 보상 효과(0.27)와 비슷했다.
    효과가 반복 변동보다 큰지 보려면 같은 조건 반복이 필요하다.

팔
  nocomp : 이력 K 번째 상태에서 N 행동을 이어서 조형 (보상 없음)
  comp   : cf_compensate 가 확정한 상태(--cf-dir/states/iter_<확정 수>.pt)에서 N 행동을 이어서 조형. 확정 행동 수 C 만큼 총 행동이 더 많다.
  모든 실행은 continue_sculpt.py 이며 같은 계획기 설정·같은 목표를 쓴다. 반복마다 시드 1..R (두 팔이 같은 시드를 나눠 쓴다. 상태가 달라서 쌍을 이룬 실험은 아니다).

목표: 계획기와 평가 모두 '요청으로 만든 새 목표'를 향한다(A -> B 조형처럼 원래 목표는 기준이 아니다).
점수 (요청 기준이 판정, 원래 목표 기준은 참고만)
  영역별 (형태/기준형태 + 기울기/기준기울기 + 테두리/기준테두리)의 영역 평균 e. 낮을수록 좋음. 기준은 K 번째 상태(= 3.00).
  이어 조형으로 판이 가라앉아 목표를 상태마다 판 높이 변화만큼 이동해서 잰다(rounds_compare 와 같은 방식).
  비교는 K 번째 이후 '총 행동 수' 기준(comp 는 확정 행동 C 개 포함)이 기본이고, --align steps 면 이어 조형 행동 수 기준.
통계
  팔별 평균·표준편차, 차이(comp-nocomp), 차이/합동표준편차, 순열검정 p 값(정확 열거, 양측). 반복이 3 정도면 표준편차와 p 값이 매우 거칠다: 참고용이다.

사용
  python repeat_protocol.py run --history-dir H --upto K --cf-dir RUN/cf --regions DIR [--names a,b] --checkpoint CK --num-actions 20 --repeats 3 --out OUT [--dry-run]
  python repeat_protocol.py collect --history-dir H --upto K --regions DIR --commits C --arm nocomp=D1,D2 --arm comp=D3,D4 [--checkpoints 5,10,15,20]
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import cf_common as cc

ROOT = Path(__file__).resolve().parent


def run_cmds(a):
    hist = a.history_dir.resolve()
    cf = a.cf_dir.resolve()
    summ = json.loads((cf / "summary.json").read_text())
    commits = int(summ["commits"])
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from region_set import RegionSet
    rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_default=np.zeros((512, 512)))
    roi = np.zeros((512, 512), bool)
    for n in rs.names:
        roi |= rs.masks[n]
    np.save(out / "roi.npy", roi)
    K = a.upto or cc.n_actions(hist)
    # 계획기가 따라갈 목표 = 요청으로 만든 '새 목표'(A -> B 조형처럼 원래 목표가 아니라 바뀐 목표를 향해 이어 조형한다).
    # 기본: K 번째 상태에서 영역 안만 영역 목표로 바꾼 맵(영역 밖은 현재 그대로라 계획기가 건드릴 이유가 없다). 전체 새 목표 맵이 있으면 --target-npy.
    if a.target_npy is not None:
        tgt_new = np.load(a.target_npy).astype(np.float32)
    else:
        h_K = np.load(hist / f"action_{K:06d}" / "transition.npz")["after_m"].astype(np.float64)
        tgt_new = rs.h_star_all(h_K).astype(np.float32)
    np.save(out / "target_new.npy", tgt_new)
    base = [sys.executable, "continue_sculpt.py", "--history-dir", hist, "--restore-action", K, "--target", out / "target_new.npy",
            "--roi", out / "roi.npy", "--checkpoint", a.checkpoint, "--num-actions", a.num_actions, "--plan-horizon", a.plan_horizon, "--candidates", a.candidates,
            "--candidate-batch-size", 1, "--refine-actions", a.refine_actions, "--refine-steps", a.refine_steps, "--mpc-execute", a.mpc_execute]
    cmds, arms = [], {"nocomp": [], "comp": []}
    for r in range(1, a.repeats + 1):
        for arm in ("nocomp", "comp"):
            d = out / f"{arm}_s{r}"
            c = list(base) + ["--seed", a.seed0 + r - 1, "--out-dir", d]
            if arm == "comp":
                if commits <= 0:
                    raise SystemExit("보상에서 확정된 행동이 없어 comp 팔이 없습니다.")
                c += ["--start-state", cf / "states" / f"iter_{commits:03d}.pt"]
            cmds.append((arm, d, [str(x) for x in c]))
            arms[arm].append(d)
    return cmds, arms, commits


def main_run(a):
    cmds, arms, commits = run_cmds(a)
    print(f"확정 행동 {commits}개 | 팔 2개 x 반복 {a.repeats}회 = 실행 {len(cmds)}개, 각 {a.num_actions} 행동 (순서: 반복마다 nocomp, comp)")
    t_est = len(cmds) * a.num_actions * a.sec_per_action / 60
    print(f"예상 시간(순차, 행동당 {a.sec_per_action:g}초 가정): 약 {t_est:.0f}분 (다른 Genesis 작업과 GPU 를 나누면 더 걸림)")
    for arm, d, c in cmds:
        print(f"$ ({arm}) " + " ".join(c))
    if a.dry_run:
        print("(--dry-run) 실행하지 않았습니다.")
        return
    for arm, d, c in cmds:
        if (d / "summary.json").exists():
            print(f"[건너뜀] 이미 있음: {d}")
            continue
        print(f"\n=== {arm} {d.name} ===", flush=True)
        rc = subprocess.call(c, cwd=ROOT)
        if rc != 0:
            raise SystemExit(f"실패(종료 코드 {rc}): {d}")
    a.arm = [f"nocomp={','.join(map(str, arms['nocomp']))}", f"comp={','.join(map(str, arms['comp']))}"]
    a.commits = commits
    main_collect(a)


def perm_p(x, y):
    """두 표본 평균 차이의 양측 순열검정 p 값(정확 열거). 표본이 너무 크면 None."""
    z = np.concatenate([x, y])
    n, m = len(z), len(x)
    if n > 16:
        return None
    obs = abs(np.mean(y) - np.mean(x))
    cnt = tot = 0
    for idx in itertools.combinations(range(n), m):
        sel = np.zeros(n, bool)
        sel[list(idx)] = True
        tot += 1
        cnt += abs(z[~sel].mean() - z[sel].mean()) >= obs - 1e-12
    return cnt / tot


def main_collect(a):
    import cf_compensate as cfc
    from region_set import RegionSet, border_median
    hist = a.history_dir.resolve()
    K = a.upto or cc.n_actions(hist)
    view = cc.make_view(hist, K, Path(a.view_dir) if a.view_dir else ROOT / "results" / "_repeat_view" / hist.parent.name)
    h_K = cc.state_height(view).astype(np.float64)
    g = cc.geometry(view)
    R = g["tool_r_px"]
    rs = RegionSet.load(a.regions, names=[n for n in a.names.split(",") if n] or None, ref_default=h_K)
    names = rs.names
    sigma = cc.band_sigma(R, 0.4)
    bm0 = border_median(h_K)
    goals_o = cc.original_target_goals(rs, view, h_K)
    crits = {"request": {n: rs.targets[n] for n in names}, "original": goals_o}
    cfg0 = cfc.build_cfg(rs, h_K, 0.0, int(round(0.94 * R)), int(round(1.17 * R)), R)
    norm = cfg0["norm"]

    def score(h, which):
        sh = border_median(h) - bm0
        pre = cc.appearance_pre(rs.masks, {n: crits[which][n] + sh for n in names}, sigma)
        v = cc.appearance(h, pre, R, sigma)
        per = {n: v[n][2] / norm[n][0] + v[n][0] / norm[n][1] + v[n][1] / norm[n][2] for n in names}
        return float(np.mean(list(per.values()))), per

    arms = {}
    for spec in a.arm:
        name, dirs = spec.split("=", 1)
        arms[name] = [Path(d) for d in dirs.split(",") if d]
    C = int(a.commits)
    offset = {"nocomp": 0, "comp": C}

    def traj(d, arm):
        """이어 조형 단계별 높이맵 {총 행동 수 T: h}. nocomp T=step, comp T=C+step (step 0 = 보상 직후 상태)."""
        out = {}
        st = d / "start_m.npy"
        if st.exists():
            out[offset[arm] + 0] = np.load(st).astype(np.float64)
        for p in sorted((d / "states").glob("step_*.npy")):
            out[offset[arm] + int(p.stem.split("_")[1])] = np.load(p).astype(np.float64)
        return out

    trajs = {arm: [traj(d, arm) for d in dirs] for arm, dirs in arms.items()}
    maxT = min(max(t) - (offset[arm] if a.align == "steps" else 0) for arm, ts in trajs.items() for t in ts)
    cps = [int(x) for x in a.checkpoints.split(",")] if a.checkpoints else sorted({T for T in range(5, maxT + 1, 5)} | {maxT})
    print(f"이력 {hist.name} 시점 {K} | 영역 {names} | 확정 행동 C={C} | 반복: " + ", ".join(f"{k} {len(v)}회" for k, v in trajs.items()) + f" | 비교 기준: {'총 행동 수' if a.align == 'total' else '이어 조형 행동 수'}")
    result = {}
    for which in ("request", "original"):
        print(f"\n[{'요청 기준' if which == 'request' else '원래 목표 기준'}] 영역 평균 점수 e (K 번째 상태 = 3.00, 낮을수록 좋음)")
        print(f"{'T':>4} | {'nocomp 평균±sd':>16} {'각 실행':<24} | {'comp 평균±sd':>16} {'각 실행':<24} | {'차이':>7} {'차/sd':>6} {'순열 p':>7}")
        for T in cps:
            vals = {}
            for arm, ts in trajs.items():
                key = T if a.align == "total" else T + offset[arm]      # total: 키가 이미 총 행동 수 / steps: comp 는 보상 확정 수만큼 뒤
                vals[arm] = [score(t[key], which)[0] for t in ts if key in t]
            if len(vals.get("nocomp", [])) == 0 or len(vals.get("comp", [])) == 0:
                continue
            x, y = np.array(vals["nocomp"]), np.array(vals["comp"])
            sx = x.std(ddof=1) if len(x) > 1 else float("nan")
            sy = y.std(ddof=1) if len(y) > 1 else float("nan")
            pooled = np.sqrt(np.nanmean([sx ** 2, sy ** 2])) if len(x) > 1 and len(y) > 1 else float("nan")
            diff = float(y.mean() - x.mean())
            p = perm_p(x, y)
            result.setdefault(which, {})[T] = dict(nocomp=x.tolist(), comp=y.tolist(), diff=diff, pooled_sd=float(pooled), p=p)
            print(f"{T:>4} | {x.mean():>8.3f}±{sx:<6.3f} {np.array2string(x, precision=3):<24} | {y.mean():>8.3f}±{sy:<6.3f} {np.array2string(y, precision=3):<24} | {diff:>+7.3f} {diff / pooled if pooled == pooled and pooled > 0 else float('nan'):>6.2f} "
                  + (f"{p:>7.3f}" if p is not None else "      -"))
    Tl = max([T for T in cps if T <= maxT] or [cps[0]])         # 모든 팔·반복에 데이터가 있는 가장 큰 체크포인트
    print(f"\n[영역별, 총 행동 {Tl}, 요청 기준] 값 = 반복 평균 (nocomp -> comp)")
    for n in names:
        row = {}
        for arm, ts in trajs.items():
            key = Tl if a.align == "total" else Tl + offset[arm]
            row[arm] = np.mean([score(t[key], "request")[1][n] for t in ts if key in t])
        print(f"  {n:<14} {row['nocomp']:.3f} -> {row['comp']:.3f} ({row['comp'] - row['nocomp']:+.3f})")
    print("\n참고: 반복이 적으면(3 이하) sd 와 p 값은 거칠다. 차이가 같은 팔 안의 변동(sd)보다 충분히 크고 p 가 작을 때만 '효과 있음'으로 읽고, 그렇지 않으면 '구분 못 함'이다.")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(dict(history=str(hist), K=K, commits=C, align=a.align, arms={k: [str(d) for d in v] for k, v in arms.items()}, result=result), ensure_ascii=False, indent=1))
        print("저장:", a.json_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "collect"):
        p = sub.add_parser(nm)
        p.add_argument("--history-dir", required=True, type=Path)
        p.add_argument("--upto", type=int, help="이어 조형을 시작하는 이력 시점 K (기본: 마지막)")
        p.add_argument("--regions", required=True, type=Path)
        p.add_argument("--names", default="")
        p.add_argument("--checkpoints", default="", help="비교할 총 행동 수들(쉼표). 기본: 5 간격과 마지막")
        p.add_argument("--align", choices=["total", "steps"], default="total")
        p.add_argument("--json-out", type=Path)
        p.add_argument("--view-dir", help="읽기 전용 이력 보기를 만들 폴더 (기본: results/_repeat_view/...)")
    pr = sub.choices["run"]
    pr.add_argument("--cf-dir", required=True, type=Path, help="cf_compensate 출력 폴더 (summary.json, states/iter_XXX.pt)")
    pr.add_argument("--checkpoint", required=True, type=Path)
    pr.add_argument("--out", required=True, type=Path)
    pr.add_argument("--target-npy", type=Path, help="계획기가 따라갈 새 목표 전체 맵(512x512, m, 절대 높이, 예: build_roi_edit_target.py 의 h_edit). 없으면 영역 목표를 K 번째 상태에 합친 맵")
    pr.add_argument("--num-actions", type=int, default=20)
    pr.add_argument("--repeats", type=int, default=3)
    pr.add_argument("--seed0", type=int, default=1)
    pr.add_argument("--plan-horizon", type=int, default=40)
    pr.add_argument("--candidates", type=int, default=64)
    pr.add_argument("--refine-actions", type=int, default=40)
    pr.add_argument("--refine-steps", type=int, default=5)
    pr.add_argument("--mpc-execute", type=int, default=5)
    pr.add_argument("--sec-per-action", type=float, default=35.0, help="시간 추정용 (계획 포함 행동당 초)")
    pr.add_argument("--dry-run", action="store_true")
    pc = sub.choices["collect"]
    pc.add_argument("--commits", type=int, required=True, help="comp 팔이 시작 전에 이미 쓴 보상 확정 행동 수 C")
    pc.add_argument("--arm", action="append", required=True, help="이름=디렉터리,디렉터리,... (이름: nocomp 또는 comp)")
    a = ap.parse_args()
    if a.cmd == "run":
        main_run(a)
    else:
        main_collect(a)


if __name__ == "__main__":
    main()
