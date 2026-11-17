#!/usr/bin/env python3
"""cf_chain.py - 반사실 국소 보상을 '계속 과거를 타고 들어가며' 여러 라운드 이어 가는 실행기 (빔 탐색).

한 라운드(단계) = cf_compensate.py 한 번 실행(--max-commits 1). 조건(D5, E3, E5 ...)은 ablation_orchestrate.CONDS 의 것을 그대로 쓴다.
단계가 끝나면 확정한 행동(과 빔이면 차순위 후보)을 extend_history.py 로 이력 끝에 붙여 '원본 K개 + 확정한 행동' 이력을 만들고,
다음 단계는 그 새 이력에서 같은 방법을 처음부터 다시 돈다. 그래서 단계마다 새 현재를 기준으로
  - 과거 행동 순위(제한 최소제곱)와 반사실 재생: C* = H_cf - H_ctrl, 반사실 목표 T = 현재 + C*  (--cf-target fixed 면 그 단계 안에서 T 고정)
  - 먼 과거 트리(--long-select tree), 개입 묶음(seq), 장거리(②L)  (조건이 E1/E2/E3/E4/E5 일 때)
  - 방금 한 보상 행동도 과거 행동이 되어 반사실 변형(더/덜 누름)과 되돌리기 후보(CFR)의 대상이 된다
가 모두 다시 계산된다. cf_compensate.py 는 한 줄도 고치지 않는다(실행 중인 대기열에 영향 없음).

빔: 단계마다 노드(이력 끝 상태)마다 자식 후보를 최대 --beam 개 만든다 = 확정한 행동 + 그 라운드 평가 기록(eval_round_*.json)에서 한도를 지키고 확정 기준보다 J 가
    --eps-j 이상 낮은, 서로 충분히 다른 차순위 후보. 모든 자식을 실제로 실행해(extend_history.py) 점수를 매기고, 노드 전체에서 J 가 낮은 --beam 개만 다음 단계로 넘긴다.
점수 J: 고칠 영역 하나의 (형태+기울기+테두리+꼬리), 정규화 기준은 영역 폴더의 h_ref(시작 시점 K 상태)라서 단계가 달라도 같은 척도의 '누적' 점수다(시작 3.0). 영역 밖 변화·보호 영역 변화도 K 상태 대비 누적이다.
목표: 첫 단계의 --level-adjust 가 줄인 목표(점토 총량 보존)를 고정해 이후 단계에 쓴다(영역 폴더 사본 regions_adj, 이후 단계는 --level-adjust 를 끈다).

비용: 노드 하나 = cf_compensate 한 번(D5 약 18분, E3 약 3~4시간) + 자식 확장(Genesis 초기화 1~2분). 라운드 R, 빔 B 면 1 + (R-1)*B 번. 이어서 하기: 이미 끝난 단계(summary.json)와 확장(extend.json)은 건너뛴다.
GPU: 한 번에 노드 하나씩만 돌린다. 시작 전에 GPU 여유가 모자라면 기다린다(--no-wait 로 끔). 대기열(orchestrate_v2)과 같이 쓸 때 메모리가 모자라 실패하지 않게 하기 위해서다.
근사: 후보 유효성은 cf_compensate.within 의 먼 곳/보호 영역/충돌 한도만 따라 한다(여러 영역을 동시에 고칠 때의 side_worse 조건은 빠져 있다 - 영역 하나만 고치는 설정 전용).

사용: python cf_chain.py --active fill3 --out results/case_A/chain_hf3 --cond D5 --rounds 3 --beam 2 [--dry-run]
산출: OUT/chain.json(전체 트리), OUT/chain_report.md, OUT/chain.png(시작 -> 라운드별 결과 -> 목표), OUT/d<단계>_<번호>/ (각 노드의 cf_compensate 출력과 확장 이력)
"""
import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ablation_orchestrate as ab      # noqa: E402  조건 표(CONDS)와 명령 만드는 함수(cmd_for)
import cf_common as cc                 # noqa: E402

PY = ab.PY


def log(msg):
    print(f"[{time.strftime('%T')}] {msg}", flush=True)


def wrap(t):
    return math.atan2(math.sin(t), math.cos(t))


def seq_of(a):
    """행동 하나([5]) 또는 순서쌍([[5],[5]]) -> 항상 [[5], ...]"""
    return [list(map(float, x)) for x in a] if isinstance(a[0], (list, tuple)) else [list(map(float, a))]


def distinct(a, b, pos_mm=3.0, th=0.35, ld_mm=3.0):
    """두 후보가 서로 다른 푸시인가(위치 pos_mm, 각도 th rad, 길이·깊이 ld_mm 중 하나라도 크게 다르면)."""
    sa, sb = seq_of(a), seq_of(b)
    if len(sa) != len(sb):
        return True
    return any(math.hypot(p[0] - q[0], p[1] - q[1]) * 1000 > pos_mm or abs(wrap(p[2] - q[2])) > th or abs(p[3] - q[3]) * 1000 > ld_mm or abs(p[4] - q[4]) * 1000 > ld_mm for p, q in zip(sa, sb))


def wait_gpu(need_mb, margin=1200, poll=60):
    t0, last = time.time(), 0.0
    while True:
        free = ab.gpu_free_mb()
        if free - margin >= need_mb:
            return
        if time.time() - last > 600:
            log(f"GPU 여유 {free} MiB < 필요 {need_mb + margin} MiB: 기다리는 중 ({(time.time() - t0) / 60:.0f}분째)")
            last = time.time()
        time.sleep(poll)


def run_logged(cmd, logfile, timeout_s):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        while p.poll() is None:
            if time.time() - t0 > timeout_s:
                p.kill()
                f.write("\n[시간 초과로 중단]\n")
                return 124
            time.sleep(10)
        return p.returncode


class Scorer:
    """고칠 영역의 누적 점수. 정규화 기준 h_ref 와 영역 목표(조정된 것)는 단계가 달라도 같다. 시작 상태(K 상태) 대비 손상도 같이 잰다."""

    def __init__(self, regions_adj, root0, weights, active, target_npy=None):
        import cf_compensate as cfc
        from region_set import RegionSet
        self.cfc, self.active = cfc, active
        self.root0 = root0
        self.h0 = cc.state_height(root0).astype(np.float64)
        self.rs = RegionSet.load(Path(regions_adj), names=None, ref_default=self.h0)
        R = cc.geometry(root0)["tool_r_px"]
        w = weights
        self.cfg = cfc.build_cfg(self.rs, self.rs.h_ref, w.get("mu", 1.0), int(round(0.94 * R)), int(round(1.17 * R)), R,
                                 (w.get("height", 1.0), w.get("grad", 1.0), w.get("edge", 1.0)), w.get("blur_tools", 0.4), w_tail=w.get("tail", 0.0))
        from region_set import border_median
        t_ = cc.load_target(root0, target_npy).astype(np.float64)       # --target-npy 를 주면 이력의 옛 목표 대신 그 목표(조형 때와 같은 변환 적용)를 쓴다
        self.goal = t_ - border_median(t_) + border_median(self.rs.h_ref)           # 원래 A 목표를 판 배경 높이에 맞춘 것(영역 목표의 바탕과 같다)
        self.foot = np.abs(self.goal - border_median(self.rs.h_ref)) >= 1e-3         # 목표 부조 영역 = |목표 - 배경| >= 1 mm (이 실행기의 정의. 메모의 'A-footprint 30,758px' 와는 정의가 다르다)

    def __call__(self, h):
        h = np.asarray(h, np.float64)
        s = self.cfc.score(h, self.cfg, self.h0)
        rj = {n: float(self.cfc.region_J(n, s["vis"], self.cfg)) for n in self.rs.names}
        if self.active == "all":      # 영역 전체를 동시에 고친다: 점수 = 영역 평균 J(시작 3.0), 영역별 J 도 같이 기록(보호 영역 없음)
            per = np.mean([s["per"][n] for n in self.rs.names], axis=0)
            return dict(J=float(np.mean(list(rj.values()))), mae_mm=float(per[0]), rmse_mm=float(per[1]), far_max_mm=float(s["far_max"]), prot_mm={}, regions_J=rj,
                        regions_mae={n: float(s["per"][n][0]) for n in self.rs.names}, foot_mae=float(np.abs(h - self.goal)[self.foot].mean() * 1000.0))
        per = s["per"][self.active]
        d = np.abs(h - self.h0)
        prot = {n: float(d[np.asarray(self.rs.masks[n]) > 0].max() * 1000.0) for n in self.rs.names if n != self.active}
        return dict(J=rj[self.active], mae_mm=float(per[0]), rmse_mm=float(per[1]), far_max_mm=float(s["far_max"]), prot_mm=prot, regions_J=rj,
                    regions_mae={n: float(s["per"][n][0]) for n in self.rs.names}, foot_mae=float(np.abs(h - self.goal)[self.foot].mean() * 1000.0))


def make_regions_adj(src_regions, cf_dir, active, out):
    """첫 단계가 줄인 목표(level_adjust)를 영역 폴더 사본에 넣는다. 원본 영역 폴더는 건드리지 않는다(사본의 링크를 끊고 쓴다)."""
    out = Path(out)
    if out.exists():
        return out
    shutil.copytree(src_regions, out, symlinks=True)
    found = sorted((Path(cf_dir) / "level_adjust").glob("*_target_adjusted.npy"))      # 고치는 영역마다(영역 하나면 그 영역만, --active all 이면 전부)
    for adj in found:
        dst = out / f"{adj.name[:-len('_target_adjusted.npy')]}_target.npy"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        shutil.copy(adj, dst)
    if not found:
        log(f"(조정된 목표가 없다: {Path(cf_dir) / 'level_adjust'}) 원본 목표를 그대로 쓴다")
    return out


def stage_cmd(a, node, regions, first):
    all_mode = a.active == "all"
    spec = dict(history_dir=node["root"], upto=node["K"], regions=str(regions), active=(ab.region_names(str(regions))[0] if all_mode else a.active), out=str(node["dir"]))
    cmd = ab.cmd_for(a.cond, a.seed, str(node["dir"]), spec)
    if all_mode:                                   # 영역 전체를 동시에 고친다(보호 영역 없음). 단계마다 확정 1번이라 오차가 가장 큰 영역 하나만 반사실을 계산한다
        i_ = cmd.index("--skip-regions")
        del cmd[i_:i_ + 2]
        cmd.append("--cf-top1")
        if a.rotate and node.get("done"):           # 이미 고친 영역은 이번 단계에서 고치지 않고 보호로 돌린다(그 영역에 해를 주는 변화는 한도로 막고, 목표 쪽 변화는 허용). 오차가 가장 큰 '남은' 영역이 다음 활성이 된다
            cmd += ["--skip-regions", ",".join(node["done"])]
    if not first:
        cmd = [c for c in cmd if c != "--level-adjust"]          # 목표는 첫 단계에서 줄인 것으로 고정
    if a.stage_budget_min:
        cmd[cmd.index("--budget-min") + 1] = str(a.stage_budget_min)
    if a.target_npy:                                             # 이력의 목표가 아닌 다른 목표(예: 하트를 뒤집은 것)로 원래 목표 기준 점수를 잰다. 영역 목표는 영역 폴더에 이미 들어 있다
        cmd += ["--target-npy", a.target_npy]
    if a.viewer:                                                 # 워커 0 에 Genesis 창: 후보를 눌러 보고 되돌리는 모습과 확정 푸시 실행이 보인다(먼 과거 재생은 창 없는 워커에서 돈다)
        cmd.append("--viewer")
    if a.commits_per_stage > 1:                                  # 한 단계에서 푸시를 여러 번 확정한다: 반사실 재생(단계에서 가장 비싼 부분)은 한 번만 하고 후보 평가·확정을 그 횟수만큼 반복
        for fl_ in ("--max-commits", "--per-region-commits"):
            cmd[cmd.index(fl_) + 1] = str(a.commits_per_stage)
    if a.extra:                                                  # 조건 표(CONDS)를 고치지 않고 플래그를 덧붙인다. 예: --extra="--plan-refine 40 --plan-seq 3"
        cmd += shlex.split(a.extra)
    return cmd


def cf_evidence(cf_dir, run_log, active):
    """그 단계에서 반사실이 실제로 쓰였는지: 재생 행 수, control 통과 수, 최대 q, 로그의 '=> 영역: ...' 줄."""
    ev = dict(rows=0, control_ok=0, q_max=None, kinds=[], used=None)
    p = Path(cf_dir) / "cf_replay.json"
    if p.exists():
        try:
            rows = json.loads(p.read_text())
            rows = rows if isinstance(rows, list) else rows.get("rows", [])
            qs = [r.get("q") for r in rows if r.get("q") is not None]
            ev.update(rows=len(rows), control_ok=sum(bool(r.get("control_ok")) for r in rows), q_max=(max(qs) if qs else None), kinds=sorted({str(r.get("kind")) for r in rows}))
        except Exception as e:      # 증거 수집 실패가 실행을 멈추지 않게
            ev["error"] = repr(e)
    try:
        for line in Path(run_log).read_text(errors="ignore").splitlines():
            if line.strip().startswith(f"=> '{active}'"):
                ev["used"] = line.strip()
    except Exception:
        pass
    return ev


def pick_children(cf_dir, rec, summ, beam, eps_j, ring_budget_mm):
    """확정한 행동(best) + 그 라운드 평가 기록에서 한도를 지키고 확정 기준보다 J 가 eps_j 이상 낮은, 서로 다른 차순위 후보. 최대 beam 개."""
    kids = [dict(kind="best", actions=seq_of(rec["action"]), J_pred=float(rec["best_J"]), family=rec.get("family"), base=rec.get("base"))]
    if beam <= 1:
        return kids
    ev_path = Path(cf_dir) / f"eval_round_{int(rec['round']):02d}.json"
    if not ev_path.exists():
        log(f"(평가 기록이 없어 빔 차순위를 못 만든다: {ev_path})")
        return kids
    far_lim, prot_lim = float(summ.get("far_limit_mm", 1e9)), summ.get("protect_limits", {})

    def valid(r):
        if r.get("J", 1e9) >= 1e8 or r["J"] >= float(rec["cur_J"]) - eps_j or r.get("far_max", 0.0) > far_lim:
            return False
        if any(d > prot_lim.get(n, 1e9) for n, d in (r.get("prot") or {}).items()):
            return False
        return r.get("coll", 0.0) <= ring_budget_mm

    pool = sorted((r for r in json.loads(ev_path.read_text()) if valid(r)), key=lambda r: r["J"])
    for r in pool:
        if len(kids) >= beam:
            break
        if all(distinct(r["action"], k["actions"] if len(k["actions"]) > 1 else k["actions"][0]) for k in kids):
            kids.append(dict(kind="alt", actions=seq_of(r["action"]), J_pred=float(r["J"]), family=r.get("family"), base=r.get("base")))
    return kids


def make_figure(out, scorer, path_nodes, active, target_adj):
    """시작 | 라운드별 결과 | (조정된) 목표 를 고칠 영역 주변만 잘라 같은 색 범위로 보여 준다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = np.any([np.asarray(scorer.rs.masks[n]) > 0 for n in scorer.rs.names], axis=0) if active == "all" else np.asarray(scorer.rs.masks[active]) > 0
        rr, cc_ = np.where(m)
        pad = 45
        r0, r1, c0, c1 = max(rr.min() - pad, 0), min(rr.max() + pad, 512), max(cc_.min() - pad, 0), min(cc_.max() + pad, 512)
        panels = [("start (K)", scorer.h0)] + [(f"round {n['depth']}  J={n['score']['J']:.2f}", np.load(Path(n["root"]) / "ext_final_m.npy")) for n in path_nodes if n["depth"] >= 1] + [("target (adjusted)", target_adj)]
        crops = [np.asarray(h, np.float64)[r0:r1, c0:c1] * 1000 for _, h in panels]
        lo, hi = np.percentile(np.concatenate([c.ravel() for c in crops]), [1, 99])
        fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
        for ax, (title, _), cr in zip(np.atleast_1d(axs), panels, crops):
            ax.imshow(cr, cmap="turbo", vmin=lo, vmax=hi)
            ax.contour(m[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=1.0)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{active}: height (mm), outline = edited region", fontsize=10)
        fig.tight_layout()
        fig.savefig(Path(out) / "chain.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"(그림을 못 그렸다: {e!r})")


def stage_figure(out, scorer, node, depth, region):
    """판 전체 높이맵: 시작 | 지금 | 목표(A, 배경 정렬) | 지금 - 목표(mm). 검은 윤곽 = 고치는 영역 전체, 흰 윤곽 = 방금 이 단계가 고친 영역."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = np.load(Path(node["root"]) / "ext_final_m.npy").astype(np.float64) * 1000
        h0, g = scorer.h0 * 1000, scorer.goal * 1000
        lo, hi = np.percentile(np.concatenate([h0.ravel(), g.ravel()]), [1, 99.5])
        union = np.any([np.asarray(scorer.rs.masks[n]) > 0 for n in scorer.rs.names], axis=0)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
        try:      # 제목: 시작 이력의 푸시 수와 목표 이름을 대상에 맞게 쓴다(예전에는 "start (80 pushes)", "target A" 로 고정돼 하트 그림에도 A 라고 나왔다)
            import cf_common as cc_t
            start_t = f"start ({cc_t.n_actions(scorer.root0)} pushes)"
        except Exception:
            start_t = "start"
        for ax, (t_, a_) in zip(axs[:3], [(start_t, h0), (f"after stage {depth}", h), ("target (plate-aligned)", g)]):
            ax.imshow(a_, cmap="turbo", vmin=lo, vmax=hi)
            ax.set_title(t_, fontsize=10)
        axs[3].imshow(h - g, cmap="RdBu_r", vmin=-6, vmax=6)
        axs[3].set_title("current - target (mm; red = too high, blue = too low)", fontsize=10)
        for ax in axs:
            ax.contour(union, levels=[0.5], colors="k", linewidths=0.6)
            for rg_ in str(region).split(","):          # 한 단계에서 여러 영역을 고쳤으면 전부 표시
                if rg_ in scorer.rs.names:
                    ax.contour(np.asarray(scorer.rs.masks[rg_]) > 0, levels=[0.5], colors="w", linewidths=1.6)
            ax.axis("off")
        sc = node["score"]
        rm = list(sc.get("regions_mae", {0: float("nan")}).values())
        fig.suptitle(f"stage {depth}: fixed {region} (white) | relief-area MAE {sc.get('foot_mae', float('nan')):.2f} mm | region MAE mean {np.mean(rm):.2f} / max {np.max(rm):.2f} mm | mean J {sc['J']:.3f}", fontsize=11)
        fig.tight_layout()
        (Path(out) / "stage_images").mkdir(exist_ok=True)
        p = Path(out) / "stage_images" / f"stage_{depth:03d}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        return p
    except Exception as e:
        log(f"(단계 그림을 못 그렸다: {e!r})")
        return None


def write_progress(out, progress):
    cols = ["stage", "region", "mean_J", "region_MAE_mean", "region_MAE_max", "footprint_MAE", "far_max_mm", "minutes"]
    (Path(out) / "progress.csv").write_text(",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in progress) + "\n")
    L = ["# 단계별 진행 (판 전체 지도: stage_images/)", "", "| 단계 | 고친 영역 | 평균 J | 영역 MAE 평균(mm) | 영역 MAE 최대(mm) | 부조 영역 MAE(mm) | 영역 밖 최대(mm) | 단계 소요(분) |", "|---|---|---|---|---|---|---|---|"]
    for r in progress:
        L.append(f"| {r['stage']} | {r['region']} | {r['mean_J']:.3f} | {r['region_MAE_mean']:.2f} | {r['region_MAE_max']:.2f} | {r['footprint_MAE']:.2f} | {r['far_max_mm']:.1f} | {r['minutes']:.0f} |")
    (Path(out) / "progress.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", default=ab.H)
    ap.add_argument("--upto", type=int, default=80)
    ap.add_argument("--regions", default=ab.REGIONS)
    ap.add_argument("--active", required=True, help="고칠 영역 하나(나머지는 점수만 기록하고 보호)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", default="D5", choices=sorted(ab.CONDS), help="한 단계에서 쓸 조건(ablation_orchestrate.CONDS). D5=가까운 과거만(빠름), E3/E5=먼 과거 트리·묶음 포함(단계당 수 시간)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps-j", type=float, default=0.01, help="차순위 후보는 확정 기준 J 보다 이만큼 이상 낮아야 한다")
    ap.add_argument("--ring-budget-mm", type=float, default=50.0)
    ap.add_argument("--stage-budget-min", type=float, default=None, help="단계별 cf_compensate --budget-min 덮어쓰기")
    ap.add_argument("--viewer", action="store_true", help="단계의 cf_compensate 워커 0 과 이력 확장에 Genesis 창을 띄운다(DISPLAY 필요, 렌더링 때문에 조금 느려진다)")
    ap.add_argument("--repeat", action="store_true", help="--rotate 와 함께: 모든 영역을 한 바퀴 고치고 나면 새 바퀴를 이어서 돈다. 기준(--stop-mae) 이하로 내려온 영역은 건너뛰고, 아래 멈춤 기준 중 하나가 되면 끝낸다. --rounds 는 최대 단계 수")
    ap.add_argument("--stop-mae", type=float, default=1.0, help="--repeat: 영역 MAE(저장된 영역 목표 대비, mm)가 이 값 이하인 영역은 더 고치지 않는다. 모든 영역이 이 값 이하면 끝")
    ap.add_argument("--patience", type=int, default=8, help="--repeat: 최근 이 단계 수 동안 평균 J 가 --stagnation-eps 이상 좋아지지 않으면 끝(0=끔)")
    ap.add_argument("--stagnation-eps", type=float, default=0.01)
    ap.add_argument("--max-hours", type=float, default=0.0, help="--repeat: 이 시간(시간)을 넘기면 끝(0=끔)")
    ap.add_argument("--commits-per-stage", type=int, default=1, help="한 단계(반사실 재생 1회)에서 푸시를 확정하는 횟수(빔 1 전용). 확정을 거듭하면 반사실 힌트 T 는 단계 처음 것을 그대로 쓰므로 조금씩 낡는다. 1 = 예전 동작")
    ap.add_argument("--target-npy", default="", help="원래 목표 높이맵(512x512, m, 원본 좌표). 주면 이력의 목표 대신 쓴다(점수·그림용). 예: results/case_heart/heart_target_raw_flipud_512_m.npy")
    ap.add_argument("--rotate", action="store_true", help="--active all 에서 영역을 돌아가며 한 번씩 고친다: 한 단계가 고친 영역은 이후 단계에서 보호로 돌리고, 오차가 가장 큰 남은 영역이 다음 활성이 된다. --rounds 를 영역 수로 준다")
    ap.add_argument("--extra", default="", help="단계마다 cf_compensate 명령 끝에 덧붙일 플래그 한 문자열. 값이 '--'로 시작하므로 --extra=\"--plan-refine 40 --plan-seq 3\" 처럼 등호로 쓴다")
    ap.add_argument("--need-mb", type=int, default=ab.NEED_MB, help="단계를 시작하기 전에 필요한 GPU 여유(MiB)")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--skip-stuck", action="store_true", help="--rotate 와 함께: 한 단계에서 확정하지 못하고 멈춘 영역(이득 있는 후보가 없다: --min-commit-gain 가드)을 이번 바퀴에서 건너뛴다. 모든 영역이 그렇게 멈추면 이력을 늘리지 않고 같은 이력으로 다음 영역을 고친다(정체 기준이 무한 반복을 막는다)")
    ap.add_argument("--dry-run", action="store_true", help="첫 단계 명령만 보여 주고 끝낸다")
    a = ap.parse_args()
    a.history_dir, a.regions, a.out = [str(p if Path(p).is_absolute() else ROOT / p) for p in (a.history_dir, a.regions, a.out)]
    if a.commits_per_stage > 1 and a.beam != 1:
        log(f"(--commits-per-stage {a.commits_per_stage} 는 빔 1 전용이라 --beam {a.beam} 을 1 로 바꾼다)")
        a.beam = 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root0 = cc.make_view(a.history_dir, a.upto, out / "d0" / "view")
    K0 = cc.n_actions(root0)
    root_node = dict(id="d0", depth=0, parent=None, root=str(root0), K=K0, dir=str(out / "d0"), actions=[], how="start", done=[])
    if a.dry_run:
        print(" ".join(stage_cmd(a, root_node, a.regions, True)))
        return

    nodes = [root_node]
    regions_adj, scorer, summ0 = None, None, None
    frontier = [root_node]
    progress, hist, t_start = [], [], time.time()
    for depth in range(1, a.rounds + 1):
        children = []
        for j, node in enumerate(frontier):
            node_dir = Path(node["dir"])
            cf = node_dir / f"{a.cond}_s{a.seed}" / "cf"
            run_log = node_dir / f"{a.cond}_s{a.seed}" / "run.log"
            if a.rotate and scorer is not None:       # 이번 단계에서 고치지 않을 영역 = 이번 바퀴에 이미 고친 영역 + 이미 기준(--stop-mae) 이하인 영역
                names_all = list(scorer.rs.names)
                done_ = list(node.get("done", []))
                low_ = [n_ for n_ in names_all if node["score"].get("regions_mae", {}).get(n_, 1e9) <= a.stop_mae]
                if a.repeat and (set(done_) | set(low_)) >= set(names_all) and len(low_) < len(names_all):
                    log(f"[단계 {depth}] 한 바퀴 완료 -> 새 바퀴 시작 (MAE {a.stop_mae:g} mm 이하인 영역 {len(low_)}개는 건너뜀)")
                    done_ = []
                node["done"] = [n_ for n_ in names_all if n_ in set(done_) | set(low_)]
            first = depth == 1
            regions = a.regions if first else str(regions_adj)
            if (cf / "summary.json").exists():
                log(f"[단계 {depth}] {node['id']}: 이미 끝난 실행을 재사용 ({cf})")
            else:
                if not a.no_wait:
                    wait_gpu(a.need_mb)
                log(f"[단계 {depth}] {node['id']}: cf_compensate 시작 (이력 행동 {node['K']}개, 조건 {a.cond})")
                t0 = time.time()
                rc = run_logged(stage_cmd(a, node, regions, first), run_log, ab.CONDS[a.cond][2])
                log(f"[단계 {depth}] {node['id']}: 종료 rc={rc} ({(time.time() - t0) / 60:.0f}분)")
            if not (cf / "summary.json").exists():
                node["status"] = "실패(summary.json 없음)"
                continue
            summ = json.loads((cf / "summary.json").read_text())
            rounds = json.loads((cf / "rounds.json").read_text()) if (cf / "rounds.json").exists() else []
            committed = [r for r in rounds if r.get("committed")]
            node["cf"] = cf_evidence(cf, run_log, (rounds[0].get("region") if (a.active == "all" and rounds) else a.active))
            node["minutes"] = summ.get("minutes")
            if first:                                     # 첫 단계가 끝나야 조정된 목표와 점수 가중치를 안다
                regions_adj = make_regions_adj(a.regions, cf, a.active, out / "regions_adj")
                summ0 = summ
                scorer = Scorer(regions_adj, root0, summ.get("weights", {}), a.active, a.target_npy or None)
                root_node["score"] = scorer(scorer.h0)
                hist.append((0, root_node["score"]["J"]))
            if not committed:
                node["status"] = "확정 없음(개선되는 후보가 없다)"
                stuck_regs = list(dict.fromkeys(r_.get("region") for r_ in rounds if not r_.get("committed") and r_.get("region")))
                if a.skip_stuck and a.rotate and stuck_regs:      # 같은 이력으로 이어 가되, 멈춘 영역은 이번 바퀴에서 건너뛴다(확장 없음: 이력 그대로)
                    child = dict(id=f"{node['id']}.s", depth=depth, parent=node["id"], root=node["root"], K=node["K"], dir=None, actions=node["actions"], how="skip", family=None, base=None,
                                 J_pred=node["score"]["J"], score=node["score"], repro=None, restore=None, region=",".join(stuck_regs),
                                 done=list(node.get("done", [])) + stuck_regs)
                    nodes.append(child)
                    children.append(child)
                    log(f"[단계 {depth}] {node['id']}: 확정 없음 -> 영역 {stuck_regs} 은(는) 이번 바퀴에서 건너뛰고 같은 이력(K={node['K']})으로 다음 영역을 고친다")
                    continue
                log(f"[단계 {depth}] {node['id']}: 확정 없음 -> 이 가지는 여기서 끝")
                continue
            rec = committed[0]
            ref_J = node["score"].get("regions_J", {}).get(rec.get("region"), node["score"]["J"])      # 그 단계가 고친 영역의 누적 J
            if not first and abs(float(rec["cur_J"]) - float(ref_J)) > 0.01:
                log(f"(주의) {node['id']}: 단계 시작 J {rec['cur_J']:.3f} 와 실행기가 잰 누적 J {ref_J:.3f} 가 다르다 - 영역 목표/기준이 어긋났는지 확인")
            if a.commits_per_stage > 1:      # 한 단계에서 여러 번 확정한 경우: 확정한 행동 전부를 순서대로 이력에 붙인다(빔 1 전용)
                acts_all = []
                for r_ in committed:
                    acts_all += seq_of(r_["action"])
                kids = [dict(kind="best", actions=acts_all, J_pred=float(committed[-1]["best_J"]), family="+".join(str(r_.get("family")) for r_ in committed), base=None)]
            else:
                kids = pick_children(cf, rec, summ, a.beam, a.eps_j, a.ring_budget_mm)
            spec, ext_dirs = [], []
            for i, k in enumerate(kids):
                ed = node_dir / f"child{i}"
                ext_dirs.append(ed)
                item = dict(out=str(ed), actions=k["actions"])
                n_c = len(committed) if a.commits_per_stage > 1 else 1
                if k["kind"] == "best" and (cf / "states" / f"iter_{n_c:03d}.npy").exists():
                    item["expect_npy"] = str(cf / "states" / f"iter_{n_c:03d}.npy")       # 워커가 확정한 결과와 새 프로세스 재실행 결과를 맞춰 본다
                spec.append(item)
            todo = [s for s in spec if not (Path(s["out"]) / "extend.json").exists()]
            if todo:
                spec_p = node_dir / "extend_spec.json"
                spec_p.write_text(json.dumps(todo, ensure_ascii=False, indent=1))
                if not a.no_wait:
                    wait_gpu(3500)
                rc = run_logged([PY, "-u", "extend_history.py", "--src", node["root"], "--spec", str(spec_p)] + (["--viewer"] if a.viewer else []), node_dir / "extend.log", 3600)
                log(f"[단계 {depth}] {node['id']}: 자식 {len(todo)}개 확장 rc={rc}")
            for i, (k, ed) in enumerate(zip(kids, ext_dirs)):
                if not (ed / "extend.json").exists():
                    continue
                ext = json.loads((ed / "extend.json").read_text())
                sc = scorer(np.load(ed / "ext_final_m.npy"))
                child = dict(id=f"{node['id']}.{i}", depth=depth, parent=node["id"], root=str(ed), K=ext["n_after"], dir=None, actions=node["actions"] + k["actions"], how=k["kind"],
                             family=k.get("family"), base=k.get("base"), J_pred=k["J_pred"], score=sc, repro=ext.get("repro_vs_expect"), restore=ext.get("restore_error_vs_record"), region=(",".join(dict.fromkeys(r_.get("region") for r_ in committed)) if a.commits_per_stage > 1 else rec.get("region")),
                             done=list(node.get("done", [])) + ([x_ for x_ in dict.fromkeys(r_.get("region") for r_ in (committed if a.commits_per_stage > 1 else [rec])) if x_] if a.rotate else []) + ([x_ for x_ in dict.fromkeys(r_.get("region") for r_ in rounds if not r_.get("committed") and r_.get("region"))] if (a.skip_stuck and a.rotate) else []))
                nodes.append(child)
                children.append(child)
                log(f"    자식 {child['id']} ({k['kind']}, {k.get('family')}{'#' + str(k['base']) if k.get('base') else ''}): J 예측 {k['J_pred']:.3f} -> 실제 누적 J {sc['J']:.3f} | MAE {sc['mae_mm']:.2f}mm | 밖 {sc['far_max_mm']:.1f}mm | 보호 최대 {max(sc['prot_mm'].values(), default=0):.2f}mm"
                    + (f" | 새 프로세스 재현 MAE {ext['repro_vs_expect']['mae_mm']:.4f}mm" if ext.get("repro_vs_expect") else ""))
        if not children:
            log(f"[단계 {depth}] 이어 갈 자식이 없어 끝낸다")
            break
        children.sort(key=lambda c: c["score"]["J"])
        keep, drop = children[:a.beam], children[a.beam:]
        for c in drop:
            c["status"] = "빔에서 탈락"
        st_min = float(sum((n_.get("minutes") or 0) for n_ in frontier))        # 이번 단계 소요(분): 방금 끝난 노드들의 cf_compensate 시간
        frontier = []
        for idx, c in enumerate(keep):
            c["dir"] = str(out / f"d{depth}_{idx}")
            Path(c["dir"]).mkdir(parents=True, exist_ok=True)
            frontier.append(c)
        log(f"[단계 {depth}] 다음 단계로: " + ", ".join(f"{c['id']}(J {c['score']['J']:.3f})" for c in keep))
        (out / "chain.json").write_text(json.dumps(dict(args=vars(a), nodes=nodes), ensure_ascii=False, indent=1, default=str))
        # ---- 단계 보고: 판 전체 지도(시작 | 지금 | 목표 | 차이)와 진행표를 단계마다 쓴다
        if scorer is not None:
            cur_, sc_ = keep[0], keep[0]["score"]
            rmae_ = list(sc_.get("regions_mae", {0: float("nan")}).values())
            fig_p = stage_figure(out, scorer, cur_, depth, cur_.get("region"))
            progress.append(dict(stage=depth, region=cur_.get("region"), mean_J=round(sc_["J"], 4), region_MAE_mean=round(float(np.mean(rmae_)), 3), region_MAE_max=round(float(np.max(rmae_)), 3),
                                 footprint_MAE=round(sc_.get("foot_mae", float("nan")), 3), far_max_mm=round(sc_["far_max_mm"], 2), minutes=round(st_min, 1)))
            write_progress(out, progress)
            log(f"[단계 {depth}] 진행: 평균 J {sc_['J']:.3f} | 영역 MAE 평균 {np.mean(rmae_):.2f} / 최대 {np.max(rmae_):.2f} mm | 부조 영역 MAE {sc_.get('foot_mae', float('nan')):.2f} mm" + (f" | 그림 {fig_p}" if fig_p else ""))
            hist.append((depth, sc_["J"]))
            stop_reason = None
            if a.repeat:
                if np.max(rmae_) <= a.stop_mae:
                    stop_reason = f"모든 영역 MAE 가 {a.stop_mae:g} mm 이하"
                elif a.patience > 0 and len(hist) > a.patience and min(j_ for _, j_ in hist[-a.patience:]) > min(j_ for _, j_ in hist[:-a.patience]) - a.stagnation_eps:
                    stop_reason = f"최근 {a.patience}단계 동안 평균 J 가 {a.stagnation_eps:g} 이상 좋아지지 않음(정체)"
                elif a.max_hours > 0 and time.time() - t_start > a.max_hours * 3600:
                    stop_reason = f"시간 제한 {a.max_hours:g}시간"
            if stop_reason:
                log(f"[단계 {depth}] 멈춤: {stop_reason}")
                break

    # ---- 정리: 가장 낮은 누적 J 의 노드까지의 경로
    scored = [n for n in nodes if n.get("score")]
    best = min(scored, key=lambda n: n["score"]["J"]) if scored else None
    byid = {n["id"]: n for n in nodes}
    path = []
    while best:
        path.append(best)
        best = byid.get(best["parent"])
    path.reverse()
    (out / "chain.json").write_text(json.dumps(dict(args=vars(a), nodes=nodes, best_path=[n["id"] for n in path]), ensure_ascii=False, indent=1, default=str))
    L = [f"# 체인 결과 ({a.active}, 조건 {a.cond}, 라운드 {a.rounds}, 빔 {a.beam}, 시드 {a.seed})", "",
         "| 노드 | 단계 | 종류 | 가족 | 누적 J (시작 3.0) | 영역 MAE(mm) | 영역 밖 최대(mm) | 보호 영역 최대(mm) | 새 프로세스 재현 MAE(mm) | 상태 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in nodes:
        s = n.get("score")
        if not s:
            L.append(f"| {n['id']} | {n['depth']} | {n.get('how')} | | | | | | | {n.get('status', '')} |")
            continue
        L.append(f"| {n['id']} | {n['depth']} | {n.get('how')} | {n.get('family') or ''} | {s['J']:.3f} | {s['mae_mm']:.2f} | {s['far_max_mm']:.1f} | {max(s['prot_mm'].values(), default=0):.2f} | "
                 f"{(n.get('repro') or {}).get('mae_mm', float('nan')):.4f} | {n.get('status', '')} |" + (f" 고친 영역 {n['region']}" if n.get("region") else ""))
    L += ["", f"가장 좋은 경로: {' -> '.join(n['id'] for n in path)}", "", "## 단계별 반사실 사용 증거(그 단계 cf_compensate 의 ② 결과)", ""]
    for n in nodes:
        if n.get("cf"):
            c = n["cf"]
            L.append(f"- {n['id']}: 재생 행 {c['rows']}개, control 통과 {c['control_ok']}개, 최대 q {c['q_max']}, 종류 {c['kinds']} | {c['used'] or '(로그에 영역 결론 줄 없음)'}")
    (out / "chain_report.md").write_text("\n".join(L) + "\n")
    if scorer is not None and path:
        if a.active == "all":
            tgt_fig = scorer.h0.copy()
            for n_ in scorer.rs.names:
                mk_ = np.asarray(scorer.rs.masks[n_]) > 0
                tgt_fig[mk_] = np.asarray(scorer.rs.targets[n_], np.float64)[mk_]
        else:
            tgt_fig = scorer.rs.targets[a.active]
        make_figure(out, scorer, path, a.active, tgt_fig)
    log(f"완료. {out / 'chain_report.md'}")


if __name__ == "__main__":
    main()
