#!/usr/bin/env python3
"""ablation 결과 보고서 (연구 메모 4쪽 검증 설계의 보고 항목). 대상(A, 하트, ...)에 의존하지 않는다: 영역 세트와 고친 영역은 인자로 준다.

입력: ablation 출력 폴더 OUT/<조건>_s<시드>/ (cf/ 안에 rounds.json, eval_round_01.json, summary.json, states/ 와, 재검증이 만든 verify.json)
산출 (OUT/report/):
  report.md / report.json     실행 상태, 평가 횟수별 최선 점수, 시간표, 손상 지표, 재검증, 유효성 경고
  curve.png                   정제 평가 n 개 시점의 '한도를 지키는' 최선 J (조건별 평균, 시드별 얇은 선)
  time.png                    단계별 시간(벽시계 초) 누적 막대
  damage.png                  영역 밖 최대 변화 / 보호 영역 변화(한도 대비) / 고친 영역 MAE 변화
  qc_<조건>.png               확정마다 육안 점검 시트: 시작 높이 | 결과 높이 | 해야 할 일(빨강=채움, 파랑=깎음) | 손상(|변화|)

정의
  평가 횟수 n 의 최선 J: 빔 단계에서 한도를 지킨 후보의 최선 J, 그리고 정제 평가(stage bads/refine/jac)를 도착 순서(seq)대로 n 개까지 보탠 뒤의 최선(낮을수록 좋음).
  '한도를 지킨다' = 먼 곳 최대 변화 <= far_limit_mm, 보호 영역마다 변화 <= protect_limits[영역] (summary.json 에 기록된 값).
사용: python ablation_report.py --dir results/case_A/ablation_v1 --regions results/case_A/regions_all_v5 --active groove_top [--status results/case_A/next_v32/status.txt]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
CONDS = ["D0", "D", "D5", "E1", "E2", "E3", "E4", "E5"]
# 범주색: 검증된 기본 팔레트의 1~4번을 정해진 순서대로(파랑, 주황, 청록, 노랑). 중립색은 같은 팔레트의 표면·글자·격자 토큰.
COL = {"D": "#2a78d6", "E1": "#eb6834", "E2": "#1baf7a", "E3": "#eda100", "E4": "#e87ba4", "E5": "#6250d6", "D0": "#52514e", "D5": "#008300"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
REFINE_STAGES = ("bads", "refine", "jac")
STAGE_COL = {"replay": "#6250d6", "beam": "#008300", "refine": "#e87ba4", "pair": "#eda100", "other": "#c3c2b7"}
N_MARK = (4, 8, 12, 20)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def load_runs(base):
    runs = []
    for d in sorted(Path(base).iterdir()):
        m = re.match(r"^(D0|D5|D|E1|E2|E3|E4|E5|E3S)_s(\d+)$", d.name)
        if not m or not d.is_dir():
            continue
        cf = d / "cf"
        rounds = jload(cf / "rounds.json", []) or []
        r = dict(cond=m.group(1), seed=int(m.group(2)), dir=d, cf=cf, name=d.name, rounds=rounds, r1=(rounds[0] if rounds else {}), summ=jload(cf / "summary.json", {}) or {},
                 evals=jload(cf / "eval_round_01.json", []) or [], verify=jload(d / "verify.json", None), cf_replay=jload(cf / "cf_replay.json", []) or [])
        r["finished"] = bool(r["summ"])
        r["committed"] = bool(r["r1"].get("committed"))
        log = d / "run.log"
        r["log_err"] = bool(log.exists() and "Traceback" in log.read_text(errors="ignore")[-20000:])
        runs.append(r)
    return runs


def feasible(e, far_lim, prot_lim):
    if e.get("J", 1e9) >= 1e8:
        return False
    if far_lim is not None and e.get("far_max", 0.0) > far_lim:
        return False
    for n, d in (e.get("prot") or {}).items():
        if n in prot_lim and d > prot_lim[n]:
            return False
    return True


def curve(run, nmax=21):
    """(시작 J, 빔 최선, [n=0..nmax 의 최선 J]) — 한도를 지킨 후보만."""
    far_lim = run["summ"].get("far_limit_mm")
    prot_lim = run["summ"].get("protect_limits") or {}
    ev = sorted(run["evals"], key=lambda e: (e.get("seq") is None, e.get("seq", 0)))
    start = run["r1"].get("cur_J")
    beam = [e["J"] for e in ev if e.get("stage") == "beam" and feasible(e, far_lim, prot_lim)]
    best = min(beam) if beam else (start if start is not None else np.nan)
    ref = [e for e in ev if e.get("stage") in REFINE_STAGES]
    out = [best]
    for k in range(1, nmax + 1):
        if k <= len(ref) and feasible(ref[k - 1], far_lim, prot_lim):
            best = min(best, ref[k - 1]["J"])
        out.append(best)
    return start, (min(beam) if beam else None), out, len(ref), sum(1 for e in ev if not feasible(e, far_lim, prot_lim) and e.get("J", 1e9) < 1e8)


class Ctx:
    def __init__(self, regions, active):
        self.regions, self.active = Path(regions), active
        j = jload(self.regions / "regions.json", {})
        self.names = [r["name"] for r in (j["regions"] if isinstance(j, dict) and "regions" in j else j)]
        self.masks = {n: np.load(self.regions / f"{n}_mask.npy").astype(bool) for n in self.names}
        self.union = np.zeros((512, 512), bool)
        for m in self.masks.values():
            self.union |= m
        self.far = ~ndimage.binary_dilation(self.union, iterations=25)
        self.tgt = np.load(self.regions / f"{active}_target.npy").astype(float)
        hs = self.regions / "h_star_all.npy"
        self.hstar = np.load(hs).astype(float) if hs.exists() else None


def damage(run, ctx):
    p1 = run["cf"] / "states" / "iter_001.npy"
    if not (run["committed"] and p1.exists()):
        return dict(committed=False)
    h0 = np.load(run["cf"] / "states" / "iter_000.npy").astype(float)
    h1 = np.load(p1).astype(float)
    dd = (h1 - h0) * 1000.0
    m = ctx.masks[ctx.active]
    prot = {n: float(np.abs(dd[ctx.masks[n]]).max()) for n in ctx.names if n != ctx.active}
    lim = run["summ"].get("protect_limits") or {}
    return dict(committed=True, far_max=float(np.abs(dd)[ctx.far].max()), far_mean=float(np.abs(dd)[ctx.far].mean()), prot=prot,
                prot_ratio=max([prot[n] / lim[n] for n in prot if n in lim and lim[n] > 0] or [0.0]),
                active_mae_before=float(np.abs(h0 - ctx.tgt)[m].mean() * 1000), active_mae_after=float(np.abs(h1 - ctx.tgt)[m].mean() * 1000),
                pit_px=int(((dd < -6.0) & ~m).sum()), h0=h0, h1=h1, dd=dd)


def timing(run):
    ss = run["r1"].get("stage_sec") or {}
    t = dict(replay=float(run["summ"].get("replay_sec", 0.0) or 0.0), beam=float(ss.get("beam", 0.0)), refine=float(sum(ss.get(k, 0.0) for k in REFINE_STAGES)),
             pair=float(ss.get("pair", 0.0) + ss.get("planseq", 0.0)))
    rs = run["r1"].get("round_sec")
    t["other"] = max(0.0, float(rs) - t["beam"] - t["refine"] - t["pair"]) if rs else 0.0
    t["total_min"] = run["summ"].get("minutes")
    return t


def validity(run):
    c, kinds, feats = run["cond"], run["summ"].get("cf_kinds") or {}, run["r1"].get("features") or {}
    w = []
    if not run["finished"]:
        w.append("summary.json 없음(실행이 끝나지 않았거나 실패)")
    if run["log_err"]:
        w.append("run.log 에 Traceback")
    if c in ("E1", "E3", "E4", "E5") and not kinds.get("tree"):
        w.append("먼 과거 트리 재생(tree) 기록 없음 -> 이 실행은 E1 효과가 없다")
    if c in ("E2", "E3", "E4", "E5") and not kinds.get("seq"):
        w.append("묶음 탐색(seq) 기록 없음 -> 이 실행은 E2 효과가 없다")
    if c in ("E3", "E3S", "E4", "E5"):
        if "turbo" not in feats:
            w.append("features.turbo 없음(TuRBO 가 돌지 않음)")
        if not feats.get("spsa_evals"):
            w.append("SPSA 평가 0개")
        if run["r1"].get("pair_budget") is None:
            w.append("순서쌍 예산 기록 없음")
    if c == "E4" and not feats.get("sq"):
        w.append("features.sq 없음(짜내기 후보가 만들어지지 않음) -> E3 와 같다")
    if c in ("E5", "D5") and not feats.get("hm"):
        w.append("features.hm 없음(이 영역에 유효한 반사실 힌트가 없어 T 가 만들어지지 않았거나 모델이 없음) -> 힌트가 확정을 이끈 것이 아니다. E3 와 같다")
    if feats.get("bads_fallback"):
        w.append("BADS 대신 패턴 탐색으로 대체: " + str(feats["bads_fallback"])[:60])
    return w


def status_map(path):
    out = {}
    if path and Path(path).exists():
        for l in Path(path).read_text().splitlines():
            m = re.match(r"^(?:abl|heart)_(\w+?)_s(\d+)\s+(.*)$", l)
            if m:
                out[f"{m.group(1)}_s{m.group(2)}"] = m.group(3).strip()
    return out


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def mean_sd(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return (None, None) if not v else (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0)


def plot_curve(by, curves, out):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ymin, ymax = 1e9, -1e9
    for c in CONDS:
        cs = [np.array(curves[r["name"]][2]) for r in by.get(c, []) if curves[r["name"]][2]]
        if not cs:
            continue
        L = min(len(x) for x in cs)
        A = np.array([x[:L] for x in cs])
        for a in A:
            ax.plot(range(L), a, color=COL[c], lw=0.9, alpha=0.28)
        mu = np.nanmean(A, axis=0)
        ax.plot(range(L), mu, color=COL[c], lw=2.0, label=f"{c} (n={len(cs)})")
        ax.plot([k for k in N_MARK if k < L], [mu[k] for k in N_MARK if k < L], "o", color=COL[c], ms=7, mec=SURF, mew=1.5)
        ymin, ymax = min(ymin, np.nanmin(A)), max(ymax, np.nanmax(A))
    starts = [curves[r["name"]][0] for r in sum(by.values(), []) if curves[r["name"]][0] is not None]
    if starts:
        ax.axhline(float(np.mean(starts)), color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.text(0.3, float(np.mean(starts)) + 0.02, "start J", color=MUTED, va="bottom", fontsize=9)
    for k in N_MARK:
        ax.axvline(k, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("refinement evaluations (physics, after the beam stage)")
    ax.set_ylabel("best feasible J so far (lower = better)")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xlim(0, 21.5)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_title("Best J vs. refinement evaluations (bold = mean over seeds, thin = single seeds)", loc="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_time(by, tms, out):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    conds = [c for c in CONDS if by.get(c)]
    keys = ["replay", "beam", "refine", "pair", "other"]
    labels = dict(replay="past replay (before rounds)", beam="beam eval", refine="refinement eval", pair="pair eval", other="other")
    left = np.zeros(len(conds))
    for k in keys:
        v = np.array([np.mean([tms[r["name"]][k] for r in by[c]]) / 60.0 for c in conds])
        ax.barh(range(len(conds)), v, left=left, color=STAGE_COL[k], edgecolor=SURF, linewidth=2, label=labels[k], height=0.62)
        left += v
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.invert_yaxis()
    ax.set_xlabel("minutes (mean over seeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Where the time went (concurrent jobs share the GPU; replay may be cached)", loc="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_damage(by, dmg, out):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
    spec = [("far_max", "max change far from regions (mm)"), ("prot_ratio", "protected-region change / its limit (1 = at limit)"), ("active_gain", "edited region MAE: before - after (mm)")]
    for ax, (k, ttl) in zip(axs, spec):
        for i, c in enumerate([c for c in CONDS if by.get(c)]):
            ys = []
            for r in by[c]:
                d = dmg[r["name"]]
                if not d.get("committed"):
                    continue
                ys.append(d["active_mae_before"] - d["active_mae_after"] if k == "active_gain" else d[k])
            ax.plot([i] * len(ys), ys, "o", color=COL[c], ms=8, mec=SURF, mew=1.5, alpha=0.9)
            if ys:
                ax.plot([i - 0.22, i + 0.22], [np.mean(ys)] * 2, color=COL[c], lw=2.0)
        if k == "prot_ratio":
            ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_xticks(range(len([c for c in CONDS if by.get(c)])))
        ax.set_xticklabels([c for c in CONDS if by.get(c)])
        ax.set_title(ttl, loc="left", fontsize=9, color=INK2)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_qc(cond, runs, dmg, ctx, out):
    runs = [r for r in runs if dmg[r["name"]].get("committed")]
    if not runs:
        return False
    rr, cc_ = np.nonzero(ndimage.binary_dilation(ctx.union, iterations=40))
    sl = (slice(max(rr.min(), 0), rr.max() + 1), slice(max(cc_.min(), 0), cc_.max() + 1))
    fig, axs = plt.subplots(len(runs), 4, figsize=(11.5, 2.9 * len(runs)), squeeze=False)
    for i, r in enumerate(runs):
        d = dmg[r["name"]]
        bg = float(np.median(np.concatenate([d["h0"][0], d["h0"][-1], d["h0"][:, 0], d["h0"][:, -1]])))
        todo = ((ctx.hstar if ctx.hstar is not None else d["h0"]) - d["h1"]) * 1000
        panels = [("start height (mm)", (d["h0"] - bg) * 1000, "turbo", (-10, 14)), ("result height (mm)", (d["h1"] - bg) * 1000, "turbo", (-10, 14)),
                  ("to-do: RED fill / BLUE cut (mm)", todo, "RdBu_r", (-6, 6)), ("damage |change| (mm)", np.abs(d["dd"]), "magma", (0, 8))]
        for j, (ttl, img, cm, (lo, hi)) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(img[sl], cmap=cm, vmin=lo, vmax=hi)
            ax.contour(ctx.masks[ctx.active][sl], levels=[0.5], colors="k", linewidths=1.0)
            for n in ctx.names:
                if n != ctx.active:
                    ax.contour(ctx.masks[n][sl], levels=[0.5], colors="k", linewidths=0.7, linestyles="dashed")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(ttl if i == 0 else "", fontsize=9, color=INK2)
        axs[i, 0].set_ylabel(f"{r['name']}\nfar max {d['far_max']:.1f}mm\nprot x{d['prot_ratio']:.2f}", fontsize=8, color=INK2)
    fig.suptitle(f"{cond}: eyeball check (solid = edited region, dashed = protected regions)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--active", required=True)
    ap.add_argument("--status", type=Path)
    a = ap.parse_args()
    out = a.dir / "report"
    out.mkdir(exist_ok=True)
    ctx = Ctx(a.regions, a.active)
    runs = [r for r in load_runs(a.dir) if r["cond"] in CONDS]
    by = {c: [r for r in runs if r["cond"] == c] for c in CONDS}
    curves = {r["name"]: curve(r) for r in runs}
    dmg = {r["name"]: damage(r, ctx) for r in runs}
    tms = {r["name"]: timing(r) for r in runs}
    st = status_map(a.status)
    L = [f"# ablation 보고서 ({a.dir.name})", "", f"고친 영역 `{a.active}`, 영역 세트 `{a.regions}`. 조건 D, E1, E2, E3 (E3S 스모크는 제외). 색은 기본 팔레트 1~4번을 정해진 순서로 썼다(검증 스크립트는 node 가 없어 못 돌렸다).", ""]
    L += ["## 1. 실행 상태", "", "| 실행 | 상태 | 확정 | 가족 | 시작 J → 확정 후 J | 시간(분) | 경고 |", "|---|---|---|---|---|---|---|"]
    warns = {}
    for r in runs:
        w = validity(r)
        warns[r["name"]] = w
        j0, ja = r["r1"].get("cur_J"), r["r1"].get("J_after")
        L.append(f"| {r['name']} | {st.get(r['name'], '완료' if r['finished'] else '—')} | {'예' if r['committed'] else '아니오'} | {r['r1'].get('family', '—')} | {fmt(j0, 3)} → {fmt(ja, 3)} | {fmt(r['summ'].get('minutes'), 0)} | {'; '.join(w) or '—'} |")
    L += ["", "## 2. 정제 평가 n 개 시점의 최선 J (한도를 지키는 후보만, 낮을수록 좋음)", "", "| 조건 | 시드 수 | 빔 최선 | n=4 | n=8 | n=12 | n=20 | 한도 초과 후보 수(평균) | CNN-물리 순위상관 ρ(MDL) |", "|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for c in CONDS:
        cs = [curves[r["name"]] for r in by[c]]
        if not cs:
            continue
        cols = [mean_sd([x[1] for x in cs])] + [mean_sd([x[2][k] for x in cs if len(x[2]) > k]) for k in N_MARK]
        rho = mean_sd([(r["r1"].get("rho") or {}).get("mdl_rho") for r in by[c]])
        over = mean_sd([x[4] for x in cs])
        agg[c] = dict(beam=cols[0], n4=cols[1], n8=cols[2], n12=cols[3], n20=cols[4], over=over, rho=rho)
        L.append(f"| {c} | {len(cs)} | " + " | ".join(f"{fmt(m, 3)} ± {fmt(s, 3)}" for m, s in cols) + f" | {fmt(over[0], 1)} | {fmt(rho[0], 2)} |")
    L += ["", "![curve](curve.png)", "", "## 3. 시간표 (평균, 초 → 분)", "", "| 조건 | 과거 재생(라운드 전) | 빔 평가 | 정제 평가 | 순서쌍 평가 | 그 밖 | 합계(분, summary) |", "|---|---|---|---|---|---|---|"]
    for c in CONDS:
        if not by[c]:
            continue
        m = {k: np.mean([tms[r["name"]][k] for r in by[c]]) for k in ("replay", "beam", "refine", "pair", "other")}
        tot = mean_sd([tms[r["name"]]["total_min"] for r in by[c]])[0]
        L.append(f"| {c} | " + " | ".join(f"{m[k]:.0f}초 ({m[k] / 60:.1f}분)" for k in ("replay", "beam", "refine", "pair", "other")) + f" | {fmt(tot, 1)} |")
    L += ["", "시드별 과거 재생 시간(캐시 효과: 시드 1 이 길고 2·3 이 짧아야 한다): " + "; ".join(f"{r['name']} {tms[r['name']]['replay']:.0f}초" for r in runs if r["cond"] in ("E1", "E2", "E3", "D")),
          "동시에 2개 작업이 GPU 를 나눠 쓰므로 시간은 조건 간 근사 비교다.", "", "![time](time.png)", "", "## 4. 손상 지표 (확정 한 번의 전후, 영역 세트 합집합에서 25px 밖을 '먼 곳'으로 본다)", "",
          "| 실행 | 먼 곳 최대(mm) | 먼 곳 평균(mm) | 보호 영역 변화/한도 최대 | 보호 영역별 최대 변화(mm) | 고친 영역 MAE 전→후(mm) | 6mm 넘게 파인 화소(영역 밖) |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        d = dmg[r["name"]]
        if not d.get("committed"):
            L.append(f"| {r['name']} | 확정 없음 | | | | | |")
            continue
        L.append(f"| {r['name']} | {d['far_max']:.1f} | {d['far_mean']:.2f} | {d['prot_ratio']:.2f} | " + ", ".join(f"{n} {v:.2f}" for n, v in d["prot"].items()) + f" | {d['active_mae_before']:.2f} → {d['active_mae_after']:.2f} | {d['pit_px']} |")
    L += ["", "![damage](damage.png)", "", "## 5. 독립 물리 재검증 (확정 행동을 새 Genesis 프로세스에서 다시 실행)", "", "| 실행 | 재검증 | 시작 복원 | 최종 재현 오차 | J(보고, 고친 영역) | J(다시 실행, 같은 영역) | 영역 전체 평균 J(참고) |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        v = r["verify"]
        if not v:
            L.append(f"| {r['name']} | 없음 | | | | | |")
            continue
        # 보고 J 는 --own-score 라 마지막 라운드 영역 하나의 점수다 -> 다시 실행한 결과도 '그 영역'의 점수로 비교한다(영역 전체 평균과 비교하면 안 맞는다).
        # verify.json 이 level-adjust 목표를 쓰기 전에 만들어졌으면(level_adjust_targets 키 없음) 원본 목표로 채점된 값이라 * 를 붙인다.
        reg = (v.get("steps") or [{}])[-1].get("region")
        rj = (v.get("region_J") or {}).get(reg, {}).get("after_rerun") if reg else None
        old = "" if "level_adjust_targets" in v else "*"
        L.append(f"| {r['name']} | 있음 | {json.dumps(v.get('restore'), ensure_ascii=False)[:60]} | {json.dumps(v.get('final_repro_vs_final_m'), ensure_ascii=False)[:60]} | {fmt(v.get('J_after_reported'), 3)} | {fmt(rj, 3)}{old} | {fmt(v.get('J_after_rerun'), 3)} |")
    L += ["", "\\* = 조정 목표를 쓰기 전 검증기로 만든 값(원본 목표로 채점). `--level-adjust` 실행에서는 보고 J 와 어긋나 보인다. verify 를 다시 돌리면 사라진다."]
    L += ["", "## 6. 육안 점검 시트", ""]
    for c in CONDS:
        if by[c] and plot_qc(c, by[c], dmg, ctx, out / f"qc_{c}.png"):
            L.append(f"- `qc_{c}.png`: 시작 | 결과 | 해야 할 일(빨강=채움, 파랑=깎음) | 손상")
    L += ["", "## 7. 읽을 때 주의", "", "- 시드가 3개뿐이라 평균±표준편차는 거칠다. 실행 사이 차이(GPU 비결정성)가 조건 차이와 비슷할 수 있다.",
          "- E3 는 D/E1/E2 와 정제 구성(BADS+SPSA, TuRBO)과 순서쌍 예산이 달라서 요소별 효과는 가려 주지 못한다.", "- 유효성 경고가 있는 실행은 그 조건의 효과를 담지 못했을 수 있다(예: E2 인데 seq 기록 없음 = D 와 같다)."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    plot_curve(by, curves, out / "curve.png")
    plot_time(by, tms, out / "time.png")
    plot_damage(by, dmg, out / "damage.png")
    js = dict(runs={r["name"]: dict(curve=curves[r["name"]][2], start=curves[r["name"]][0], timing=tms[r["name"]], warnings=warns[r["name"]],
                                    damage={k: v for k, v in dmg[r["name"]].items() if k not in ("h0", "h1", "dd")}) for r in runs}, aggregate=agg)
    (out / "report.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float))
    print(f"보고서: {out}/report.md  (실행 {len(runs)}개, 경고 {sum(1 for w in warns.values() if w)}개 실행)")


if __name__ == "__main__":
    main()
