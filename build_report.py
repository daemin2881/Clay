"""보고서 HTML 생성기. results/case_A/design 의 결과 파일을 읽어 표와 그림을 채운다 (숫자를 손으로 옮기지 않는다).
사용: python build_report.py   ->  results/case_A/design/report.html
"""
import base64
import re
import csv
import json
from pathlib import Path

import numpy as np

from visual_score import VisualScorer

ROOT = Path(__file__).resolve().parent
D = ROOT / "results/case_A/design"
HIST = ROOT / "results/A_history200_20261114_060837/history"


def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def cls(v, good_lt=1.0):
    return "good" if v < good_lt else ("bad" if v > 1.0 else "")


def row_arm(name, r, note=""):
    return (f"<tr><th scope='row'>{name}</th>"
            f"<td class='{cls(r['l3d_win'])}'>{r['l3d_win']:.2f}</td><td class='{cls(r['lviz_win'])}'>{r['lviz_win']:.2f}</td>"
            f"<td class='{cls(r['edge_win'])}'>{r['edge_win']:.2f}</td>"
            f"<td>{r['rms_win_mm']:.2f}</td><td>{r['grad_err_pct']:.0f}%</td><td>{r['deepen_mm']:+.2f}</td>"
            f"<td>{r['outside_win_max_mm']:.1f}</td><td class='{cls(r['lviz_full'])}'>{r['lviz_full']:.1f}</td><td class='note'>{note}</td></tr>")


def arm_metrics(sc, h_start, h, hs_mm, gref, gx, gy, win):
    from visual_score import _grad
    r = sc.summary(h, h_start)
    r["edge_win"] = sc.ratios(h)["edge_win"]
    r.update({k: v for k, v in sc.ratios(h).items() if k in ("l3d_win", "lviz_win", "l3d_full", "lviz_full")})
    hm = h.astype(np.float64) * 1000
    r["rms_win_mm"] = float(np.sqrt(((hm - hs_mm) ** 2)[win].mean()))
    g2 = _grad(hm)
    r["grad_err_pct"] = float(100 * np.sqrt(((g2[0] - gx) ** 2 + (g2[1] - gy) ** 2)[win[1:-1, 1:-1]].mean()) / gref)
    return r


def main():
    from visual_score import _grad
    spec = D / "spec_d3_w1.0"
    h_star = np.load(spec / "h_star.npy")
    me = np.load(spec / "m_edit.npy") > 0
    last = len([p for p in HIST.glob("action_??????") if p.is_dir()])
    h_t = np.load(HIST / f"action_{last:06d}" / "transition.npz")["after_m"].astype(np.float32)
    sc = VisualScorer(h_star, h_t, me)
    hs_mm = h_star.astype(np.float64) * 1000
    gx, gy = _grad(hs_mm)
    gref = float(np.sqrt((gx ** 2 + gy ** 2)[sc.win[1:-1, 1:-1]].mean()))

    def M(h):
        return arm_metrics(sc, h_t, h, hs_mm, gref, gx, gy, sc.win)

    arms = [
        ("아무것도 안 함 (기준)", M(h_t), "모든 비율의 기준 = 1.00"),
        ("Visual Sculpting 재계획 (10행동)", M(np.load(D / "arm_baseline_replan/final_m.npy")), "과거 기록을 쓰지 않음. 새 target H*만 받아 현재에서 다시 계획"),
        ("설계: 현재 상태 보정 (10행동, Broyden 끔)", M(np.load(D / "lc_w1.0_nobroyden/final_m.npy")), "과거 기록 없음. 모델 기반 beam + 자코비안 + Adam"),
        ("설계: 현재 상태 보정 (10행동, Broyden 켬)", M(np.load(D / "lc_w1.0_broyden_v2/final_m.npy")), "위와 같고 모델 오차를 Broyden으로 갱신"),
    ]
    arm_rows = "".join(row_arm(n, r, note) for n, r, note in arms)

    # 과거 롤백(반사실) 결과
    sweep_rows, sweep_n = [], 0
    for start in (183, 170, 164, 29):
        d = D / f"rollback{start}"
        rep_f = d / "rollback_report.json"
        if not (d / "H_control.npy").exists():
            continue
        import subprocess
        subprocess.run(["python", str(ROOT / "rollback_cf.py"), "report", "--history-dir", str(HIST), "--start", str(start),
                        "--spec-dir", str(spec), "--out", str(d)], capture_output=True, cwd=ROOT)
        rep = json.loads(rep_f.read_text())
        ctrl = rep["control"]
        h_ctrl = np.load(d / "H_control.npy")
        for r in rep["runs"]:
            h_cf = np.load(d / f"H_{r['name']}.npy")
            m = M((h_t + (h_cf - h_ctrl)).astype(np.float32))
            sweep_n += 1
            gate = "통과" if ctrl["passed"] else "불통과"
            verdict = "개선 없음" if r["q_genesis"] <= 0 else "개선"
            sweep_rows.append(
                f"<tr><th scope='row'>#{start}</th><td>{rep['n_replayed']}</td>"
                f"<td class='{'good' if ctrl['passed'] else 'bad'}'>{gate} ({ctrl['bad_pixels']}화소, 최대 {ctrl['drift_max_mm']:.2f}mm)</td>"
                f"<td>길이 ×{r['lf']:g}, 깊이 ×{r['df']:g}</td><td class='{'good' if r['q_genesis'] > 0 else 'bad'}'>{r['q_genesis']:+.2f}</td>"
                f"<td class='{cls(m['l3d_win'])}'>{m['l3d_win']:.2f}</td><td class='{cls(m['lviz_win'])}'>{m['lviz_win']:.2f}</td>"
                f"<td class='{cls(m['edge_win'])}'>{m['edge_win']:.2f}</td><td>{m['deepen_mm']:+.2f}</td><td>{verdict}</td></tr>")

    # 모델 예측 이득 vs 실제 이득
    pred_real = []
    for key in ("lc_w1.0_nobroyden", "lc_w1.0_broyden_v2"):
        rows = list(csv.DictReader(open(D / key / "steps.csv")))[1:]
        pred_real += [(float(x["pred_gain"]), float(x["real_gain"])) for x in rows]
    n_bad = sum(1 for p, r in pred_real if r < 0)
    corr = float(np.corrcoef([p for p, _ in pred_real], [r for _, r in pred_real])[0, 1])

    # 영역 반사실 탐색(cf_search.py) 로그 파싱
    cf_rows = []
    pat = re.compile(r"^\s+(\S+): 영역 창 RMS ([\d.]+) -> ([\d.]+)mm \| 영역 안 목표 쪽 이동 ([+-][\d.]+) \| S×([\d.]+) \(L3D ([\d.]+) Lviz ([\d.]+) 테두리 ([\d.]+)\) \| 창 밖 최대 ([\d.]+)mm \| (개선 없음|개선)")
    for log in sorted(list((D / "cf_search_gap").glob("proc*.log")) + list((D / "cf_search_bump").glob("proc*.log")) + list((D / "cf_search_groove").glob("proc*.log"))):
        start = None
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m0 = re.match(r"^\[#(\d+)\] control 재생 (\d+)행동 \d+초: .*?(통과|불통과)", line)
            if m0:
                start, nrep, gate = m0.group(1), m0.group(2), m0.group(3)
                continue
            m = pat.match(line)
            if m and start:
                v, r0_, r1_, tw, S, l3, lv, ed, far, _strict = m.groups()
                twf = float(tw)
                verdict = ("영역 개선 (둘레 변화 %smm)" % far) if twf >= 0.3 else ("영역 변화 미미" if twf > 0 else "영역 악화")
                cf_rows.append(f"<tr><th scope='row'>#{start}</th><td>{nrep}</td><td class='{'good' if gate == '통과' else 'bad'}'>{gate}</td><td>{v}</td>"
                               f"<td>{r0_} → {r1_}</td><td class='{'good' if twf >= 0.3 else ('bad' if twf <= 0 else '')}'>{tw}</td>"
                               f"<td class='{cls(float(l3))}'>{l3}</td><td class='{cls(float(lv))}'>{lv}</td><td class='{cls(float(ed))}'>{ed}</td><td>{far}</td><td>{verdict}</td></tr>")
    # 최종 비교: 190회 A vs 과거(#186)를 고쳐 재생한 A
    from region_edit import _bg
    rev = np.load(D / "cf_combined_186/combined.npy").astype(np.float64)
    ref = np.load(HIST / "target_used_512_m.npy").astype(np.float64)
    final_rows = []
    for i, nm in enumerate(("왼쪽 튀어나온 부분", "끊어진 왼쪽 다리", "가운데 홈")):
        mk = np.load(D / f"cf_specs/r{i}/m_edit.npy") > 0
        need = float(((ref - _bg(ref) + _bg(h_t.astype(np.float64)) - h_t)[mk]).mean()) * 1000
        got = float(((rev - h_t)[mk]).mean()) * 1000
        frac = got / need if abs(need) > 1e-6 else 0.0
        good = (need * got > 0) and abs(got) > 0.3
        final_rows.append(f"<tr><th scope='row'>{nm}</th><td>{need:+.2f}</td><td class='{'good' if good else ''}'>{got:+.2f}</td><td class='{'good' if good else ''}'>{frac * 100:+.0f}%</td></tr>")
    n_changed = int((np.abs(rev - h_t) * 1000 > 0.5).sum())
    max_changed = float(np.abs(rev - h_t).max()) * 1000
    # 80번째 상태 비교
    s80_rows, s80_inter, fig_s80 = "", "", ""
    if (D / "s80_compare.json").exists():
        sj = json.loads((D / "s80_compare.json").read_text())
        s80_inter = sj.pop("_interaction", "")
        for n, m in sj.items():
            s80_rows += (f"<tr><th scope='row'>{n}</th><td class='{'good' if m['toward'] > 0.05 else ('bad' if m['toward'] < -0.05 else '')}'>{m['toward']:+.2f}</td>"
                         f"<td>{m['mae_all']:.3f}</td><td>{m['mae_A']:.3f}</td><td>{m['mae_regions'][0]:.2f} / {m['mae_regions'][1]:.2f} / {m['mae_regions'][2]:.2f}</td>"
                         f"<td class='{cls(m['l3d'])}'>{m['l3d']:.2f}</td><td class='{cls(m['lviz'])}'>{m['lviz']:.2f}</td><td class='{cls(m['edge'])}'>{m['edge']:.2f}</td>"
                         f"<td>{m['n_changed']}</td><td>{m['far_max']:.1f}</td></tr>")
        fig_s80 = b64(D / "figs/s80_compare.png")
    # 전/후 이미지, 복원 행동 결과
    ba = D / "figs/before_after"
    fig_before, fig_after, fig_diff, fig_part = (b64(ba / n) for n in ("before_heightmap.png", "after_heightmap.png", "diff_heightmap.png", "particles_before_after.png"))
    undo_rows, undo_sel, fig_undo = "", "", ""
    uj = D / "undo186/undo_candidates.json"
    if uj.exists():
        u = json.loads(uj.read_text())
        for r in sorted(u["rows"], key=lambda r: -r["toward_cf"]):
            is_best = abs(r["toward_cf"] - u["best"]["toward_cf"]) < 1e-12
            undo_rows += (f"<tr><th scope='row'>깊이 ×{r['depth_fac']:g}, 길이 ×{r['length_fac']:g}, 옆 {r['offset_mm']:+.0f}mm{' (선택)' if is_best else ''}</th>"
                          f"<td class='{'good' if r['toward_cf'] > 0.05 else ('bad' if r['toward_cf'] < -0.05 else '')}'>{r['toward_cf']:+.2f}</td><td>{r['mae_cf']:.2f}</td><td>{r['mae_target']:.2f}</td><td>{r['far_max']:.1f}</td></tr>")
        bst = u["best"]
        undo_sel = (f"선택한 복원 행동: 깊이 ×{bst['depth_fac']:g}, 길이 ×{bst['length_fac']:g}, 옆 {bst['offset_mm']:+.0f}mm. 영역 MAE(수정된 과거 대비) {u['base']['mae_cf']:.2f} → {bst['mae_cf']:.2f}mm, "
                    f"영역 MAE(원래 목표 대비) {u['base']['mae_target']:.2f} → {bst['mae_target']:.2f}mm, 영역 밖 최대 변화 {bst['far_max']:.1f}mm.")
        if (D / "figs/undo_compare.png").exists():
            fig_undo = b64(D / "figs/undo_compare.png")
    # 80번째 요청 사항별 결과
    KO = {"groove": "A 사이의 홈", "left_leg": "왼쪽 다리", "right_leg": "오른쪽 다리", "left_foot": "왼쪽 다리 끝 받침", "right_foot": "오른쪽 다리 끝 받침"}
    NM = list(KO)
    pe_path = D / "s80b/past_effects.json"
    s80b_matrix, s80b_best, s80b_comp = "", "", ""
    if pe_path.exists():
        pe = json.loads(pe_path.read_text())
        base, need = pe.pop("_base"), pe.pop("_need")
        s80b_best = "".join(f"<tr><th scope='row'>{KO[n]}</th><td>{base[n]:.2f}</td><td>{need[n]:+.2f}</td></tr>" for n in NM)
        for k, v in pe.items():
            cells = ""
            for n in NM:
                dlt = v[n] - base[n]
                cells += f"<td class='{'good' if dlt < -0.05 else ('bad' if dlt > 0.05 else '')}'>{v[n]:.2f} ({dlt:+.2f})</td>"
            s80b_matrix += f"<tr><th scope='row'>{k}</th>{cells}</tr>"
    comp_html = ""
    for name, title in (("comp_s80_right_leg", "오른쪽 다리 (과거 #73 건너뛰기의 효과를 보상 목표로)"), ("comp_s80_groove", "A 사이의 홈 (과거 #76 건너뛰기의 효과를 보상 목표로)")):
        cd = D / name
        if not (cd / "run.log").exists():
            continue
        txt = (cd / "run.log").read_text(encoding="utf-8", errors="ignore").splitlines()
        start_l = next((l for l in txt if l.startswith("시작:")), "")
        done_l = next((l for l in txt if l.startswith("완료:")), "")
        steps = [l.strip() for l in txt if l.strip().startswith("확정:") or l.strip().startswith("개선되는")]
        frames = sorted((cd / "frames").glob("frame_*.png")) if (cd / "frames").exists() else []
        imgs = "".join(f"<figure><img alt='{f.name}' src='{b64(f)}'><figcaption>{f.name}</figcaption></figure>" for f in ([frames[0], frames[-1]] if len(frames) > 1 else frames[:1]))
        comp_html += (f"<h3>{title}</h3><p>{start_l}</p><ul>" + "".join(f"<li>{x}</li>" for x in steps) + f"</ul><p>{done_l or '실행 중이거나 종료되지 않았습니다.'}</p>" + imgs)
    s80b_comp = comp_html
    probe = json.load(open(D / "probe_stamp.json"))
    n_probe = len(probe)
    best_probe = min(probe, key=lambda r: r["l3d_win"] + r["lviz_win"])

    html = TEMPLATE
    rep_vals = dict(
        CF_ROWS="".join(cf_rows), CF_N=str(len(cf_rows)),
        FIG_DEV=b64(D / "figs/deviation_from_target.png"), FIG_CF=b64(D / "figs/cf186_masked.png"),
        FINAL_ROWS="".join(final_rows), FIG_FINAL=b64(D / "figs/final_vs_revised.png"), N_CHANGED=str(n_changed), MAX_CHANGED=f"{max_changed:.0f}",
        S80_ROWS=s80_rows, S80_INTER=s80_inter, FIG_S80=fig_s80,
        FIG_BA_BEFORE=fig_before, FIG_BA_AFTER=fig_after, FIG_BA_DIFF=fig_diff, FIG_PART=fig_part,
        UNDO_ROWS=undo_rows, UNDO_SEL=undo_sel, FIG_UNDO=fig_undo,
        FIG_S80_RL=b64(D / "figs/s80_ba_right_leg.png"), FIG_S80_GR=b64(D / "figs/s80_ba_groove.png"),
        S80B_BEST=s80b_best, S80B_MATRIX=s80b_matrix, S80B_COMP=s80b_comp,
        ARM_ROWS=arm_rows, SWEEP_ROWS="".join(sweep_rows), SWEEP_N=str(sweep_n),
        FIG_SHADED=b64(D / "figs/shaded_compare.png"), FIG_SAM=b64(D / "figs/sam_process.png"), FIG_METRICS=b64(D / "figs/visual_compare.png"),
        PRED_N=str(len(pred_real)), PRED_BAD=str(n_bad), PRED_CORR=f"{corr:+.2f}",
        PROBE_N=str(n_probe), PROBE_BEST_L3D=f"{best_probe['l3d_win']:.2f}", PROBE_BEST_LVIZ=f"{best_probe['lviz_win']:.2f}",
        PROBE_BEST_DEEP=f"{best_probe['deepen_mm']:.2f}",
        BASE_RMS=f"{arms[0][1]['rms_win_mm']:.2f}", BASE_GRAD=f"{arms[0][1]['grad_err_pct']:.0f}",
        DES_RMS=f"{arms[2][1]['rms_win_mm']:.1f}", DES_GRAD=f"{arms[2][1]['grad_err_pct']:.0f}",
    )
    for k, v in rep_vals.items():
        html = html.replace("{{" + k + "}}", v)
    out = D / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"저장: {out} ({out.stat().st_size / 1e6:.1f} MB), 롤백 행 {sweep_n}개")


TEMPLATE = open(Path(__file__).with_name("report_template.html"), encoding="utf-8").read()

if __name__ == "__main__":
    main()
