"""영역 하나에 대한 과거 반사실 탐색: "과거에 이 행동을 이렇게 했더라면 지금 이렇게 됐을 것이다"를 Genesis 로 재생해서 확인한다.

후보 행동 t 마다
  1) S_{t-1} 로 롤백, 원래 행동 그대로 끝까지 재생 -> H_ctrl (같은 엔진에서 다시 돌린 값. 기록과 맞는지 확인)
  2) a_t 의 길이·깊이만 바꾼 변형들(얕게, 짧게, 건너뛰기 등)을 끝까지 재생 -> H_cf
  3) 목표 H* 에 얼마나 가까워졌나를 눈에 보이는 지표로 비교 (H_ctrl 이 기준 = 1.00)
       영역 창 RMS 오차(mm), 영역 안에서 목표 쪽으로 간 정도(1.0 = 목표 도달), 형태 L3D / 외형 Lviz / 테두리 / 전체 Lviz 비율, 창 밖 최대 변화
  4) C* = H_cf - H_ctrl 을 저장해서 현재 상태 보정(local_correct.py --cstar)에 쓴다. 상태 스냅샷도 저장하면 view_state.py 로 '과거를 고친 뒤의 점토'를 볼 수 있다.
사용: python cf_search.py --history-dir H --spec-dir REGION_SPEC --starts 186,179 --out DIR [--save-state]
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import ndimage


def parse_variants(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "skip":
            out.append(("skip", None, None))
        else:
            lf, df = (float(x) for x in tok.split(":"))
            out.append((f"l{lf:g}_d{df:g}", lf, df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--spec-dir", required=True, type=Path, help="영역 하나의 spec (m_edit = 그 영역, h_star = 그 영역의 목표)")
    ap.add_argument("--starts", default="", help="후보 행동 번호들. 예: 186,179")
    ap.add_argument("--combine", help="여러 과거 수정을 한 번에 적용해 재생. 예: 160:skip,186:skip (행동번호:skip 또는 행동번호:길이배율:깊이배율)")
    ap.add_argument("--region-specs", help="영역별 spec 폴더들(쉼표). 합친 재생의 영역별 개선을 따로 보고한다")
    ap.add_argument("--variants", default="1:0.3,1:0.6,0.6:1,1.3:1.3,skip", help="길이배율:깊이배율 목록, skip = 거의 아무것도 안 하는 행동으로 교체")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--save-state", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="Genesis 뷰어 창에서 재생을 실시간으로 본다(화면 필요)")
    ap.add_argument("--live-only", action="store_true", help="--combine 에서 control 재생을 건너뛰고 수정한 과거만 재생한다. 비교 기준은 기록된 최종 상태")
    ap.add_argument("--max-drift-mm", type=float, default=0.5)
    a = ap.parse_args()

    root = a.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if a.viewer else "0"
    import torch
    import vs_simulator as sim
    import vs_config as cfg
    import counterfactual_replay as cfr
    from vs_history import GenesisStateAdapter, physics_signature, tensor_save
    from visual_score import VisualScorer

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")
    last = len([d for d in root.glob("action_??????") if d.is_dir()])
    h_rec = np.load(root / f"action_{last:06d}" / "transition.npz")["after_m"].astype(np.float32)
    h_star = np.load(a.spec_dir / "h_star.npy")
    me = np.load(a.spec_dir / "m_edit.npy") > 0
    win = ndimage.binary_dilation(me, iterations=25)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(a.variants)
    present = adapter.capture()
    report = []
    try:
        if a.combine:
            from vs_history import tensor_save as _ts
            mods = []
            for tok in a.combine.split(","):
                parts = tok.split(":")
                mods.append((int(parts[0]), "skip" if parts[1] == "skip" else (float(parts[1]), float(parts[2]))))
            start = min(n for n, _ in mods)
            pre, records, rngs = cfr.load_plan(root, start)
            if a.live_only:
                h_ctrl = h_rec.copy()            # 기록된 최종 상태(= 190회 A)와 비교
            else:
                h_ctrl, _ = cfr.run_replay(adapter, pre, records, rngs, start, last, {})
            edits = {}
            for n, v in mods:
                act = torch.tensor(records[n]["action"], dtype=torch.float64)
                if v == "skip":
                    act[3], act[4] = cfg.LENGTH_MIN, cfg.DEPTH_MIN
                else:
                    act[3] = float(np.clip(act[3] * v[0], cfg.LENGTH_MIN, cfg.LENGTH_MAX)); act[4] = float(np.clip(act[4] * v[1], cfg.DEPTH_MIN, cfg.DEPTH_MAX))
                edits[n] = act.float()
            h_cf, sec = cfr.run_replay(adapter, pre, records, rngs, start, last, edits)
            _ts(out / "combined_state.pt", adapter.capture())
            np.save(out / "combined_control.npy", h_ctrl.astype(np.float32)); np.save(out / "combined.npy", h_cf.astype(np.float32))
            print(f"[합친 재생] 과거 수정 {mods} / 행동 {start}~{last} ({last - start + 1}행동) {sec:.0f}초", flush=True)
            for rs in (a.region_specs.split(",") if a.region_specs else []):
                rs = Path(rs); hs_r = np.load(rs / "h_star.npy"); m_r = np.load(rs / "m_edit.npy") > 0
                w_r = ndimage.binary_dilation(m_r, iterations=25)
                rm = lambda h: float(np.sqrt((((h.astype(np.float64) - hs_r) * 1000) ** 2)[w_r].mean()))
                den = float(((hs_r - h_ctrl)[m_r] ** 2).sum()) + 1e-12
                tw = float(((h_cf - h_ctrl)[m_r] * (hs_r - h_ctrl)[m_r]).sum() / den)
                print(f"   영역 {rs.name}: 창 RMS {rm(h_ctrl):.2f} -> {rm(h_cf):.2f}mm, 영역 안 목표 쪽 이동 {tw:+.2f}, 영역 안 평균 변화 {float(((h_cf - h_ctrl)[m_r]).mean() * 1000):+.2f}mm", flush=True)
            return
        for start in (int(x) for x in a.starts.split(",")):
            pre, records, rngs = cfr.load_plan(root, start)
            n_replay = last - start + 1
            t0 = time.perf_counter()
            h_ctrl, _ = cfr.run_replay(adapter, pre, records, rngs, start, last, {})
            dd = np.abs(h_ctrl - h_rec) * 1000
            bad, p999 = int((dd > a.max_drift_mm).sum()), float(np.quantile(dd, 0.999))
            gate = bad <= 5 and p999 <= 0.2
            np.save(out / f"s{start}_control.npy", h_ctrl.astype(np.float32))
            sc = VisualScorer(h_star, h_ctrl, me)
            toward_den = float(((h_star - h_ctrl)[me] ** 2).sum()) + 1e-12
            rms = lambda h: float(np.sqrt((((h.astype(np.float64) - h_star) * 1000) ** 2)[win].mean()))
            print(f"[#{start}] control 재생 {n_replay}행동 {time.perf_counter() - t0:.0f}초: 기록 대비 0.5mm 초과 {bad}화소, p99.9 {p999:.3f}mm -> "
                  f"{'통과' if gate else '불통과(이 후보는 신뢰 낮음)'} | 영역 창 RMS(목표 대비) {rms(h_ctrl):.2f}mm", flush=True)
            for name, lf, df in variants:
                act = torch.tensor(records[start]["action"], dtype=torch.float64)
                if name == "skip":
                    act[3], act[4] = cfg.LENGTH_MIN, cfg.DEPTH_MIN
                else:
                    act[3] = float(np.clip(act[3] * lf, cfg.LENGTH_MIN, cfg.LENGTH_MAX))
                    act[4] = float(np.clip(act[4] * df, cfg.DEPTH_MIN, cfg.DEPTH_MAX))
                h_cf, sec = cfr.run_replay(adapter, pre, records, rngs, start, last, {start: act.float()})
                if a.save_state:
                    tensor_save(out / f"s{start}_{name}_state.pt", adapter.capture())
                np.save(out / f"s{start}_{name}.npy", h_cf.astype(np.float32))
                r = sc.ratios(h_cf)
                S = float((r["l3d_win"] + r["lviz_win"] + r["edge_win"] + r["lviz_full"]) / 4.0)
                toward = float(((h_cf - h_ctrl)[me] * (h_star - h_ctrl)[me]).sum() / toward_den)
                far = float(np.abs(h_cf - h_ctrl)[~win].max()) * 1000
                row = dict(start=start, variant=name, action_before=records[start]["action"], action_after=[float(v) for v in act],
                           n_replay=n_replay, control_gate=gate, rms_before=rms(h_ctrl), rms_after=rms(h_cf), toward_target=toward,
                           S=S, l3d_win=r["l3d_win"], lviz_win=r["lviz_win"], edge_win=r["edge_win"], lviz_full=r["lviz_full"],
                           far_change_mm=far, seconds=sec, improved=bool(S < 1.0 and rms(h_cf) < rms(h_ctrl)))
                report.append(row)
                print(f"   {name:>10}: 영역 창 RMS {row['rms_before']:.2f} -> {row['rms_after']:.2f}mm | 영역 안 목표 쪽 이동 {toward:+.2f} | "
                      f"S×{S:.3f} (L3D {r['l3d_win']:.2f} Lviz {r['lviz_win']:.2f} 테두리 {r['edge_win']:.2f}) | 창 밖 최대 {far:.2f}mm | "
                      f"{'개선' if row['improved'] else '개선 없음'}", flush=True)
            (out / f"result_{a.starts.replace(',', '_')}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    finally:
        adapter.restore(present)
    good = [r for r in report if r["improved"] and r["control_gate"]]
    print("\n요약: 개선 + 재생 신뢰 통과 =", [(r["start"], r["variant"], round(r["S"], 3)) for r in sorted(good, key=lambda r: r["S"])] or "없음")


if __name__ == "__main__":
    main()
