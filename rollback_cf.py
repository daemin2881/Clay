"""설계 §5 + 8~9단계: 후보 행동 k 직전 상태 S_{k-1} 로 롤백해서 a_k 의 length/depth 만 바꾸고 나머지는 그대로 재생.

  tau_F  = (S_{k-1}, a_k,  a_{k+1}, ..., a_T)     -> H^F_T   (control: 같은 엔진에서 다시 돌린 값)
  tau_CF = (S_{k-1}, a'_k, a_{k+1}, ..., a_T)     -> H^CF_T
  C_k = H^CF_T - H^F_T,   q_k = L(H^F_T) - L(H^CF_T)

두 단계로 쓴다(변형 하나가 프로세스 하나라서 병렬로 돌릴 수 있다).
  재생:  python rollback_cf.py run --history-dir H --start k --variant control=1:1 --variant cf=1.3:1.3 --out DIR
  판정:  python rollback_cf.py report --history-dir H --start k --spec-dir S --out DIR
유효(valid) 조건: control 재생이 기록과 맞고(robust gate), 효과/재생잡음 SNR>=min-snr, q>0. 하나라도 어긋나면 abstain.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def parse_variant(s):
    name, ratio = s.split("=")
    lf, df = (float(x) for x in ratio.split(":"))
    return name, lf, df


def cmd_run(a):
    root = a.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["static_files"].items():
        from vs_history import sha256
        if Path(name).name != name or sha256(root / name) != expected:
            raise RuntimeError(f"고정 목표/마스크 파일이 변경됐습니다: {name}")
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0"
    import torch
    import vs_simulator as sim
    import vs_config as cfg
    import counterfactual_replay as cfr
    from vs_history import GenesisStateAdapter, physics_signature

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")
    last = len([d for d in root.glob("action_??????") if d.is_dir()])
    k = a.start
    if not 1 <= k <= last:
        raise ValueError(f"--start 는 1~{last}")
    pre, records, rngs = cfr.load_plan(root, k)
    out = a.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    present = adapter.capture()
    try:
        for spec in a.variant:
            name, lf, df = parse_variant(spec)
            act = torch.tensor(records[k]["action"], dtype=torch.float64)
            act[3] = float(np.clip(act[3] * lf, cfg.LENGTH_MIN, cfg.LENGTH_MAX))
            act[4] = float(np.clip(act[4] * df, cfg.DEPTH_MIN, cfg.DEPTH_MAX))
            edits = {} if (lf == 1.0 and df == 1.0) else {k: act.float()}
            h, sec = cfr.run_replay(adapter, pre, records, rngs, k, last, edits)
            np.save(out / f"H_{name}.npy", h.astype(np.float32))
            (out / f"H_{name}.json").write_text(json.dumps(dict(
                name=name, start=k, last=last, lf=lf, df=df, seconds=sec,
                action_before=records[k]["action"], action_after=[float(v) for v in act])))
            print(f"[{name}] 행동 {k}~{last} 재생 {sec:.0f}초, l×{lf:g} d×{df:g}", flush=True)
    finally:
        adapter.restore(present)


def cmd_report(a):
    import torch
    from edit_spec import load_spec
    root = a.history_dir.expanduser().resolve()
    out = a.out.expanduser().resolve()
    k = a.start
    last = len([d for d in root.glob("action_??????") if d.is_dir()])
    spec = load_spec(a.spec_dir)
    me = spec.m_edit[0, 0].numpy()
    h_rec = np.load(root / f"action_{last:06d}" / "transition.npz")["after_m"].astype(np.float32)
    h_ctrl = np.load(out / "H_control.npy")
    dd = np.abs(h_ctrl - h_rec) * 1000
    bad, p999 = int((dd > a.max_drift_mm).sum()), float(np.quantile(dd, 0.999))
    control_ok = bad <= a.max_bad_pixels and p999 <= a.max_p999_mm
    L = lambda h: float(spec.loss(torch.from_numpy(h)[None, None]))
    L_ctrl = L(h_ctrl)
    rep = out / "H_replicate.npy"
    noise = None
    if rep.exists():
        d = np.abs(np.load(rep) - h_ctrl) * 1000
        noise = dict(edit_mae_mm=float(d[me].mean()), global_max_mm=float(d.max()))
    report = dict(schema=1, start=k, last=last, n_replayed=last - k + 1,
                  control=dict(drift_max_mm=float(dd.max()), bad_pixels=bad, p999_mm=p999, passed=control_ok,
                               L=L_ctrl, L_recorded=L(h_rec)),
                  noise=noise, min_snr=a.min_snr, runs=[])
    zone = me | spec.m_boundary[0, 0].numpy()
    for f in sorted(out.glob("H_*.npy")):
        name = f.stem[2:]
        if name in ("control", "replicate"):
            continue
        meta = json.loads((out / f"H_{name}.json").read_text())
        h = np.load(f)
        C = (h - h_ctrl)
        q = L_ctrl - L(h)
        eff = float(np.abs(C)[me].mean()) * 1000
        snr = None if noise is None else eff / max(noise["edit_mae_mm"], 1e-6)
        valid = bool(control_ok and q > 0 and (snr is None or snr >= a.min_snr))
        report["runs"].append(dict(name=name, lf=meta["lf"], df=meta["df"], q_genesis=q, L=L(h),
                                   effect_edit_mm=eff, snr=snr, valid=valid,
                                   outside_zone_max_mm=float(np.abs(C)[~zone].max()) * 1000,
                                   outside_zone_mean_mm=float(np.abs(C)[~zone].mean()) * 1000,
                                   seconds=meta["seconds"]))
    ok = [r for r in report["runs"] if r["valid"]]
    report["abstain"] = not ok
    if ok:
        best = max(ok, key=lambda r: r["q_genesis"])
        report["best"] = best["name"]
        np.save(out / "c_star_512_m.npy", (np.load(out / f"H_{best['name']}.npy") - h_ctrl).astype(np.float32))
    (out / "rollback_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"control: 기록 대비 최대 {dd.max():.3f}mm, {a.max_drift_mm}mm 초과 {bad}픽셀, p99.9={p999:.4f}mm -> "
          f"{'통과' if control_ok else '불통과 (이 구간 재생은 신뢰 기준 미달)'} | L(H^F)={L_ctrl:.4f}")
    if noise:
        print(f"재생 잡음(편집영역 MAE) {noise['edit_mae_mm']:.4f}mm")
    for r in report["runs"]:
        print(f"[{r['name']}] l×{r['lf']:g} d×{r['df']:g}: q(Genesis)={r['q_genesis']:+.4f}, 편집영역 효과 {r['effect_edit_mm']:.3f}mm, "
              f"SNR={r['snr'] if r['snr'] is None else round(r['snr'], 1)}, 영역 밖 최대 변화 {r['outside_zone_max_mm']:.2f}mm -> "
              f"{'valid' if r['valid'] else 'INVALID'}")
    print("=> abstain: 유효한 C* 없음 (현재 상태 보정만 사용)" if report["abstain"] else f"=> C* 저장: {out}/c_star_512_m.npy ({report['best']})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--history-dir", required=True, type=Path)
    r.add_argument("--start", required=True, type=int)
    r.add_argument("--variant", action="append", required=True, help="NAME=lf:df  (control=1:1)")
    r.add_argument("--out", required=True, type=Path)
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("--history-dir", required=True, type=Path)
    q.add_argument("--start", required=True, type=int)
    q.add_argument("--spec-dir", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q.add_argument("--max-drift-mm", type=float, default=0.5)
    q.add_argument("--max-bad-pixels", type=int, default=5)
    q.add_argument("--max-p999-mm", type=float, default=0.2)
    q.add_argument("--min-snr", type=float, default=5.0)
    q.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
