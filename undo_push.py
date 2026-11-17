"""'과거 행동 #k 를 누르지 않았으면 좋았다'를 현재 상태에서 복원하는 행동을 계산하고 시뮬레이터에서 실행한다.

반사실 재생(cf_search.py)이 '#k 를 안 눌렀을 때의 현재 모습' H_cf 를 준다. 현재 A(190회)에서 그 모습에 가까워지도록 되돌리는 행동을 찾는다.
  후보 : #k 가 점토를 밀어낸 경로의 역방향 푸시 (끝점에서 시작점 쪽으로). 길이 x{1,2}, 깊이 x{0.5,1}, 옆으로 +-4mm 이동의 조합.
  평가 : 후보를 Genesis 에서 실제로 실행해 보고 매번 롤백한다. 영역 안에서 H_cf 쪽으로 간 정도 t, 영역 MAE(H_cf 대비), 영역 밖 변화를 잰다.
  선택 : 영역 밖 최대 변화가 --far-limit 이하인 후보 중 t 가 가장 큰 것. 없으면 t 최대. --commit 이면 그것을 실제로 실행하고 결과를 저장한다.
사용: python undo_push.py --history-dir H --action 186 --target-npy H_cf.npy --region-spec SPEC --out DIR --commit [--viewer]
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--action", required=True, type=int)
    ap.add_argument("--target-npy", required=True, type=Path, help="과거를 고쳐 재생한 최종 높이맵 H_cf (512x512, m)")
    ap.add_argument("--region-spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--far-limit-mm", type=float, default=6.0)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()
    root = a.history_dir.resolve()
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if a.viewer else "0"
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature, tensor_save

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != man["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")
    last = len([d for d in root.glob("action_??????") if d.is_dir()])
    adapter.restore(load_snapshot(root / f"action_{last:06d}" / "post_state.pt"))
    start = adapter.observe().numpy().astype(np.float64)
    saved = adapter.capture()
    hcf = np.load(a.target_npy).astype(np.float64)
    hstar = np.load(a.region_spec / "h_star.npy").astype(np.float64)
    m = np.load(a.region_spec / "m_edit.npy") > 0
    win = ndimage.binary_dilation(m, iterations=25)
    rec = json.loads((root / f"action_{a.action:06d}" / "record.json").read_text())["action"]
    x, y, th, L, d = rec
    print(f"기록된 #{a.action}: x={x:.4f} y={y:.4f} th={th:+.2f} 길이={L * 1000:.1f}mm 깊이={d * 1000:.1f}mm -> 그것이 밀어낸 점토를 되돌리는 역방향 푸시 후보를 만듭니다")
    den = float(((hcf - start)[m] ** 2).sum()) + 1e-12
    mae = lambda h, t: float(np.abs(h - t)[m].mean()) * 1000

    def evaluate(h):
        return dict(toward_cf=float(((h - start)[m] * (hcf - start)[m]).sum() / den), mae_cf=mae(h, hcf), mae_target=mae(h, hstar),
                    far_max=float(np.abs(h - start)[~win].max()) * 1000, region_mean_change=float((h - start)[m].mean()) * 1000)

    base = evaluate(start)
    print(f"기준(지금): 영역 MAE(H_cf 대비) {base['mae_cf']:.2f}mm, 영역 MAE(목표 대비) {base['mae_target']:.2f}mm")
    rows = []
    for dfac, lfac, off in itertools.product((0.5, 1.0), (1.0, 2.0), (-0.004, 0.0, 0.004)):
        Lr = L * lfac
        sx = x + Lr * math.cos(th) + off * (-math.sin(th))
        sy = y + Lr * math.sin(th) + off * math.cos(th)
        ang = math.atan2(math.sin(th + math.pi), math.cos(th + math.pi))
        act = [sx, sy, ang, Lr, d * dfac]
        adapter.execute(torch.tensor(act, dtype=torch.float32), 0)
        h = adapter.observe().numpy().astype(np.float64)
        r = evaluate(h)
        r.update(action=act, depth_fac=dfac, length_fac=lfac, offset_mm=off * 1000)
        rows.append(r)
        adapter.restore(saved)
        print(f"  역푸시 깊이x{dfac:g} 길이x{lfac:g} 옆{off * 1000:+.0f}mm: H_cf 쪽 이동 {r['toward_cf']:+.2f} | 영역 MAE(H_cf) {r['mae_cf']:.2f}mm | 영역 MAE(목표) {r['mae_target']:.2f}mm | 영역 밖 최대 {r['far_max']:.1f}mm", flush=True)
    ok = [r for r in rows if r["far_max"] <= a.far_limit_mm] or rows
    best = max(ok, key=lambda r: r["toward_cf"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "undo_candidates.json").write_text(json.dumps(dict(base=base, rows=rows, best=best), ensure_ascii=False, indent=1))
    print(f"\n선택: 깊이x{best['depth_fac']:g} 길이x{best['length_fac']:g} 옆{best['offset_mm']:+.0f}mm -> H_cf 쪽 이동 {best['toward_cf']:+.2f}, 영역 MAE(H_cf) {base['mae_cf']:.2f} -> {best['mae_cf']:.2f}mm, 영역 밖 최대 {best['far_max']:.1f}mm")
    if a.commit:
        adapter.execute(torch.tensor(best["action"], dtype=torch.float32), 0)
        h = adapter.observe().numpy()
        np.save(a.out / "restored_heightmap.npy", h.astype(np.float32))
        tensor_save(a.out / "restored_state.pt", adapter.capture())
        np.save(a.out / "start_m.npy", start.astype(np.float32))
        np.save(a.out / "final_m.npy", h.astype(np.float32))
        print("실행 완료(시뮬레이터에서 확정):", a.out)
    adapter.restore(saved)


if __name__ == "__main__":
    main()
