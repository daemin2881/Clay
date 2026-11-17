"""영역(홈)에 관련된 과거 행동을 전부 후보로 삼아, 현재 상태에서 물리 시도(실행 -> 관측 -> 롤백)로 목표 map 에 가까워지는 보상을 찾는다.

  관련 행동 : 80 번째까지의 모든 행동 중, 도구 접촉영역이 영역 근처(10화소)를 지났거나 기록된 전/후 높이 변화가 영역 안에서 --min-change-mm 이상인 것 (전부)
  후보      : 관련 행동마다 (깊이 x0.5), (방향 반대 + 깊이 x0.5), (옆으로 +4mm + 깊이 x0.5), (깊이 x1 로 다시 누르기)
  평가      : 후보를 Genesis 에서 실제로 실행해 보고 되돌린다. 영역 주변 창에서 원래 목표 map(h_star) 대비 MAE 를 잰다. 영역 밖 변화가 --far-limit-mm 초과면 제외.
  병렬      : --shard i --nshards N 으로 후보를 나눠 여러 프로세스가 동시에 평가한다. 각자 out/shard_i.json 저장.
  확정      : --commit 이면 모든 shard 결과 중 최선을 실제로 실행(뷰어 가능)하고 전/후 높이맵 PNG 와 상태를 저장한다.
사용: python groove_search.py --history-dir H --region-spec S --out DIR [--list | --shard 0 --nshards 3 | --commit [--viewer]]
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

X0, Y0, PS = 0.44 - 0.18, -0.18, 0.36 / 511


def related_actions(root, m, min_change_mm):
    import first_beam as fb
    man, actions, before, after = fb.load_history(root)
    g = fb.map_geometry(man)
    near = ndimage.binary_dilation(m, iterations=10)
    out = []
    for t in range(len(actions)):
        fp = fb.footprint(g, actions[t])
        ch = float(np.abs((after[t] - before[t]) * 1000)[m].mean())
        if (fp & near).any() or ch >= min_change_mm:
            out.append((t + 1, actions[t].tolist(), ch))
    return out


def make_candidates(rel):
    c = []
    for n, (x, y, th, L, d), ch in rel:
        px, py = -math.sin(th), math.cos(th)
        wrap = lambda t: math.atan2(math.sin(t), math.cos(t))
        c += [(n, "d*0.5", [x, y, th, L, d * 0.5]),
              (n, "reverse d*0.5", [x + L * math.cos(th), y + L * math.sin(th), wrap(th + math.pi), L, d * 0.5]),
              (n, "shift+4mm d*0.5", [x + 0.004 * px, y + 0.004 * py, th, L, d * 0.5]),
              (n, "repress d*1", [x, y, th, L, d])]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--region-spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-change-mm", type=float, default=0.15)
    ap.add_argument("--far-limit-mm", type=float, default=6.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()
    root = a.history_dir.resolve()
    m_raw = np.load(a.region_spec / "m_edit.npy") > 0
    h_star = np.load(a.region_spec / "h_star.npy").astype(np.float64)
    rel = related_actions(root, m_raw, a.min_change_mm)
    cands = make_candidates(rel)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.list:
        print(f"영역 마스크 {int(m_raw.sum())}px 에 관련된 과거 행동 {len(rel)}개 (전체 {len(list(root.glob('action_??????')))}개 중): " + ", ".join(f"#{n}" for n, _, _ in rel))
        print(f"후보 {len(cands)}개 (행동당 4가지 변형)")
        return
    man = json.loads((root / "manifest.json").read_text())
    det = man["signature"]["details"]
    os.environ["VS_MATERIAL"] = det["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if det["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "1" if a.viewer else "0"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature, tensor_save

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != man["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다.")
    last = len([d for d in root.glob("action_??????") if d.is_dir()])
    adapter.restore(load_snapshot(root / f"action_{last:06d}" / "post_state.pt"))
    h0 = adapter.observe().numpy().astype(np.float64)
    saved = adapter.capture()
    win = ndimage.binary_dilation(m_raw, iterations=25)
    score = lambda h: float(np.abs(h - h_star)[win].mean()) * 1000
    mae = lambda h: float(np.abs(h - h_star)[m_raw].mean()) * 1000
    base = dict(window_mae=score(h0), region_mae=mae(h0))
    if not a.commit:
        mine = cands[a.shard::a.nshards]
        res = []
        for n, kind, act in mine:
            adapter.execute(torch.tensor(act, dtype=torch.float32), 0)
            h = adapter.observe().numpy().astype(np.float64)
            res.append(dict(source=n, kind=kind, action=act, window_mae=score(h), region_mae=mae(h), far_max=float(np.abs(h - h0)[~win].max()) * 1000))
            adapter.restore(saved)
            print(f"  #{n:>3} {kind:<16} 창 MAE {res[-1]['window_mae']:.3f} (현재 {base['window_mae']:.3f}) 영역 MAE {res[-1]['region_mae']:.3f} 밖 {res[-1]['far_max']:.1f}mm", flush=True)
        (a.out / f"shard_{a.shard}.json").write_text(json.dumps(dict(base=base, results=res), ensure_ascii=False, indent=1))
        print(f"shard {a.shard} 완료: {len(res)}개 평가")
        return
    # 확정: 모든 shard 결과 중 최선
    allr = []
    for f in sorted(a.out.glob("shard_*.json")):
        allr += json.loads(f.read_text())["results"]
    ok = [r for r in allr if r["far_max"] <= a.far_limit_mm] or allr
    allr.sort(key=lambda r: r["window_mae"])
    print(f"평가한 후보 {len(allr)}개. 현재 창 MAE {base['window_mae']:.3f}mm, 영역 MAE {base['region_mae']:.3f}mm")
    print("상위 5개:", "; ".join(f"#{r['source']} {r['kind']} 창 {r['window_mae']:.3f}" for r in allr[:5]))
    best = min(ok, key=lambda r: r["window_mae"])
    improved = best["window_mae"] < base["window_mae"] - 0.01
    print(f"최선: #{best['source']} {best['kind']} 창 MAE {best['window_mae']:.3f} / 영역 MAE {best['region_mae']:.3f} / 영역 밖 {best['far_max']:.1f}mm -> {'개선' if improved else '개선 없음(확정하지 않음)'}")
    if not improved:
        (a.out / "commit.json").write_text(json.dumps(dict(base=base, best=best, committed=False, n_candidates=len(allr)), ensure_ascii=False, indent=1))
        return
    adapter.execute(torch.tensor(best["action"], dtype=torch.float32), 0)
    h1 = adapter.observe().numpy().astype(np.float64)
    np.save(a.out / "start_m.npy", h0.astype(np.float32))
    np.save(a.out / "final_m.npy", h1.astype(np.float32))
    tensor_save(a.out / "final_state.pt", adapter.capture())
    ys, xs = np.where(m_raw)
    r0, r1, c0, c1 = max(ys.min() - 50, 0), min(ys.max() + 50, 512), max(xs.min() - 50, 0), min(xs.max() + 50, 512)
    bg = float(np.median(np.concatenate([h0[0], h0[-1], h0[:, 0], h0[:, -1]])))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8), facecolor="#fcfcfb")
    kw = dict(cmap="turbo", vmin=-10, vmax=14)
    for a_, img, t in ((ax[0], h_star, "Target map (groove region)"), (ax[1], h0, "Before (state 80)"), (ax[2], h1, "After compensation action")):
        a_.imshow(((img - bg) * 1000)[r0:r1, c0:c1], **kw); a_.set_title(t, fontsize=9)
    im = ax[3].imshow(((h1 - h0) * 1000)[r0:r1, c0:c1], cmap="RdBu", vmin=-4, vmax=4); ax[3].set_title("After - Before (mm)", fontsize=9)
    for a_ in ax:
        a_.contour(m_raw[r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=0.9, linestyles="--"); a_.set_xticks([]); a_.set_yticks([])
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    fig.suptitle(f"Committed: past #{best['source']} {best['kind']} | window MAE {base['window_mae']:.3f} -> {score(h1):.3f} mm | region MAE {base['region_mae']:.3f} -> {mae(h1):.3f} mm", fontsize=10)
    fig.savefig(a.out / "before_after_compensation.png", dpi=95, facecolor="#fcfcfb", bbox_inches="tight")
    (a.out / "commit.json").write_text(json.dumps(dict(base=base, best=best, committed=True, after=dict(window_mae=score(h1), region_mae=mae(h1)), n_candidates=len(allr)), ensure_ascii=False, indent=1))
    print(f"확정 실행 완료: 창 MAE {base['window_mae']:.3f} -> {score(h1):.3f}, 영역 MAE {base['region_mae']:.3f} -> {mae(h1):.3f}")


if __name__ == "__main__":
    main()
