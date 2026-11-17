#!/usr/bin/env python3
"""체인 실행(cf_chain.py)의 단계별 높이맵을 512x512 흑백 PNG 로 저장한다 (밝을수록 높다). 돌고 있는 실행에는 손대지 않고 결과 파일만 읽는다.

저장 위치: <out>/heightmaps/
  target_A.png            목표 A 높이맵(판 배경 높이에 맞춘 것)
  stage_000_start.png     시작 상태(푸시 80번째 직후)
  stage_001_<영역>.png    1단계 확정 직후 상태, stage_002_...  (확정한 단계마다)
  *.npy                   같은 내용의 높이맵(float32, m)
  scale.txt               밝기 기준(0 = ? mm, 255 = ? mm). 모든 그림에 같은 기준을 써서 그림끼리 밝기를 직접 비교할 수 있다.
밝기 기준은 시작 상태와 목표의 높이 분포(0.5%~99.5%)로 한 번 정해 고정한다. 범위를 벗어난 곳은 검정/흰색으로 잘린다.

사용: python export_chain_heightmaps.py --out results/case_A/chain_A_rot            (한 번 내보내고 끝)
      python export_chain_heightmaps.py --out results/case_A/chain_A_rot --watch 30   (30초마다 새 단계가 있는지 보고 내보낸다)
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


def to_u8(h_m, lo_m, hi_m):
    return np.clip((np.asarray(h_m, np.float64) - lo_m) / max(hi_m - lo_m, 1e-9), 0.0, 1.0).__mul__(255.0).round().astype(np.uint8)


def save_png(path, u8):
    try:
        from PIL import Image
        Image.fromarray(u8, mode="L").save(path)
    except ImportError:
        import matplotlib.image as mpimg
        mpimg.imsave(str(path), u8, cmap="gray", vmin=0, vmax=255)


def stage_dirs(out):
    """단계 k 의 노드 폴더: d0 (1단계), d1_0 (2단계), d2_0 (3단계) ... 정렬해서 돌려준다."""
    found = []
    for p in Path(out).iterdir():
        m = re.fullmatch(r"d(\d+)(?:_0)?", p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def export_once(out, cond="E5", seed=1, target_npy=None):
    out = Path(out)
    hm = out / "heightmaps"
    hm.mkdir(exist_ok=True)
    import cf_common as cc
    from region_set import border_median
    root0 = out / "d0" / "view"
    if not root0.exists() and (out / "chain.json").exists():      # 이력 전체를 그대로 쓰는 실행(예: 하트 K=190)은 d0/view 를 만들지 않는다 -> 체인 기록의 시작 노드 폴더를 쓴다
        root0 = Path(json.loads((out / "chain.json").read_text())["nodes"][0]["root"])
    h0 = cc.state_height(root0).astype(np.float64)
    t = cc.load_target(root0, target_npy).astype(np.float64)
    goal = t - border_median(t) + border_median(h0)
    scale_p = hm / "scale.txt"
    if scale_p.exists():                                  # 한 번 정한 기준은 바꾸지 않는다(단계 사이 비교가 깨지지 않게)
        lo, hi = [float(x) for x in re.findall(r"=\s*([-\d.]+)\s*mm", scale_p.read_text())[:2]]
        lo, hi = lo / 1000.0, hi / 1000.0
    else:
        both = np.concatenate([h0.ravel(), goal.ravel()])
        lo, hi = [float(v) for v in np.percentile(both, [0.5, 99.5])]
        scale_p.write_text(f"밝기 0   = {lo * 1000:.2f} mm\n밝기 255 = {hi * 1000:.2f} mm\n(시작 상태와 목표 A 의 높이 분포 0.5%~99.5% 로 정한 고정 기준. 밝을수록 높다. 범위 밖은 잘림)\n")
    wrote = []
    for name, arr in (("target_A", goal), ("stage_000_start", h0)):
        p = hm / f"{name}.png"
        if not p.exists():
            save_png(p, to_u8(arr, lo, hi))
            np.save(hm / f"{name}.npy", arr.astype(np.float32))
            wrote.append(p.name)
    for k, nd in enumerate(stage_dirs(out), 1):
        ext = nd / "child0" / "ext_final_m.npy"
        if not ext.exists():
            continue
        region = "x"
        rj = nd / f"{cond}_s{seed}" / "cf" / "rounds.json"
        if rj.exists():
            rec = [r for r in json.loads(rj.read_text()) if r.get("committed")]
            if rec:
                region = "+".join(dict.fromkeys(r.get("region", "x") for r in rec))      # 한 단계에서 여러 번 확정하면 영역 이름을 이어 붙인다
        p = hm / f"stage_{k:03d}_{region}.png"
        if not p.exists():
            h = np.load(ext).astype(np.float64)
            save_png(p, to_u8(h, lo, hi))
            np.save(hm / f"stage_{k:03d}_{region}.npy", h.astype(np.float32))
            wrote.append(p.name)
    return wrote, (lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--watch", type=float, default=0.0, help="초. 0 이면 한 번 내보내고 끝")
    ap.add_argument("--cond", default="E5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-npy", default="", help="이력의 목표 대신 쓸 원래 목표(512x512, m). 예: results/case_heart/heart_target_raw_flipud_512_m.npy")
    a = ap.parse_args()
    while True:
        try:
            wrote, (lo, hi) = export_once(a.out, a.cond, a.seed, a.target_npy or None)
            if wrote:
                print(f"[{time.strftime('%T')}] 저장: {', '.join(wrote)} (밝기 0={lo * 1000:.1f} mm, 255={hi * 1000:.1f} mm)", flush=True)
        except (FileNotFoundError, KeyError, IndexError) as e:      # 실행이 막 시작돼 아직 시작 상태/체인 기록이 없을 수 있다: 감시 모드에서는 기다렸다가 다시 시도한다
            if a.watch <= 0:
                raise
            print(f"[{time.strftime('%T')}] (아직 준비 안 됨: {e!r})", flush=True)
        if a.watch <= 0:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
