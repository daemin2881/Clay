"""영역 목표의 '기준 높이'(판이 가라앉은 만큼 목표를 내리는 값)를 어떻게 구하느냐에 따라 후보 순위가 달라지는지,
Genesis 를 다시 돌리지 않고 저장된 후보 결과(--save-transitions 폴더)로 비교한다.

영역 목표 = 목표 + offset_n,  offset_n = median(현재 - 목표) (기준 영역에서, 목표가 평평한 판인 화소만)
  frame   맵 가장자리 1화소 띠        (cf_compensate/region_set 의 border_median 방식. 영역과 무관한 전역 값)
  adj     영역 바로 바깥 띠            (영역에서 0~1.0 도구 반경 밖, 다른 영역 제외)
  wide    더 넓은 바깥 띠              (1.0~2.5 도구 반경 밖)
  plane   수정하지 않을 보존 영역(영역 합집합 둘레 제외, 6 도구 반경 이내)의 국소 평면을 영역 위치에서 읽음
기준 영역의 화소가 모자라면 띠를 넓힌다. offset 은 후보를 실행하기 '전' 상태에서 한 번 구해 모든 후보에 같게 쓴다(후보가 기준을 바꿔 이득을 보지 못하게).
각 방식으로 모든 후보의 J(cf_compensate.score 와 같은 정의)를 다시 계산해 순위를 비교한다:
  최선 후보가 같으면 현재 방식 유지 / 순위가 크게 달라지면 전후 그림으로 확인 / adj·wide·plane 이 계속 더 맞으면 local_offset 적용.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

import cf_common as cc
import cf_compensate as cfc
from region_set import RegionSet


def ref_mask(kind, n, rs, flat, dist_n, union, R, min_px=400):
    others = union & ~rs.masks[n]
    if kind == "frame":
        m = np.zeros(flat.shape, bool)
        m[0] = m[-1] = True
        m[:, 0] = m[:, -1] = True
        return m
    base = flat & ~union
    if kind == "adj":
        lo, hi = 0.0, 1.0
    elif kind == "wide":
        lo, hi = 1.0, 2.5
    else:
        lo, hi = 0.0, 6.0
    for _ in range(6):
        m = base & (dist_n > lo * R) & (dist_n <= hi * R)
        if kind == "plane":
            m &= ~ndimage.binary_dilation(union, iterations=int(0.4 * R))
        if m.sum() >= min_px:
            return m
        hi += 1.0
    return m


def robust_plane(diff, m, region_mask):
    """diff(= 현재 - 목표)를 m 의 화소에서 평면으로 적합(중앙값 기준 반복 재가중)하고, 그 평면을 영역 화소에서 읽는다."""
    ys, xs = np.nonzero(m)
    z = diff[ys, xs]
    A = np.c_[np.ones(len(z)), (xs - 256) / 256.0, (ys - 256) / 256.0]
    w = np.ones(len(z))
    for _ in range(8):
        coef = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
        r = z - A @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = 1.0 / np.maximum(np.abs(r) / (2.0 * s), 1.0)
    ry, rx = np.nonzero(region_mask)
    return float((np.c_[np.ones(len(ry)), (rx - 256) / 256.0, (ry - 256) / 256.0] @ coef).mean()), coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--transitions", required=True, type=Path)
    ap.add_argument("--states", required=True, type=Path, help="확정 상태 높이맵 iter_XXX.npy 가 있는 폴더(cf/states). state_tag 번호와 같은 번호")
    ap.add_argument("--tag", type=int, default=0, help="이 상태에서 실행해 본 후보들만 비교")
    ap.add_argument("--eps", type=float, default=0.01)
    a = ap.parse_args()
    root = a.history_dir.resolve()
    R = cc.geometry(root)["tool_r_px"]
    tgt = cc.load_target(root).astype(np.float64)
    h0 = np.load(a.states / f"iter_{a.tag:03d}.npy").astype(np.float64)
    rs0 = RegionSet.load(a.regions, ref_default=h0)
    names, union = rs0.names, rs0.union()
    fp = cc.relief_focus(tgt, R, base=cc.plate_level(root))
    flat = ~(ndimage.distance_transform_edt(~fp["relief"]) <= 0.4 * R)
    # 후보
    cands = []
    for f in sorted(glob.glob(str(a.transitions / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if int(z["state_tag"]) == a.tag:
            cands.append((np.asarray(z["action"], float), h0 + z["delta"].astype(np.float64)))
    for n in names:
        d = (rs0.targets[n] - tgt)[rs0.masks[n]] * 1000
        print(f"  [{n}] 저장 목표 - 순수 목표(mm) 영역 안: 중앙값 {np.median(d):+.2f}, 표준편차 {d.std():.2f}, 최소/최대 {d.min():+.2f}/{d.max():+.2f}")
    print(f"상태 iter_{a.tag:03d} 에서 실행해 본 후보 {len(cands)}개 | 영역 {names}")
    diff0 = h0 - tgt
    kinds = ["frame", "adj", "wide", "plane"]
    ring_px, far_px = int(round(0.94 * R)), int(round(1.17 * R))
    cfgs, offs = {}, {}
    for kd in kinds:
        new_t, off = {}, {}
        for n in names:
            dist_n = ndimage.distance_transform_edt(~rs0.masks[n])
            m = ref_mask(kd, n, rs0, flat, dist_n, union, R)
            if kd == "plane":
                off[n], _ = robust_plane(diff0, m, rs0.masks[n])
            else:
                off[n] = float(np.median(diff0[m]))
            sv = float(np.median((rs0.targets[n] - tgt)[rs0.masks[n]]))
            new_t[n] = rs0.targets[n] + (off[n] - sv)          # 저장된 영역 목표의 '모양'은 그대로 두고 기준 높이만 바꾼다
        rs = RegionSet(names, rs0.masks, new_t, rs0.labels, h0, rs0.meta)
        cfgs[kd] = cfc.build_cfg(rs, h0, 1.0, ring_px, far_px, R)
        offs[kd] = off
    base_cfg = cfc.build_cfg(rs0, h0, 1.0, ring_px, far_px, R)      # 저장된 영역 목표 그대로(= 실제 실행 때 쓴 것)
    cfgs["saved"], kinds = base_cfg, ["saved"] + kinds
    print("\n영역별 기준 높이 offset (mm, 목표에 더하는 값 = 현재 판이 목표 판보다 낮은 정도), 영역 목표가 현재 판에 얼마나 맞는지")
    print(f"{'영역':<14}" + "".join(f"{k:>9}" for k in kinds))
    for n in names:
        sv = float(np.median((rs0.targets[n] - tgt)[rs0.masks[n]]) * 1000)
        print(f"{n:<14}" + f"{sv:>+9.2f}" + "".join(f"{offs[k][n] * 1000:>+9.2f}" for k in kinds[1:]))
    J = {k: np.array([cfc.score(m, cfgs[k], h0)["J"] for _, m in cands]) for k in kinds}
    Jnow = {k: cfc.score(h0, cfgs[k], h0)["J"] for k in kinds}
    print("\n방식별 후보 순위 (J 낮을수록 좋음; 현재 상태 J 는 방식마다 따로 정규화되어 시작=3.0)")
    ref = J["saved"]
    print(f"{'방식':<8}{'현재J':>8}{'최선J':>8}{'개선 후보':>9}{'순위 상관(저장 목표 대비)':>24}{'상위5 겹침':>12}{'최선 후보':>16}")
    top = lambda v, k=5: set(np.argsort(v)[:k])
    for k in kinds:
        v = J[k]
        rho = stats.spearmanr(ref, v)[0]
        print(f"{k:<8}{Jnow[k]:>8.3f}{v.min():>8.3f}{int((v < Jnow[k] - a.eps).sum()):>9d}{rho:>24.2f}{len(top(ref) & top(v)):>12d}/5{'  #' + str(int(np.argmin(v))):>16}"
              + ("   (같은 후보)" if int(np.argmin(v)) == int(np.argmin(ref)) else "   (다른 후보)"))
    b = int(np.argmin(ref))
    print(f"\n저장 목표 방식의 최선 후보 #{b}: x={cands[b][0][0]:.4f} y={cands[b][0][1]:.4f} th={cands[b][0][2]:+.2f} L={cands[b][0][3] * 1000:.0f}mm d={cands[b][0][4] * 1000:.0f}mm")
    for k in kinds:
        print(f"  {k:<7} 에서 그 후보의 순위 {int((J[k] < J[k][b]).sum()) + 1}/{len(cands)}, J {Jnow[k]:.3f} -> {J[k][b]:.3f}")


if __name__ == "__main__":
    main()
