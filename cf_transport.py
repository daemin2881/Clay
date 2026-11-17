"""점토 수송 계획: 점토를 '보존되는 자원'으로 보고, 채울 곳에 필요한 점토를 어디서 가져올지를 최적 수송(선형계획)으로 정한 뒤 그 흐름을 따라 미는 푸시 후보를 만든다.

왜: 채울 곳이 필요로 하는 점토는 깎을 곳이 내놓는 점토보다 많을 수 있다(A 의 요청 영역 10곳: 필요 14,934 mm^3, 깎을 곳 8,404 mm^3). 기존 G2 후보는 구덩이 주변 창 안에서 12 mm 만 밀어서
먼 곳의 점토를 가져오는 구조가 없고, 가져오다 보면 이웃을 깊게 파는 구덩이가 생겼다(영역 밖 12~16 mm). 여기서는 가져올 곳을 계획하고, 한 곳에서 너무 많이 깎지 않게 하며(깎는 양을 화소당 얕게),
이미 솟아 있는 곳(구조)에서 가져오는 것을 비싸게 매긴다.

대상(A, 하트, 고양이, ...)에 의존하지 않는다. 입력은 현재 높이맵, 채울 영역과 그 목표, 건드리면 안 되는 영역(요청·보호 영역), 깎아도 되는 영역(깎을 곳), 도달 범위뿐이다.
후보는 계획일 뿐이고 확정 전에 항상 Genesis 로 실제로 눌러 보고 평가한다.

수송 문제(선형계획)
  공급 j(칸): 판 칸 = 깊이 depth_mm x 칸 면적 x 가져가도 되는 비율, 깎을 곳 칸 = 현재 - 목표(전부)
  수요 i(채울 칸): 목표 - 현재
  비용 = 거리(mm) x 해(harm): 판 칸은 1 + relief_weight x (그 칸이 판 배경보다 솟은 정도 0~1), 깎을 곳 칸은 0.2(원래 깎아야 하는 점토라 값싸다)
  부족한 수요는 큰 벌점(big_m)의 미충족 변수로 둔다 -> 점토가 모자라면 가능한 만큼만 채우는 계획이 나온다.
"""
import math

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

SIZE = 512


def _bsum(a, cell):
    n = SIZE // cell
    return a[: n * cell, : n * cell].reshape(n, cell, n, cell).sum(axis=(1, 3))


def _bmean(a, cell):
    return _bsum(a, cell) / float(cell * cell)


def plan_transport(h, sink_mask, sink_target, forbid, excess_donor, reach, px_mm, *, cell=16, depth_mm=0.5, relief_weight=3.0, max_dist_mm=200.0, big_m=1e4, min_flow_mm3=15.0):
    """채울 영역(sink_mask)에 필요한 점토의 수송 계획.

    h, sink_target: 512x512 높이(m). sink_mask: 채울 영역(bool). forbid: 점토를 가져가면 안 되는 곳(요청·보호 영역 합집합 등, bool).
    excess_donor: 512x512 (m) — 내놓아도 되는 점토(깎을 곳의 현재-목표, 0 이상. 없으면 0). reach: 도구가 닿는 곳(bool). px_mm: 1 화소의 mm.
    반환 dict(flows=[(행,열,행,열,부피mm3,거리mm,종류)], need_mm3, supplied={plate, cut}, unmet_mm3, n_sinks, n_donors, status)."""
    cell_area = (cell * px_mm) ** 2                      # 칸 하나의 면적(mm^2)
    pix_area = px_mm ** 2
    need_mm = np.clip(sink_target - h, 0.0, None) * 1000.0 * sink_mask                       # mm
    demand = _bsum(need_mm * pix_area, cell)                                               # 칸별 수요(mm^3)
    n = demand.shape[0]
    centers = (np.arange(n) + 0.5) * cell
    sink_cells = np.argwhere(demand > max(0.02 * demand.max(), 1.0)) if demand.max() > 0 else np.zeros((0, 2), int)
    need_total = float(demand[tuple(sink_cells.T)].sum()) if len(sink_cells) else 0.0
    if not len(sink_cells):
        return dict(flows=[], need_mm3=0.0, supplied=dict(plate=0.0, cut=0.0), unmet_mm3=0.0, n_sinks=0, n_donors=0, status="수요 없음")
    free = 1.0 - _bmean((np.asarray(forbid, bool) | ~np.asarray(reach, bool)).astype(float), cell)     # 가져가도 되는 비율
    cap_plate = depth_mm * cell_area * np.clip(free, 0.0, 1.0)
    bg = float(np.median(np.concatenate([h[0], h[-1], h[:, 0], h[:, -1]])))
    hc = _bmean((h - bg) * 1000.0, cell)
    scale = max(float(np.percentile((h - bg) * 1000.0, 90)), 1e-6)
    relief = np.clip(hc / scale, 0.0, 1.0)                                                  # 판 배경보다 솟은 정도(0~1): 구조일수록 비싸다
    cap_cut = _bsum(np.clip(excess_donor, 0.0, None) * 1000.0 * pix_area, cell)
    harm_plate = 1.0 + relief_weight * relief
    donors = []                                                                             # (행, 열, 용량, 해, 종류)
    for r in range(n):
        for c in range(n):
            if cap_cut[r, c] > 5.0:
                donors.append((r, c, float(cap_cut[r, c]), 0.2, "cut"))
            elif cap_plate[r, c] > 1.0:
                donors.append((r, c, float(cap_plate[r, c]), float(harm_plate[r, c]), "plate"))
    if not donors:
        return dict(flows=[], need_mm3=need_total, supplied=dict(plate=0.0, cut=0.0), unmet_mm3=need_total, n_sinks=len(sink_cells), n_donors=0, status="공급 없음")
    D = np.array([[centers[d[0]], centers[d[1]]] for d in donors])
    arcs, cost = [], []
    for si, (sr, sc) in enumerate(sink_cells):
        dist = np.hypot(D[:, 0] - centers[sr], D[:, 1] - centers[sc]) * px_mm
        for j in np.nonzero(dist <= max_dist_mm)[0]:
            arcs.append((int(j), si))
            cost.append(max(dist[j], px_mm * cell * 0.5) * donors[j][3])
    if not arcs:
        return dict(flows=[], need_mm3=need_total, supplied=dict(plate=0.0, cut=0.0), unmet_mm3=need_total, n_sinks=len(sink_cells), n_donors=len(donors), status="도달 가능한 공급 없음")
    na, ns, nd = len(arcs), len(sink_cells), len(donors)
    rows_u = [j for j, _ in arcs]
    A_ub = sparse.csr_matrix((np.ones(na), (rows_u, np.arange(na))), shape=(nd, na + ns))
    b_ub = np.array([d[2] for d in donors])
    rows_e = [si for _, si in arcs] + list(range(ns))
    cols_e = list(range(na)) + [na + k for k in range(ns)]
    A_eq = sparse.csr_matrix((np.ones(na + ns), (rows_e, cols_e)), shape=(ns, na + ns))
    b_eq = np.array([demand[sr, sc] for sr, sc in sink_cells])
    res = linprog(np.concatenate([cost, np.full(ns, big_m)]), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return dict(flows=[], need_mm3=need_total, supplied=dict(plate=0.0, cut=0.0), unmet_mm3=need_total, n_sinks=ns, n_donors=nd, status=f"선형계획 실패: {res.message[:40]}")
    x = res.x
    flows, sup = [], dict(plate=0.0, cut=0.0)
    for k, (j, si) in enumerate(arcs):
        if x[k] >= min_flow_mm3:
            r, c, _, _, kind = donors[j]
            sr, sc = sink_cells[si]
            flows.append((int(r), int(c), int(sr), int(sc), float(x[k]), float(np.hypot(centers[r] - centers[sr], centers[c] - centers[sc]) * px_mm), kind))
            sup[kind] += float(x[k])
    unmet = float(x[na:].sum())
    flows.sort(key=lambda f: -f[4])
    return dict(flows=flows, need_mm3=need_total, supplied=sup, unmet_mm3=unmet, n_sinks=ns, n_donors=nd, status="ok", cell=cell)


def flows_to_pushes(plan, g, bounds, clamp, k_flows=8, cell=16, tool_r_mm=15.0, l_pref_mm=(40.0, 80.0), depth_mult=(0.5, 1.0, 2.0)):
    """수송 흐름 -> 푸시 후보 [x, y, theta, L, d] (단위 m, rad).

    흐름마다: 채울 칸 중심에서 가져올 칸 쪽으로 L 만큼 물러난 곳에서 시작해 채울 칸 중심까지 민다(점토가 도구 앞쪽으로 밀려 채울 곳에 쌓인다). 방향은 가져올 칸 -> 채울 칸.
    깊이는 흐름의 부피를 도구 폭(2 x 반경) x 길이로 나눈 값을 기준으로 0.5, 1, 2 배(플래너 안전 범위로 자름). 거리가 길면 L 을 두 가지(40, 80 mm)로 만든다.
    g: 이력 manifest 기하(cf_common.geometry). bounds: cf_common.action_bounds(). clamp: 행동을 안전 범위로 자르는 함수."""
    import cf_common as cc
    out = []
    for (r, c, sr, sc, vol, dist, kind) in plan["flows"][:k_flows]:
        ra, ca = (r + 0.5) * cell, (c + 0.5) * cell
        rb, cb = (sr + 0.5) * cell, (sc + 0.5) * cell
        xa, ya = cc.pix_to_world(g, ra, ca)
        xb, yb = cc.pix_to_world(g, rb, cb)
        th = math.atan2(yb - ya, xb - xa)
        D = math.hypot(xb - xa, yb - ya)
        for lp in l_pref_mm:
            L = min(D, lp * 1e-3, bounds[5])
            if L < max(bounds[4], 2e-3):
                continue
            x, y = xb - L * math.cos(th), yb - L * math.sin(th)
            d_est = vol / (2.0 * tool_r_mm * (L * 1000.0))           # mm
            for m in depth_mult:
                d = float(np.clip(m * d_est * 1e-3, max(bounds[6], 1e-3), bounds[7]))
                out.append(clamp([x, y, th, L, d]))
    seen, uniq = set(), []
    for a in out:
        k = tuple(round(float(v), 6) for v in a)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq


def round_candidates(h, n, names, masks, tgts, requested, g, bounds, clamp, reach, k_flows=8, depth_mm=0.5, relief_weight=3.0, cell=16):
    """한 라운드(채우는 영역 n)의 점토 수송 후보와 계획 요약. 대상·영역 이름에 의존하지 않는다.
    채울 곳 = 영역 n 의 부족분(목표 - 현재 > 0). 가져와도 되는 곳 = 요청한 영역들(requested)에서 현재가 목표보다 높은 만큼(깎을 곳, 영역 n 자신의 남는 곳 포함)과, 어떤 영역에도 속하지 않는 판(깊이 depth_mm 까지).
    가져가면 안 되는 곳 = 모든 영역(채울 다른 영역, 보호 영역 포함)과 도구가 닿지 않는 곳. 한 번에 밀 수 있는 길이(bounds[5])가 짧아서 가져올 최대 거리는 그 3배로 둔다(더 먼 점토는 라운드를 거듭하며 이어서 옮긴다).
    반환: (후보 행동 목록, 요약 dict)."""
    px_mm = g["px_mm"]
    sink = np.asarray(masks[n], bool)
    forbid = np.zeros_like(sink)
    for m in names:
        forbid |= np.asarray(masks[m], bool)
    ex = np.zeros((SIZE, SIZE))
    for m in requested:
        mm_ = np.asarray(masks[m], bool)
        ex[mm_] = np.clip(h - tgts[m], 0.0, None)[mm_]
    plan = plan_transport(h, sink, tgts[n], forbid, ex, reach, px_mm, cell=cell, depth_mm=depth_mm, relief_weight=relief_weight, max_dist_mm=3.0 * bounds[5] * 1000.0)
    cands = flows_to_pushes(plan, g, bounds, clamp, k_flows=k_flows, cell=cell, tool_r_mm=g["tool_r"] * 1000.0)
    summ = dict(status=plan["status"], need_mm3=round(plan["need_mm3"]), cut_mm3=round(plan["supplied"]["cut"]), plate_mm3=round(plan["supplied"]["plate"]), unmet_mm3=round(plan["unmet_mm3"]), flows=len(plan["flows"]), cands=len(cands))
    return cands, summ


def squeeze_candidates(h, mask, target, g, bounds, clamp, avoid=None, k_blobs=3, n_angles=8, keep_angles=4, lengths_mm=(5.0, 10.0), depths_mm=(3.0, 6.0), thr_mm=1.5, min_blob_px=40):
    """'짜내기' 후보: 채울 곳(구덩이) 바로 바깥 둘레를 눌러서 눌린 점토가 구덩이 쪽으로 밀려 올라오게 한다(여드름 짜듯이). 기존 linear_push 와 같은 행동 [x, y, theta, L, d]이고
    점토를 멀리서 가져오지 않는다. 대상·영역 이름에 의존하지 않는다.
    부족 덩어리(목표 - 현재 > thr_mm)마다 무게중심을 구하고, 덩어리 가장자리에 도구가 맞닿는 둘레(반경 = 덩어리 반경 + 도구 반경 + 여유)의 n_angles 개 지점 중
    그 자리 높이가 가장 높은(눌러서 짜낼 점토가 많은) keep_angles 개를 골라, 중심 쪽으로 짧게(lengths_mm) 누른다(깊이 depths_mm). avoid: 건드리면 안 되는 곳(보호 영역)과 겹치는 지점은 뺀다.
    반환: (후보 행동 목록, 요약 dict)."""
    import cf_common as cc
    import first_beam as fb
    from scipy import ndimage
    mask = np.asarray(mask, bool)
    res = (target - h) * 1000.0
    deficit = (res > thr_mm) & mask
    lab, nb = ndimage.label(deficit)
    if nb == 0:
        return [], dict(blobs=0, cands=0)
    sizes = ndimage.sum(deficit, lab, range(1, nb + 1))
    order = [i + 1 for i in np.argsort(-np.asarray(sizes)) if sizes[i] >= min_blob_px][:k_blobs]
    px_per_m = g["px_per_m_x"]
    R_tool_px = g["tool_r"] * px_per_m
    rr, cc_ = np.mgrid[0:SIZE, 0:SIZE]
    out, skipped = [], 0
    centers_by_action = {}
    for b in order:
        m = lab == b
        w = res[m]
        cy, cx = float((rr[m] * w).sum() / w.sum()), float((cc_[m] * w).sum() / w.sum())
        r_b = math.sqrt(m.sum() / math.pi)
        rho = r_b + R_tool_px + 0.15 * R_tool_px
        pts = []
        for k in range(n_angles):
            ph = 2 * math.pi * k / n_angles
            py, px = cy + rho * math.sin(ph), cx + rho * math.cos(ph)
            if not (0 <= py < SIZE and 0 <= px < SIZE):
                continue
            y0, y1, x0, x1 = int(max(py - R_tool_px, 0)), int(min(py + R_tool_px + 1, SIZE)), int(max(px - R_tool_px, 0)), int(min(px + R_tool_px + 1, SIZE))
            pts.append((float(h[y0:y1, x0:x1].mean()), py, px, ph))
        pts.sort(reverse=True)                                         # 높은 곳 = 짜낼 점토가 많은 곳
        for _, py, px, ph in pts[: keep_angles + 2]:
            xs, ys = cc.pix_to_world(g, py, px)
            xc, yc = cc.pix_to_world(g, cy, cx)
            th = math.atan2(yc - ys, xc - xs)                          # 둘레 지점 -> 구덩이 중심
            made = 0
            for L in lengths_mm:
                for d in depths_mm:
                    a = clamp([xs, ys, th, L * 1e-3, d * 1e-3])
                    if avoid is not None and np.asarray(avoid).any():
                        fp = fb.footprint(g, a)
                        if (fp & np.asarray(avoid, bool)).sum() > 0.02 * max(fp.sum(), 1):
                            skipped += 1
                            continue
                    out.append(a)
                    centers_by_action[id(a)] = (float(xc), float(yc))
                    made += 1
            if made and len({tuple(round(float(v), 6) for v in c[:2]) for c in out}) >= keep_angles * len(order):
                break
    seen, uniq = set(), []
    for a in out:
        key = tuple(round(float(v), 6) for v in a)
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq, dict(blobs=len(order), cands=len(uniq), skipped_protected=skipped, centers={tuple(round(float(v), 6) for v in a): centers_by_action[id(a)] for a in uniq if id(a) in centers_by_action})


def hint_match_candidates(h, T, win, g, bounds, clamp, avoid=None, k_blobs=3, thr_mm=0.5, min_blob_px=30, keep_angles=4,
                          lengths_mm=(5.0, 8.0, 12.0), depths_mm=(2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 22.0, 26.0)):
    """힌트 맞춤 후보(HM): 반사실 목표 T(과거를 달리했다면 지금 이랬을 상태)와 현재의 차이 D = T - h 를 '얼마나 더 올리고(+) 얼마나 더 눌러야(-) 하는지' 지도로 보고,
    D 가 큰 덩어리(양수, 음수 따로)마다 푸시 후보의 넓은 격자(시작 위치 x 길이 x 깊이)를 만든다. 세기(길이·깊이)는 여기서 정하지 않고, 호출하는 쪽이 모델로 예측한
    결과가 T 에 가장 가까운 것만 골라 Genesis 로 평가한다. 대상·영역 이름에 의존하지 않는다.
      시작 위치: (a) 덩어리 둘레(반경 = 덩어리 반경 + 도구 반경 + 여유)에서 높은 곳 keep_angles 개 -> 중심 쪽으로 누름(양옆에서 눌러 솟게 / 점토 이동)
                 (b) 덩어리 중심 -> 4방향으로 눌러 지나감(깎기, 또는 지나가며 옮기기)
    win: D 를 볼 창(영역을 넓힌 것). avoid: 건드리면 안 되는 곳(보호 영역). 반환: (후보 행동 목록, 요약 dict: 덩어리별 부호·크기·평균 D)."""
    import cf_common as cc
    import first_beam as fb
    from scipy import ndimage
    win = np.asarray(win, bool)
    D = (T - h) * 1000.0
    px_per_m = g["px_per_m_x"]
    R_tool_px = g["tool_r"] * px_per_m
    rr, cc_ = np.mgrid[0:SIZE, 0:SIZE]
    blobs = []
    for sign in (+1, -1):
        m_ = (sign * D > thr_mm) & win
        m_ = ndimage.binary_opening(m_, iterations=1) if m_.sum() > 4 * min_blob_px else m_
        lab, nb = ndimage.label(m_)
        for i in range(1, nb + 1):
            mm_ = lab == i
            if mm_.sum() >= min_blob_px:
                blobs.append((float(np.abs(D[mm_]).sum()), sign, mm_))
    blobs.sort(key=lambda b: -b[0])
    blobs = blobs[:k_blobs]
    if not blobs:
        return [], dict(blobs=0, cands=0)
    out, skipped, info = [], 0, []
    def _ok(a):
        nonlocal skipped
        if avoid is not None and np.asarray(avoid).any():
            fp = fb.footprint(g, a)
            if (fp & np.asarray(avoid, bool)).sum() > 0.02 * max(fp.sum(), 1):
                skipped += 1
                return False
        return True
    for tot_, sign, m in blobs:
        w = np.abs(D[m])
        cy, cx = float((rr[m] * w).sum() / w.sum()), float((cc_[m] * w).sum() / w.sum())
        r_b = math.sqrt(m.sum() / math.pi)
        info.append(dict(sign="채움(+)" if sign > 0 else "눌림(-)", px=int(m.sum()), mean_D_mm=round(float(sign * D[m].mean()), 2), center_px=(round(cy), round(cx))))
        xc, yc = cc.pix_to_world(g, cy, cx)
        starts = []
        rho = r_b + R_tool_px * 1.15
        pts = []
        for k in range(8):
            ph = 2 * math.pi * k / 8
            py, px = cy + rho * math.sin(ph), cx + rho * math.cos(ph)
            if 0 <= py < SIZE and 0 <= px < SIZE:
                y0, y1, x0, x1 = int(max(py - R_tool_px, 0)), int(min(py + R_tool_px + 1, SIZE)), int(max(px - R_tool_px, 0)), int(min(px + R_tool_px + 1, SIZE))
                pts.append((float(h[y0:y1, x0:x1].mean()), py, px))
        pts.sort(reverse=True)
        for _, py, px in pts[:keep_angles]:
            xs, ys = cc.pix_to_world(g, py, px)
            starts.append((xs, ys, math.atan2(yc - ys, xc - xs)))
        for k in range(4):
            th = k * math.pi / 4
            half = 0.5 * 0.008                                             # 중심을 지나가도록 시작점을 중심에서 L/2(대표 8mm) 뒤로
            starts.append((xc - half * math.cos(th), yc - half * math.sin(th), th))
        for xs, ys, th in starts:
            for L in lengths_mm:
                for d in depths_mm:
                    a = clamp([xs, ys, th, L * 1e-3, d * 1e-3])
                    if _ok(a):
                        out.append(a)
    seen, uniq = set(), []
    for a in out:
        key = tuple(round(float(v), 6) for v in a)
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq, dict(blobs=info, cands=len(uniq), skipped_protected=skipped)


def level_adjust_region_targets(h, names, masks, targets, requested, px_mm, tool_r_px, ring_tools=1.2, depth_mm=0.5):
    """영역 목표의 높이 자동 조정: 베이스라인(vs_representation.adjust_target_to_state_gpu)이 목표를 α·목표+β 로 점토 총량이 보존되게 맞추는 것과 같은 원리를
    국소 보상의 영역 목표에 적용한다. 대상·영역 이름에 의존하지 않는다.
    점토는 국소적으로만 옮길 수 있으므로 공급 = 요청 영역들이 내놓을 점토(현재 > 목표인 곳 전부) + 요청 영역 둘레(ring_tools x 도구 반경 폭)에서 눌러 얻을 점토(화소당 depth_mm 까지).
    둘레에는 어떤 영역(요청하지 않은 영역, 보호 영역 포함)도 속하지 않은 곳만 센다. 수요 = 요청 영역들의 부족분(목표 > 현재). 수요가 공급보다 크면 부족분 목표를 같은 비율 alpha 로 줄인다:
    목표_조정 = 현재 + alpha x (목표 - 현재)  (부족분 화소에만. 깎을 곳은 그대로). alpha = min(1, 공급/수요).
    반환: (조정한 목표 dict, 요약 dict). 요청 영역이 아닌 영역은 건드리지 않는다."""
    from scipy import ndimage
    pix = px_mm ** 2
    req = [n for n in requested if n in masks]
    need = sum(float(np.clip(targets[n] - h, 0, None)[masks[n]].sum()) * 1000.0 * pix for n in req)
    cut = sum(float(np.clip(h - targets[n], 0, None)[masks[n]].sum()) * 1000.0 * pix for n in req)
    allm = np.zeros((SIZE, SIZE), bool)
    for n in names:
        allm |= np.asarray(masks[n], bool)
    reqm = np.zeros((SIZE, SIZE), bool)
    for n in req:
        reqm |= np.asarray(masks[n], bool)
    ring = ndimage.binary_dilation(reqm, iterations=int(round(ring_tools * tool_r_px))) & ~allm
    ring_vol = float(ring.sum()) * pix * depth_mm
    supply = cut + ring_vol
    alpha = 1.0 if need <= 0 else float(min(1.0, supply / need))
    out = dict(targets)
    for n in req:
        t = np.asarray(targets[n], dtype=np.float64)
        m = np.asarray(masks[n], bool)
        adj = t.copy()
        dfc = (t > h) & m
        adj[dfc] = h[dfc] + alpha * (t - h)[dfc]
        out[n] = adj.astype(targets[n].dtype) if hasattr(targets[n], "dtype") else adj
    return out, dict(need_mm3=round(need), cut_mm3=round(cut), ring_mm3=round(ring_vol), supply_mm3=round(supply), alpha=round(alpha, 3), regions=req)


def opposite_press(a, center, g, clamp, avoid=None):
    """양옆 짝: 구덩이 중심(center, 세계 좌표 m)을 사이에 두고 반대편에서 같은 길이·깊이로 중심 쪽으로 누르는 행동. 구덩이 양옆을 눌러 가운데로 점토를 몰아 솟게 한다.
    avoid(보호 영역 마스크)와 겹치면 None."""
    import first_beam as fb
    x, y, th, L, d = [float(v) for v in a]
    xc, yc = center
    x2, y2 = 2.0 * xc - x, 2.0 * yc - y
    th2 = math.atan2(yc - y2, xc - x2)
    b = clamp([x2, y2, th2, L, d])
    if avoid is not None and np.asarray(avoid).any():
        fp = fb.footprint(g, b)
        if (fp & np.asarray(avoid, bool)).sum() > 0.02 * max(fp.sum(), 1):
            return None
    return b
