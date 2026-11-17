"""과거 반사실 국소 보상(cf_compensate / cf_edit / build_regions)이 공통으로 쓰는 일반 도구. 대상(A, 고양이, 하트, ...)과 시점에 의존하지 않는다.

  n_actions        이력 폴더의 행동 수
  make_view        이력의 앞쪽 K개 행동만 보이는 읽기 전용 폴더(심볼릭 링크). '조형 K번째 상태'를 현재로 삼을 때 쓴다.
  state_height     그 폴더의 마지막 행동 직후 높이맵(m)
  load_target      조형 때 실제로 쓴 목표(target_used_512_m.npy) 또는 새 목표를 같은 변환으로 맞춘 것
  geometry         픽셀-세계 좌표 변환과 도구 반경(px): 이력 manifest 에서 읽는다
  action_bounds    행동 범위. 플래너가 실제로 쓰는 안전 범위(vs_planner.SAFE_PLAN_*)를 그대로 가져온다
"""
import re
from pathlib import Path

import numpy as np

SIZE = 512
_ACTION_DIR = re.compile(r"^action_(\d{6})$")


def n_actions(root):
    return len([d for d in Path(root).iterdir() if d.is_dir() and _ACTION_DIR.match(d.name)])


def make_view(history_dir, upto, out_dir):
    """앞쪽 upto 개 행동만 있는 이력처럼 보이는 폴더를 만든다(심볼릭 링크, 원본은 읽기만 한다).

    upto 가 None 이거나 전체 행동 수 이상이면 원본 폴더를 그대로 돌려준다.
    """
    root = Path(history_dir).resolve()
    n = n_actions(root)
    if upto is None or upto >= n:
        return root
    if upto < 1:
        raise ValueError(f"--upto 는 1 이상이어야 합니다: {upto}")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(root.iterdir()):
        m = _ACTION_DIR.match(p.name)
        if m and int(m.group(1)) > upto:
            continue
        if p.name.startswith("action_") and not m:   # .incomplete 같은 미완성 폴더
            continue
        link = out / p.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p)
    return out


def state_height(root):
    n = n_actions(root)
    return np.load(Path(root) / f"action_{n:06d}" / "transition.npz")["after_m"].astype(np.float32)


def load_target(root, target_npy=None):
    """조형 때의 목표(기본) 또는 새 목표. 새 목표(512x512, m)에는 조형 때의 XY 배율/alpha/beta/경계 변환을 똑같이 적용한다."""
    root = Path(root)
    if target_npy is None:
        return np.load(root / "target_used_512_m.npy").astype(np.float32)
    from vs_history import apply_saved_target_transform
    return apply_saved_target_transform(np.load(target_npy).astype(np.float32), root).numpy().astype(np.float32)


def geometry(root):
    """이력 manifest 에서 좌표 변환·도구 크기를 읽는다. px_mm: 1 화소의 mm, tool_r_px: 도구 반경(화소)."""
    import json
    import first_beam as fb
    g = fb.map_geometry(json.loads((Path(root) / "manifest.json").read_text()))
    g["px_mm"] = 1000.0 / g["px_per_m_x"]
    g["tool_r_px"] = g["tool_r"] * g["px_per_m_x"]
    return g


def pix_to_world(g, row, col):
    return g["x0"] + (col + 0.5) / g["px_per_m_x"], g["y0"] + (row + 0.5) / g["px_per_m_y"]


def band_sigma(tool_r_px, tools=0.4):
    """'눈에 보이는 형태'를 재는 흐림 크기(화소). 도구 반경의 tools 배(기본 0.4 = 점토 입자 크기 정도).
    이보다 작은 점토 입자 질감은 걸러지고, 도구 크기 이상의 형태만 비교된다. 대상이 바뀌어도 도구 반경 기준이라 그대로 쓴다.
    (원본 픽셀 단위 Lviz 는 조형이 진행될수록 입자 질감 때문에 커져서 눈의 판단과 반대로 움직인다: A 이력 1->190 에서 0.151->0.303)"""
    return max(float(tool_r_px) * float(tools), 0.0)


def appearance_pre(masks, targets, sigma_px, h_ref=None, pit_sigma_px=0.0):
    """영역별 형태 오차 계산에 필요한 목표 쪽 값을 미리 구한다 (후보를 수백 번 채점하므로 한 번만).

    모두 sigma_px 로 흐린 지도에서 잰다:
      형태     |blur(H) - blur(H*)|          영역 안 평균 (mm)
      기울기   |grad blur(H) - grad blur(H*)|  영역 안 평균 (mm / 도구 반경)   외형(음영) 차이, 프로젝트 Lviz 의 흐린 버전
      테두리   | |grad blur(H)| - |grad blur(H*)| |  영역 가장자리 띠(마스크 경계 +-3화소)  선명도
    """
    from scipy import ndimage
    pre = {}
    for n, m in masks.items():
        tb = np.asarray(targets[n], dtype=np.float64)
        if sigma_px > 0:
            tb = ndimage.gaussian_filter(tb, sigma_px)
        ty, tx = np.gradient(tb)
        band = ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_erosion(m, iterations=3)
        idx, idb = np.flatnonzero(m), np.flatnonzero(band)
        pre[n] = dict(idx=idx, tb=tb.ravel()[idx], tx=tx.ravel()[idx], ty=ty.ravel()[idx], idb=idb, tmag=np.hypot(tx, ty).ravel()[idb])
        if h_ref is not None and pit_sigma_px > 0:
            # 구덩이 측정용: 가벼운 블러(pit_sigma_px)에서 목표 방향을 정한다. 기준 상태 -> 목표가 올라가야 하면 +1, 내려가야 하면 -1, 0.1 mm 이내면 0(과행 없음).
            tp = ndimage.gaussian_filter(np.asarray(targets[n], dtype=np.float64), pit_sigma_px).ravel()[idx]
            rp = ndimage.gaussian_filter(np.asarray(h_ref, dtype=np.float64), pit_sigma_px).ravel()[idx]
            d = tp - rp
            pre[n]["pit"] = dict(sigma=float(pit_sigma_px), tb=tp, sgn=np.where(np.abs(d) < 1e-4, 0.0, np.sign(d)))
    return pre


def appearance(h, pre, tool_r_px, sigma_px):
    """{영역: (기울기 오차, 테두리 오차, 형태 오차, 꼬리 오차)}. 기울기·테두리 단위 'mm / 도구 반경' (= mm/화소 x 도구 반경 화소),
    형태·꼬리 오차 단위 mm. 모두 높이 오차(mm)와 비슷한 크기라 같은 척도에서 합칠 수 있다.
    꼬리 = '목표를 지나친 양'(목표 방향으로 목표보다 더 간 높이)이 가장 큰 5% 화소의 평균(mm). 가벼운 블러(pit_sigma)에서 잰다. 한 번 눌러 한 곳만 파인
    구덩이는 도구 반경 x0.4 블러(형태 오차)에서는 지워지지만(과행 상위5% 0.02~0.68mm) 가벼운 블러에서는 드러난다(0.30~1.70mm).
    appearance_pre 에 h_ref/pit_sigma_px 를 주지 않으면 0. 항 0~2 의 순서와 의미는 그대로라 인덱스로 읽는 코드는 영향이 없다."""
    from scipy import ndimage
    hb = np.asarray(h, dtype=np.float64)
    if sigma_px > 0:
        hb = ndimage.gaussian_filter(hb, sigma_px)
    gy, gx = np.gradient(hb)
    hr, gxr, gyr = hb.ravel(), gx.ravel(), gy.ravel()
    k = 1000.0 * tool_r_px
    out, pit_maps = {}, {}
    for n, p in pre.items():
        g = float(np.hypot(gxr[p["idx"]] - p["tx"], gyr[p["idx"]] - p["ty"]).mean() * k) if len(p["idx"]) else 0.0
        e = float(np.abs(np.hypot(gxr[p["idb"]], gyr[p["idb"]]) - p["tmag"]).mean() * k) if len(p["idb"]) else 0.0
        sh = float(np.abs(hr[p["idx"]] - p["tb"]).mean() * 1000.0) if len(p["idx"]) else 0.0
        tail = 0.0
        pit = p.get("pit")
        if pit is not None and len(p["idx"]):
            if pit["sigma"] not in pit_maps:
                pit_maps[pit["sigma"]] = ndimage.gaussian_filter(np.asarray(h, dtype=np.float64), pit["sigma"]).ravel()
            over = np.maximum(0.0, pit["sgn"] * (pit_maps[pit["sigma"]][p["idx"]] - pit["tb"])) * 1000.0
            k_t = max(1, int(np.ceil(0.05 * over.size)))
            tail = float(np.partition(over, over.size - k_t)[over.size - k_t:].mean())
        out[n] = (g, e, sh, tail)
    return out


# ---------------------------------------------------------------- 부조(볼록한 부분) + 그 근처만 보는 초점
def plate_level(root, target=None):
    """목표의 '판 높이'(부조가 얹힌 바닥). 이력 manifest 의 목표 변환(alpha x 배경 높이 + beta)에서 읽는다 -> 목표를 만든 파이프라인이 정한 값이라
    모양이 판보다 넓어도(예: 하트) 틀리지 않는다. manifest 가 없으면 None."""
    import json
    try:
        tt = json.loads((Path(root) / "manifest.json").read_text())["target_transform"]
        base = float(tt["alpha"]) * float(tt["spatial_background_m"]) + float(tt["beta_m"])
        return float(min(max(base, tt.get("clip_min_m", -1e9)), tt.get("clip_max_m", 1e9)))
    except Exception:
        return None


def relief_focus(target, tool_r_px, near_tools=1.5, thr_mm=None, rel_frac=0.15, min_thr_mm=0.5, frame_px=25, base=None, level_gap_tools=0.4):
    """목표에서 '만들려는 모양(부조: 볼록, 음각: 오목)'과 그 근처만 골라낸다. 배경(판의 평평한 곳, 가장자리, 멀리 쌓인 점토)은 점수에서 뺀다.
    대상(A, 하트, 고양이, ...)에 의존하지 않는다. 규칙은 목표 높이맵(과 그 판 높이)에서만 나온다.

      판 높이 base   : 주어지면 그 값(plate_level 이 이력에서 읽음). 없으면 목표 가장자리 안쪽 frame_px 바깥쪽(테두리 제외) 높이의 최빈값(0.1mm 구간).
      모양 relief    : |목표 - base| >= thr 인 덩어리. thr = max(min_thr_mm, rel_frac x 모양 높이(상위 0.5%)), 즉 낮은 부조는 낮은 기준, 높은 부조는 높은 기준.
                       맵 가장자리 frame_px 안(경계 처리로 생긴 띠, 플래너 작업 범위 밖)과 도구 한 발자국보다 작은 조각은 뺀다.
                       판 가장자리에 걸친 모양은 작업 영역 안쪽 부분만 남는다.
      초점 focus     : 모양에서 near_tools x 도구 반경 이내(= 모양 + 그 근처).
      근처 띠 band   : 초점 중 모양 바깥(모양 경사면 level_gap_tools x 도구 반경은 제외). 현재 점토의 '국소 바닥 높이'를 재는 곳이다.
      테두리 edge    : 모양 경계 +-3화소. 선명도를 재는 곳.
    """
    from scipy import ndimage
    t = np.asarray(target, dtype=np.float64)
    S = t.shape[0]
    if base is None:
        inner = t[frame_px:S - frame_px, frame_px:S - frame_px] * 1000.0
        edges = np.arange(inner.min(), inner.max() + 0.1, 0.1)
        hist, edges = np.histogram(inner, bins=edges)
        base = (edges[int(np.argmax(hist))] + 0.05) / 1000.0
    dev = t - base
    di = dev[frame_px:S - frame_px, frame_px:S - frame_px]
    dis = ndimage.gaussian_filter(di, 2.0)          # 한두 화소짜리 튐은 모양으로 보지 않는다
    peak = float(max(dis.max(), -dis.min()))
    if peak < min_thr_mm * 1e-3:
        raise ValueError(f"목표에 판({base * 1000:.1f}mm)과 다른 모양이 없습니다(최대 {peak * 1000:.2f}mm). 평평한 목표는 초점을 정할 수 없습니다.")
    thr = (thr_mm * 1e-3) if thr_mm is not None else max(min_thr_mm * 1e-3, rel_frac * peak)
    interior = np.zeros(t.shape, bool)
    interior[frame_px:S - frame_px, frame_px:S - frame_px] = True      # 작업 영역 밖(경계 처리로 생긴 테두리 띠)은 모양으로 보지 않는다
    rel = (np.abs(dev) >= thr) & interior
    lab, n = ndimage.label(rel)
    if n:
        sizes = ndimage.sum(rel, lab, range(1, n + 1))
        rel = np.isin(lab, [i + 1 for i in range(n) if sizes[i] >= np.pi * float(tool_r_px) ** 2])      # 도구 한 발자국보다 작은 조각 제외
    if not rel.any():
        raise ValueError("목표의 모양 덩어리를 찾지 못했습니다(너무 작거나 판 테두리 띠뿐). thr_mm / base 를 직접 주세요.")
    near_px = max(int(round(near_tools * float(tool_r_px))), 1)
    dist = ndimage.distance_transform_edt(~rel)
    focus = (dist <= near_px) & interior
    band = focus & (dist > level_gap_tools * float(tool_r_px))
    ring = ndimage.binary_dilation(rel, iterations=3) & ~ndimage.binary_erosion(rel, iterations=3)
    return dict(relief=rel, focus=focus, band=band, edge=ring, base=float(base), thr=float(thr), near_px=near_px, peak_mm=peak * 1000.0,
                raised_frac=float((dev[rel] > 0).mean()))


def focus_pre(fp, target, sigma_px):
    """초점 점수에 필요한 목표 쪽 값을 미리 구한다."""
    from scipy import ndimage
    t = np.asarray(target, dtype=np.float64)
    tb = ndimage.gaussian_filter(t, sigma_px) if sigma_px > 0 else t
    ty, tx = np.gradient(tb)
    idf, idb, idr, idband = (np.flatnonzero(fp[k]) for k in ("focus", "edge", "relief", "band"))
    return dict(idf=idf, idb=idb, idr=idr, idband=idband, t_raw=t.ravel()[idf], t_raw_r=t.ravel()[idr], tb=tb.ravel()[idf],
                tx=tx.ravel()[idf], ty=ty.ravel()[idf], tmag=np.hypot(tx, ty).ravel()[idb], base=fp["base"], target=t)


def local_level(h, pre, fp=None):
    """현재 점토의 국소 바닥 높이(근처 띠의 중앙값) - 목표의 기준 높이. 판 전체가 가라앉거나 부풀어도 부조 모양만 보게 한다."""
    return float(np.median(np.asarray(h).ravel()[pre["idband"]]) - pre["base"])


def focus_scores(h, pre, tool_r_px, sigma_px, level="local"):
    """부조 + 근처에서만 잰 점수. 낮을수록 목표 부조에 가깝다.
      shape  흐린 높이 오차(mm)       기울기 grad  외형(음영) 차이(mm/도구 반경)       테두리 edge  경계 선명도 차이(mm/도구 반경)
      mae    원본 높이 MAE(mm, 초점)   mae_relief  원본 높이 MAE(mm, 부조 안만)
      l3d, lviz  프로젝트 L3D(높이 차 제곱 평균, m^2)와 Lviz(Sobel 기울기 차 제곱 평균) 를 '초점 안에서만' 평균한 것
    level='local' 이면 목표를 현재의 국소 바닥에 맞춘 뒤 비교(부조 높이를 본다), 'none' 이면 절대 높이 그대로.
    """
    from scipy import ndimage
    h = np.asarray(h, dtype=np.float64)
    c = local_level(h, pre) if level == "local" else 0.0
    hb = ndimage.gaussian_filter(h, sigma_px) if sigma_px > 0 else h
    gy, gx = np.gradient(hb)
    k = 1000.0 * float(tool_r_px)
    shape = float(np.abs(hb.ravel()[pre["idf"]] - (pre["tb"] + c)).mean() * 1000.0)
    grad = float(np.hypot(gx.ravel()[pre["idf"]] - pre["tx"], gy.ravel()[pre["idf"]] - pre["ty"]).mean() * k)
    edge = float(np.abs(np.hypot(gx.ravel()[pre["idb"]], gy.ravel()[pre["idb"]]) - pre["tmag"]).mean() * k)
    d = h.ravel()[pre["idf"]] - (pre["t_raw"] + c)
    mae = float(np.abs(d).mean() * 1000.0)
    mae_r = float(np.abs(h.ravel()[pre["idr"]] - (pre["t_raw_r"] + c)).mean() * 1000.0)
    return dict(shape=shape, grad=grad, edge=edge, mae=mae, mae_relief=mae_r, level_mm=c * 1000.0, l3d=float((d ** 2).mean()))


def focus_lviz(h, pre, fp):
    """프로젝트 vs_representation.spatial_gradient(Sobel/(8*pixel))로 잰 Lviz 를 초점 안에서만 평균 (원본 픽셀, 입자 질감 포함)."""
    import torch
    from vs_representation import spatial_gradient
    a = torch.from_numpy(np.asarray(h, dtype=np.float32))[None, None]
    b = torch.from_numpy(np.asarray(pre["target"], dtype=np.float32))[None, None]
    pgx, pgy = spatial_gradient(a)
    tgx, tgy = spatial_gradient(b)
    e = ((pgx - tgx) ** 2 + (pgy - tgy) ** 2)[0, 0].numpy()
    return float(e.ravel()[pre["idf"]].mean()), float(e.mean())


def original_target_goals(rs, root, h_ref, target_npy=None):
    """'원래 목표' 기준의 영역별 목표 높이맵 {영역: 512x512 m}. 조형 때 목표(또는 --target-npy)의 판 높이를 기준 상태(h_ref)의 판 높이에 맞춘 것.
    요청 기준 목표(영역 편집 목표: 더 파기, 얇게, 선명하게 ...)와 같은 점수식으로 나란히 재서, 요청과 원래 목표가 서로 반대 방향인 영역을 드러낸다."""
    from region_set import border_median
    t = load_target(root, target_npy).astype(np.float64)
    goal = t - border_median(t) + border_median(np.asarray(h_ref, dtype=np.float64))
    return {n: goal.copy() for n in rs.names}


def reach_mask(g, bounds=None):
    """도구가 닿을 수 있는 화소(bool 512x512): 도구 중심이 행동 범위의 x,y 상자 안에 있을 때 도구 반경(g['tool_r']) 이내인 곳.

    대상(A, 하트, 고양이, ...)이 바뀌면 영역이 이 범위 밖(판 가장자리)에 놓일 수 있다. 그런 곳은 어떤 후보 행동도 닿지 못해서
    라운드 예산만 쓰고 확정이 없다. 범위는 이력을 만든 플래너의 안전 범위(action_bounds)와 같다.
    """
    b = bounds if bounds is not None else action_bounds()
    x0, x1, y0, y1 = b[:4]
    rows, cols = np.mgrid[0:512, 0:512]
    x, y = pix_to_world(g, rows, cols)
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
    return np.hypot(dx, dy) <= g["tool_r"]


def reach_fraction(mask, g, bounds=None, rmask=None):
    """영역 마스크 중 도구가 닿을 수 있는 화소의 비율 (0~1). 빈 마스크는 0."""
    m = np.asarray(mask) > 0
    if not m.any():
        return 0.0
    r = rmask if rmask is not None else reach_mask(g, bounds)
    return float(r[m].mean())


def action_bounds(depth_min_mm=2.0):
    """(x_min, x_max, y_min, y_max, L_min, L_max, d_min, d_max), 단위 m.

    위치·길이·깊이 상한은 플래너의 안전 범위. 깊이 하한만 반사실 탐색용으로 더 낮춘다(깊이를 줄인 변형을 보려고).
    """
    import vs_config as c
    import vs_planner as p
    return (p.SAFE_PLAN_X_MIN, p.SAFE_PLAN_X_MAX, p.SAFE_PLAN_Y_MIN, p.SAFE_PLAN_Y_MAX,
            c.LENGTH_MIN, p.SAFE_PLAN_LENGTH_MAX, min(depth_min_mm * 1e-3, p.SAFE_PLAN_DEPTH_MIN), p.SAFE_PLAN_DEPTH_MAX)


def far_tolerance(before, after, far_mask, quantile=1.0):
    """영역에서 먼 곳(far_mask)의 부수 변화 한도를 '이 대상의 과거 기록'에서 잰다. 대상(A, 고양이, 하트, ...)에 의존하지 않는다.
    before/after: 과거 푸시마다의 전/후 높이맵(m). 반환: (한도 mm = 푸시별 먼 곳 최대 변화(mm)의 quantile 분위수, 푸시별 값 배열).
    quantile=1.0 이면 '조형 중 실제로 받아들여진 푸시 중 가장 큰 값'이다. 먼 곳이 비어 있으면 (inf, 빈 배열)."""
    far_mask = np.asarray(far_mask, dtype=bool)
    if not far_mask.any() or len(before) == 0:
        return float("inf"), np.zeros(0)
    vals = np.array([float(np.abs(np.asarray(after[t], dtype=np.float64) - np.asarray(before[t], dtype=np.float64))[far_mask].max() * 1000.0) for t in range(len(before))])
    return float(np.quantile(vals, min(max(float(quantile), 0.0), 1.0))), vals


def protect_tolerance(before, after, fps, mask, near_px, quantile=1.0, min_untouched=5):
    """고치지 않기로 한(보존할) 영역 mask 안에서 허용할 변화(mm)를 '이 대상의 과거 기록'에서 잰다. 대상에 의존하지 않는다.
    그 영역(+도구 반경 근처)에 닿지 않은 과거 푸시도 번짐 때문에 영역을 조금 바꾼다. 그 최대 변화의 quantile 분위수가 '닿지 않고도 생기는 정상 변화'다.
    보존 영역에서 새 푸시가 이보다 더 바꾸면 훼손으로 본다. 반환: (허용 mm 또는 None(닿지 않은 푸시가 min_untouched 개 미만), 닿지 않은 푸시별 최대 변화 배열)."""
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    zone = ndimage.binary_dilation(mask, iterations=int(near_px))
    vals = []
    for t in range(len(before)):
        if np.asarray(fps[t], dtype=bool)[zone].any():
            continue
        vals.append(float(np.abs(np.asarray(after[t], dtype=np.float64) - np.asarray(before[t], dtype=np.float64))[mask].max() * 1000.0))
    vals = np.array(vals)
    if len(vals) < min_untouched:
        return None, vals
    return float(np.quantile(vals, min(max(float(quantile), 0.0), 1.0))), vals
