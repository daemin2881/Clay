"""편집 종류별 목표 높이맵 생성. VLM 은 '어디를 어떤 종류로 고치나'만 정하고, 높이(mm)는 여기서 규칙으로 계산한다.

  deepen / deepen_widen : 영역(deepen_widen 은 widen_px 만큼 넓힌 영역)을 delta_mm 만큼 낮춘다.   (홈을 더 깊게/크게)
  trim                  : 영역 안에서 주변 고리(ring)의 중앙값보다 높은 부분을 그 높이까지 깎는다.  (튀어나온 부분 정리)
  connect               : 영역 안에서 주변 고리의 75% 분위수보다 낮은 부분을 그 높이까지 채운다.   (끊어진 획 연결)
  match_target          : 영역 안을 '원하는 목표 height map'(reference)과 같게 만든다. 목표의 배경 높이를 현재 배경 높이에 맞춰 정렬한 뒤 사용한다.
                          (원래 A 목표나 사용자가 새로 준 목표. 사람이 고른 규칙이 아니라 목표 자체가 기준이다.)
  thin                  : 영역(튀어나온 획/다리)의 폭을 delta_mm(총 폭 감소)만큼 줄인다. 가장자리에서 안쪽으로 delta/2 띠를 주변 고리 높이까지 깎고
                          안쪽(core)은 현재 높이를 유지한다.                                             (다리를 더 얇게)
  sharpen               : 윤곽을 계단처럼 선명하게. 영역 안쪽은 안쪽 높이(core 중앙값)의 평평한 윗면으로, 경계는 좁은 feather 로 둔다.
                          윤곽이 뚜렷한 평평한 부분(받침, 판)에 맞는다.                                 (받침을 선명하게)
  groove                : 홈을 '깊고 조금 더 넓게, 안의 정리 안 된 점토는 정리'. 홈 바닥 = 지금 홈 높이의 중앙값 - delta_mm 의 평평한 면.
                          홈 안에서 바닥 위로 솟은 덩어리는 바닥까지 눌러 정리하고, 이미 더 깊은 곳은 그대로 둔다.
                          홈 가장자리는 GROOVE_WIDEN_MM(3 mm) 만큼 넓히며, 넓힌 띠는 바닥에서 바깥 끝의 현재 높이로 비스듬히 잇는다.   (홈 파기)
  groove_wide           : groove 에서 정리(바닥 평탄화)만 뺀 것. 홈 안을 delta_mm 만큼 더 깊게 하고 가장자리를 3 mm 넓힌다.         (홈을 깊고 조금 더 넓게)
  flatten               : 영역 안에서 솟은 점토를 영역 주변 고리의 낮은 쪽(25% 분위) 높이까지 눌러 정리한다. 이미 낮은 곳은 그대로.
                          trim 은 고리 중앙값을 쓰는데, 홈 안의 덩어리는 고리가 홈 벽(높은 획)이라 거의 깎이지 않아서 따로 둔다.   (홈 안 튀어나온 점토 정리)
  thin/sharpen 은 경계 띠가 바로 목표라서 경계 가중치(feather)를 기본 0/1 화소로 좁게 쓴다(다른 종류는 4 화소).
  모두 경계는 거리 변환으로 부드럽게 잇고(feather), 영역 밖은 정확히 현재 높이를 유지한다.
  변화량은 max_mm 로 제한한다(도구로 만들 수 없는 큰 변형을 요구하지 않기 위해).
"""
import numpy as np
from scipy import ndimage

from edit_spec import soft_mask

EDIT_TYPES = ("deepen", "deepen_widen", "trim", "connect", "match_target", "thin", "sharpen", "groove", "groove_wide", "flatten")
FEATHER_DEFAULT = {"thin": 0.0, "sharpen": 1.0}          # 그 외는 4 화소
GROOVE_WIDEN_MM = 3.0                                    # groove: 홈 가장자리를 넓히는 폭(한쪽, mm) = '조금 더 넓게'


def ring_of(mask, inner=2, outer=10):
    return ndimage.binary_dilation(mask, iterations=outer) & ~ndimage.binary_dilation(mask, iterations=inner)


def _bg(a):
    return float(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))


def _default_px_mm():
    import vs_config as c
    return 1000.0 * min(c.PIXEL_SIZE_X, c.PIXEL_SIZE_Y)


def thickness_mm(mask, px_mm=None):
    """영역의 두께(mm) = 안에 들어가는 가장 큰 원의 지름. thin 의 정도를 '두께의 몇 %'로 정할 때 쓴다."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return 0.0
    return float(2.0 * ndimage.distance_transform_edt(m).max() * (px_mm or _default_px_mm()))


def region_target(h_t, mask, edit_type, *, delta_mm=3.0, widen_px=6, feather_px=None, max_mm=6.0, reference=None, px_mm=None):
    """영역 하나의 목표 높이맵 (512x512, m)과 실제로 적용한 마스크를 돌려준다.
    delta_mm: deepen 류는 낮출 깊이, thin 은 줄일 총 폭(mm). px_mm: 화소 크기(mm), thin/sharpen 의 화소 환산용(없으면 vs_config)."""
    if edit_type not in EDIT_TYPES:
        raise ValueError(f"알 수 없는 편집 종류: {edit_type} (가능: {EDIT_TYPES})")
    if feather_px is None:
        feather_px = FEATHER_DEFAULT.get(edit_type, 4.0)
    h = h_t.astype(np.float64)
    m = mask.astype(bool)
    ring = ring_of(m)
    if edit_type in ("deepen", "deepen_widen"):
        if edit_type == "deepen_widen":
            m = ndimage.binary_dilation(m, iterations=int(widen_px))
        goal = h - delta_mm / 1000.0
    elif edit_type in ("groove", "groove_wide"):
        pm = px_mm or _default_px_mm()
        wpx = max(1, int(round(GROOVE_WIDEN_MM / pm)))
        m0 = m
        m = ndimage.binary_dilation(m0, iterations=wpx)                                        # 조금 더 넓게
        floor = float(np.median(h[m0])) - float(delta_mm) / 1000.0                             # 홈 바닥: 지금 홈 높이 중앙값보다 delta_mm 더 깊은 평평한 면
        t = np.clip(ndimage.distance_transform_edt(~m0) / wpx, 0.0, 1.0)                       # 넓힌 띠: 가장자리(0)에서 바닥, 바깥 끝(1)에서 현재 높이로
        inside = np.minimum(h, floor) if edit_type == "groove" else h - float(delta_mm) / 1000.0   # groove: 바닥까지 눌러 정리 / groove_wide: 정리 없이 delta_mm 만큼 더 깊게
        goal = np.where(m0, inside, np.minimum(h, floor + (h - floor) * t))
        max_mm = max(max_mm, 10.0)
    elif edit_type == "flatten":
        goal = np.minimum(h, np.percentile(h[ring], 25))                                       # 솟은 덩어리를 주변 고리의 낮은 쪽(25% 분위) 높이까지 눌러 정리
    elif edit_type == "trim":
        goal = np.minimum(h, np.median(h[ring]))
    elif edit_type == "match_target":
        if reference is None:
            raise ValueError("match_target 은 reference(목표 높이맵)가 필요합니다.")
        ref = reference.astype(np.float64)
        goal = ref - _bg(ref) + _bg(h)          # 배경 높이를 현재에 정렬
        max_mm = max(max_mm, 8.0)
    elif edit_type == "thin":
        pm = px_mm or _default_px_mm()
        shave = max(1, int(round(0.5 * float(delta_mm) / pm)))              # 한쪽 가장자리에서 깎을 폭(화소)
        core = ndimage.binary_erosion(m, iterations=shave)
        if not core.any():
            raise ValueError(f"thin: 영역이 얇아서(두께 {thickness_mm(m, pm):.1f} mm) {delta_mm:g} mm 를 줄이면 남는 것이 없습니다. delta_mm 를 줄이세요.")
        band = m & ~core
        goal = h.copy()
        goal[band] = np.minimum(h[band], np.median(h[ring]))               # 가장자리 띠를 주변 높이까지 (이미 낮으면 그대로)
    elif edit_type == "sharpen":
        pm = px_mm or _default_px_mm()
        core = ndimage.binary_erosion(m, iterations=max(1, int(round(2.0 / pm))))
        if not core.any():
            core = m
        goal = np.full_like(h, np.median(h[core]))                          # 안쪽은 평평한 윗면으로
    else:  # connect
        goal = np.maximum(h, np.percentile(h[ring], 75))
    w = soft_mask(m, feather_px).astype(np.float64)
    change = np.clip((goal - h) * w, -max_mm / 1000.0, max_mm / 1000.0)
    return (h + change).astype(np.float32), m
