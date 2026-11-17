"""'눈으로 원하는 height map 과 같은가'를 프로젝트 원래 손실(L3D 형태, Lviz 외형)로 잰다. 평균 MAE 가 아니다.

  L3D  = mean (H - H*)^2                  형태(깊이) 오차
  Lviz = mean |grad H - grad H*|^2        외형(음영·기울기) 오차   (중앙차분, 화소 단위)
두 값을 '아무것도 안 했을 때(start)'의 값으로 나눈 비율로 보고한다. 1.0 = 변화 없음, <1 = 목표에 가까워짐, >1 = 악화.
  full : 지도 전체 (홈 밖이 망가진 것도 모두 반영 -> 눈에 보이는 부수 피해가 그대로 점수에 들어간다)
  win  : 홈 + 도구 반경 2배(기본 45화소) 이내의 국소 창 (편집한 곳이 목표 모양에 가까워졌는지)
  edge : 홈 가장자리 띠(마스크 경계 +-3화소)에서 경사 크기 |grad H| 가 목표와 같은가 (테두리 선명도 오차)
"""
import numpy as np
from scipy import ndimage


def _grad(h):
    return (h[1:-1, 2:] - h[1:-1, :-2]) / 2.0, (h[2:, 1:-1] - h[:-2, 1:-1]) / 2.0


class VisualScorer:
    def __init__(self, h_star, h_start, m_edit, win_px=45):
        self.h_star = h_star.astype(np.float64)
        self.me = m_edit.astype(bool)
        self.win = ndimage.binary_dilation(self.me, iterations=win_px)
        self.gs = _grad(self.h_star)
        self.band = ndimage.binary_dilation(self.me, iterations=3) & ~ndimage.binary_erosion(self.me, iterations=3)
        self.gmag_star = np.sqrt(self.gs[0] ** 2 + self.gs[1] ** 2)
        self.base = self.raw(h_start)

    def raw(self, h):
        h = h.astype(np.float64)
        d2 = (h - self.h_star) ** 2
        gx, gy = _grad(h)
        gd = (gx - self.gs[0]) ** 2 + (gy - self.gs[1]) ** 2
        wi = self.win[1:-1, 1:-1]
        bi = self.band[1:-1, 1:-1]
        gmag = np.sqrt(gx ** 2 + gy ** 2)
        edge = (gmag - self.gmag_star) ** 2          # 테두리(홈 가장자리 띠)에서 경사의 크기가 목표와 같은가 = 선명도
        return dict(l3d_full=float(d2.mean()), lviz_full=float(gd.mean()),
                    l3d_win=float(d2[self.win].mean()), lviz_win=float(gd[wi].mean()),
                    edge_win=float(edge[bi].mean()))

    def ratios(self, h):
        r = self.raw(h)
        return {k: r[k] / max(self.base[k], 1e-30) for k in r}

    def summary(self, h, h_start):
        r = self.ratios(h)
        dh = (h_start.astype(np.float64) - h.astype(np.float64))
        r["deepen_mm"] = float(dh[self.me].mean() * 1000)                 # 편집영역이 평균 몇 mm 깊어졌나 (목표 ~2.35)
        r["outside_win_max_mm"] = float(np.abs(dh)[~self.win].max() * 1000)  # 국소 창 밖에서 가장 크게 변한 곳
        r["win_max_mm"] = float(np.abs(dh)[self.win].max() * 1000)
        return r
