"""선택한 반사실(과거 #186 건너뛰기)의 MAE·형태·외형·테두리 오차를 190회 A 와 같은 기준으로 계산한다.
사용: python chosen_cf_numbers.py > results/case_A/design/chosen_cf_numbers.txt
"""
import numpy as np
from visual_score import VisualScorer, _grad
from region_edit import _bg

D = "results/case_A/design/"
ht = np.load(D + "cf_specs/r1/h_t.npy").astype(np.float64)
rev = np.load(D + "cf_combined_186/combined.npy").astype(np.float64)
ref = np.load("results/A_history200_20261114_060837/history/target_used_512_m.npy").astype(np.float64)
ref_al = ref - _bg(ref) + _bg(ht)
fp = ref > _bg(ref) + 5e-4
masks = [np.load(D + f"cf_specs/r{i}/m_edit.npy") > 0 for i in range(3)]
names = ["왼쪽 튀어나온 부분", "끊어진 왼쪽 다리", "가운데 홈"]
mae = lambda h, t, m=None: float(np.abs(h - t)[m if m is not None else slice(None)].mean()) * 1000
print("[MAE mm] 원래 A 목표(배경 높이 정렬) 대비: 190회 A -> 과거(#186)를 고쳐 재생한 A")
print(f"전체 지도 {mae(ht, ref_al):.3f} -> {mae(rev, ref_al):.3f}")
print(f"A 글자 영역 {mae(ht, ref_al, fp):.3f} -> {mae(rev, ref_al, fp):.3f}")
for n, m in zip(names, masks):
    print(f"{n} {mae(ht, ref_al, m):.3f} -> {mae(rev, ref_al, m):.3f}")
spec = np.load(D + "cf_specs/r1/h_star.npy")
sc = VisualScorer(spec, ht, masks[1])
r = sc.ratios(rev)
print("[끊어진 다리 영역 목표 H*_gap 대비 비율, 190회 A = 1.00] " + ", ".join(f"{k}={v:.2f}" for k, v in r.items()))
hs = spec.astype(np.float64) * 1000
gx, gy = _grad(hs)
win = sc.win
for nm, h in (("190회 A", ht), ("과거수정 A", rev)):
    hm = h * 1000
    g2 = _grad(hm)
    gref = np.sqrt((gx ** 2 + gy ** 2)[win[1:-1, 1:-1]].mean())
    print(f"  {nm}: 영역 MAE {mae(h, spec.astype(np.float64), masks[1]):.2f}mm, 창 RMS {np.sqrt(((hm - hs) ** 2)[win].mean()):.2f}mm, 창 기울기오차 {100 * np.sqrt(((g2[0] - gx) ** 2 + (g2[1] - gy) ** 2)[win[1:-1, 1:-1]].mean()) / gref:.0f}%")
S = "results/live/multi_run2_target/spec/"
hall = np.load(S + "h_star.npy")
mall = np.load(S + "m_edit.npy") > 0
ra = VisualScorer(hall, ht, mall).ratios(rev)
print("[세 영역 합친 목표 H*_all 대비 비율, 190회 A = 1.00] " + ", ".join(f"{k}={v:.2f}" for k, v in ra.items()))
print(f"수정이 바꾼 양: 전체 평균 {mae(rev, ht):.4f}mm, 0.5mm 초과 화소 {int((np.abs(rev - ht) * 1000 > 0.5).sum())}개, 최대 {np.abs(rev - ht).max() * 1000:.1f}mm")
