"""저장된 점토 상태(MPM 입자 + 로봇)를 Genesis 뷰어로 복원해서 눈으로 본다.

상태를 2개 이상 주면 --hold 초마다 번갈아 보여 준다(수정 전 <-> 수정 후). 뷰어 창을 닫으면 종료한다.
복원 직후 높이맵이 기록과 맞는지(max 차이)를 출력한다. 뷰어를 켜려면 화면(DISPLAY)이 있어야 한다.
뷰어가 점토를 계속 그리려면 물리 step 이 필요해서 보는 동안 점토가 아주 조금 안정화(settle)될 수 있다. 저장 파일은 바꾸지 않는다.

사용 예 (수정 전 = 이력 190번 행동 직후, 수정 후 = 보정 10번째 행동 직후):
  python view_state.py --history-dir results/A_history200_20261114_060837/history \
      --state before=results/A_history200_20261114_060837/history/action_000190/post_state.pt \
      --state after=results/case_A/design/lc_w1.0_nobroyden/history/action_000010/post_state.pt --hold 15
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True, type=Path, help="물리 서명(signature) 확인용 원본 이력")
    ap.add_argument("--state", action="append", required=True, help="이름=post_state.pt 또는 pre_state.pt 경로")
    ap.add_argument("--hold", type=float, default=15.0, help="상태마다 보여 주는 시간(초)")
    ap.add_argument("--rounds", type=int, default=20, help="번갈아 보는 횟수. 0이면 창을 닫을 때까지")
    ap.add_argument("--no-viewer", action="store_true", help="창 없이 복원만 검증")
    a = ap.parse_args()
    root = a.history_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    details = manifest["signature"]["details"]
    os.environ["VS_MATERIAL"] = details["material"]
    os.environ["VS_TIP_CORRECTION"] = "1" if details["tip_correction"] else "0"
    os.environ["VS_SHOW_VIEWER"] = "0" if a.no_viewer else "1"
    import vs_simulator as sim
    from vs_history import GenesisStateAdapter, load_snapshot, physics_signature

    adapter = GenesisStateAdapter(sim)
    if physics_signature(sim) != manifest["signature"]:
        raise RuntimeError("저장 당시와 장면/코드/환경이 다릅니다. 이 상태는 복원할 수 없습니다.")
    states = []
    for s in a.state:
        name, path = s.split("=", 1)
        states.append((name, load_snapshot(Path(path).expanduser())))
    ref = states[0][1]["height"].numpy()
    for name, snap in states:
        adapter.restore(snap)
        err = float(np.abs(adapter.observe().numpy() - snap["height"].numpy()).max()) * 1000
        d = (snap["height"].numpy() - ref) * 1000
        print(f"[{name}] 복원 확인: 기록과 최대 {err:.4f}mm | 첫 상태 대비 최대 높이 변화 {np.abs(d).max():.2f}mm")
    if a.no_viewer:
        return
    print("뷰어를 보세요. 창을 닫으면 종료합니다.")
    i = 0
    while a.rounds == 0 or i < a.rounds * len(states):
        name, snap = states[i % len(states)]
        adapter.restore(snap)
        print(f"  -> {name}")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < a.hold:
            sim.initialize().step(force_visual=True)
        i += 1


if __name__ == "__main__":
    main()
