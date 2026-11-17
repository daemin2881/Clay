"""조형 이력의 K번째 상태에서 '과거 반사실 국소 보상'을 처음부터 끝까지 한 번에 실행한다. 대상(A, 고양이, 하트, ...)에 의존하지 않는다.

  1) 시점     : --upto K 면 조형 K번째 상태를 '현재', 앞쪽 K개 행동을 '과거'로 본다 (없으면 이력 전체)
  2) 영역     : --regions auto            목표와 현재의 차이에서 국소 오차 덩어리를 자동으로 찾는다(build_regions.py auto)
                --regions prompt          요청 문장(--prompt)을 VLM(Gemini)에 주고 SAM2 로 영역 마스크를 만든다(live_edit.py 의 영역 생성 단계 재사용).
                                          편집 종류는 --types (예: match_target,match_target,match_target = 원하는 목표를 따르게).
                                          API 키(CF_VLM_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)가 없으면 --regions-json 대역이 필요하다.
                --regions DIR             만들어 둔 영역 집합(build_regions.py masks, live_edit.py --stop-after spec 출력, 예전 형식)
  3) 보상     : (cf_compensate.py) 영역마다 ① 관련 과거 행동을 beam 으로 탐색 -> ② 재생 가능한 범위(--replay-len, 기본 직전 20개)의 상위 행동을
                Genesis 에서 바꿔 다시 재생해 '그때 달랐다면' 상태 C* 를 만들고 -> ③ 그것을 현재 점토에서 만드는 행동을 찾아 실제로 실행해 보고 되돌려서
                좋아지는 것만 확정한다. 터미널에 계산 과정이 나오고, --viewer 면 Genesis 창에서 확정 행동이 보인다. --no-cf-replay 로 ②를 끌 수 있다.
  4) 전/후    : 전(K번째 상태) / 목표 / 후를 높이맵·음영·차이로 한 장에 (before_after.py)

사용:
  python cf_edit.py --history-dir results/<run>/history --upto 80 --regions auto --out results/<name> [--viewer] [--workers 4]
  python cf_edit.py --history-dir H --regions results/<name>/regions --out results/<name2>
cf_edit.py 가 모르는 옵션은 cf_compensate.py 에 그대로 넘어간다 (예: --budget-min 30 --max-commits 5 --mode full).
영역 자동 탐색 옵션: --thr-mm --max-regions --within-roi --within MASK --detrend-mm --cap-mm --target-npy (build_regions.py auto 설명 참고)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import cf_common as cc

ROOT = Path(__file__).resolve().parent


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise SystemExit(f"실패(종료 코드 {rc}): {cmd[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", required=True, type=Path)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--regions", default="auto", help="auto | prompt | 영역 집합 폴더")
    ap.add_argument("--prompt", help="--regions prompt 일 때 요청 문장. 영역은 ';' 또는 줄바꿈으로 나눠 쓴다 (예: \"홈을 더 깊게; 왼쪽 다리를 더 얇게; 다리 받침을 선명하게\")")
    ap.add_argument("--types", help="--regions prompt 일 때 영역별 편집 종류를 순서대로 (match_target deepen deepen_widen trim connect thin sharpen)")
    ap.add_argument("--regions-json", type=Path, help="API 키가 없을 때 쓰는 VLM 대역(영역 목록 JSON, live_edit.py 형식)")
    ap.add_argument("--delta-mm", type=float, help="deepen 류 편집의 깊이(mm). 사람이 정한다")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-npy", type=Path, help="새 목표(512x512, m). 없으면 조형 때의 목표")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--names", default="", help="영역 집합에서 일부만 쓸 때 쉼표로 구분한 이름 (cf_compensate 와 전/후 그림에 모두 적용)")
    ap.add_argument("--skip-compensate", action="store_true", help="영역 만들기까지만 (확인용)")
    auto = ap.add_argument_group("영역 자동 탐색 (--regions auto)")
    auto.add_argument("--thr-mm", type=float)
    auto.add_argument("--max-regions", type=int)
    auto.add_argument("--min-area-mm2", type=float)
    auto.add_argument("--max-area-tools", type=float)
    auto.add_argument("--detrend-mm", type=float)
    auto.add_argument("--cap-mm", type=float)
    auto.add_argument("--within", type=Path)
    auto.add_argument("--within-roi", action="store_true")
    a, passthrough = ap.parse_known_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"새 폴더를 지정하세요(이미 내용이 있음): {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    hist = a.history_dir.resolve()
    n_all = cc.n_actions(hist)
    k = n_all if a.upto is None else min(a.upto, n_all)
    print("=" * 78 + f"\n[1] 시점: {hist.name} 의 {k}번째 상태 (이력 {n_all}개 중)\n" + "=" * 78, flush=True)

    # ---- 영역 ----
    if a.regions == "auto":
        print("=" * 78 + "\n[2] 영역 자동 탐색: 목표와 현재의 국소 차이에서 채울 곳/깎을 곳을 찾는다\n" + "=" * 78, flush=True)
        reg = a.out / "regions"
        cmd = [py, "build_regions.py", "auto", "--history-dir", hist, "--out", reg]
        if a.upto is not None:
            cmd += ["--upto", a.upto]
        if a.target_npy:
            cmd += ["--target-npy", a.target_npy]
        for flag, v in (("--thr-mm", a.thr_mm), ("--max-regions", a.max_regions), ("--min-area-mm2", a.min_area_mm2),
                        ("--max-area-tools", a.max_area_tools), ("--detrend-mm", a.detrend_mm), ("--cap-mm", a.cap_mm), ("--within", a.within)):
            if v is not None:
                cmd += [flag, v]
        if a.within_roi:
            cmd += ["--within-roi"]
        run(cmd)
    elif a.regions == "prompt":
        if not a.prompt:
            raise SystemExit("--regions prompt 에는 --prompt 가 필요합니다.")
        keys = ("CF_VLM_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
        if not any(os.environ.get(k) for k in keys) and not a.regions_json:
            raise SystemExit(f"VLM API 키({', '.join(keys)})가 없습니다. 키를 설정하거나, 영역을 직접 쓴 대역 JSON 을 --regions-json 으로 주세요. "
                             "(live_edit.py 기본 대역은 A 190번째 상태의 홈 전용이라 일부러 쓰지 않습니다.)")
        print("=" * 78 + "\n[2] 영역: 요청 문장 -> VLM -> SAM2 마스크 -> 편집 종류별 목표 (live_edit.py 의 영역 생성 단계)\n" + "=" * 78, flush=True)
        view0 = cc.make_view(hist, a.upto, a.out / "history_view")
        reg = a.out / "prompt"
        cmd = [py, "live_edit.py", "--history-dir", view0, "--prompt", a.prompt, "--stop-after", "spec", "--out-dir", reg]
        for flag, v in (("--regions-json", a.regions_json), ("--types", a.types), ("--delta-mm", a.delta_mm)):
            if v is not None:
                cmd += [flag, v]
        run(cmd)
    else:
        reg = Path(a.regions).resolve()
        print("=" * 78 + f"\n[2] 영역: 기존 영역 집합 사용 {reg}\n" + "=" * 78, flush=True)
    if a.skip_compensate:
        print(f"(--skip-compensate) 영역만 만들었습니다: {reg}/regions_preview.png")
        return

    # ---- 보상 ----
    print("=" * 78 + "\n[3] 과거 반사실 국소 보상 (후보를 Genesis 에서 실제로 실행해 보고 되돌려서 좋아지는 것만 확정)\n" + "=" * 78, flush=True)
    cf = a.out / "cf"
    cmd = [py, "cf_compensate.py", "--history-dir", hist, "--regions", reg, "--out", cf, "--workers", a.workers]
    if a.upto is not None:
        cmd += ["--upto", a.upto]
    if a.viewer:
        cmd += ["--viewer"]
    if a.names:
        cmd += ["--names", a.names]
    if a.target_npy:
        cmd += ["--target-npy", a.target_npy]
    run(cmd + passthrough)

    # ---- 전/후 ----
    print("=" * 78 + "\n[4] 전/후 비교\n" + "=" * 78, flush=True)
    view = cc.make_view(hist, a.upto, a.out / "history_view")
    np.save(a.out / "before_m.npy", cc.state_height(view))
    ba_extra = (["--names", a.names] if a.names else []) + ["--tool-r-px", cc.geometry(view)["tool_r_px"], "--history-dir", hist]
    if a.target_npy:
        ba_extra += ["--target-npy", a.target_npy]
    cf_stages = []                       # 과거를 바꿔 끝까지 재생한 세계(반사실 H_cf): 쓴 반사실이 있을 때만. 보상 결과가 이 세계에 얼마나 가까운지 눈으로 비교한다
    for p in sorted((cf / "cf_replay").glob("H_cf_*.npy")):
        cf_stages += ["--stage", f"Counterfactual world\\n(past action changed: {p.stem[5:]})={p}"]
    run([py, "before_after.py", "--regions", reg, "--ref-h", a.out / "before_m.npy", *ba_extra,
         "--stage", f"Before (state {k})={a.out / 'before_m.npy'}", *cf_stages, "--stage", f"After compensation\\n({hist.name[:24]})={cf / 'final_m.npy'}",
         "--out", a.out / "before_after", "--title", f"{hist.name} @ action {k}: local compensation before vs after"])
    print(f"\n산출물: {a.out}\n  cf/summary.json, cf/frames/ (확정할 때마다 1장), cf/states/ (확정 상태 스냅샷)\n  before_after/before_after.png, before_after/table.json")


if __name__ == "__main__":
    main()
