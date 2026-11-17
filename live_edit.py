"""조형이 끝난 상태(예: 190행동 후 A)에서 프롬프트 하나로 국소 수정까지 이어서 실행한다.

  1) 프롬프트  : --prompt 또는 터미널 입력. 예) "A의 가운데 홈을 더 깊게 파고 싶다"
  2) VLM       : 최종 높이맵을 색 이미지로 그려 VLM(Gemini)에 준다 -> 박스·양성점·음성점. API 키가 없으면 --vlm-json 의 대역을 쓰고 화면에 '대역'이라고 표시한다.
  3) SAM 2     : 박스·점으로 홈 마스크 -> M_edit/M_boundary/M_preserve, 목표 H* (깊이 delta 는 사람이 고정, VLM 은 정하지 않는다)
  4) 과거 기록 : 행동 점수 + 계층 beam(attribution.py) -> 후보 -> Genesis 롤백 반사실(rollback_cf.py) -> 신뢰 기준 통과하면 C*, 아니면 abstain
  5) 현재 보정 : 자코비안 + Adam + Broyden 으로 한 행동씩 실행(local_correct.py). --viewer 면 Genesis 뷰어에서 점토가 눌리는 것을 보고,
                 --watch 면 목표/현재/차이 높이맵 창이 반복마다 갱신된다. 계산 과정(beam, 예측 이득, 실제 이득)은 이 터미널에 출력된다.
  6) 결과      : 목표와의 눈에 보이는 오차 표(L3D 형태, Lviz 외형, 테두리; 아무것도 안 함 = 1.00)와 음영 비교 그림

이력 폴더는 읽기만 한다. 모든 산출물은 --out-dir 에 새로 만든다.
사용: python live_edit.py --prompt "A의 가운데 홈을 더 깊게" --viewer --watch
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SAM2_ROOT = Path("/home/airlab/research/clay_genesis/sam2")
# (대상 전용 기본값은 두지 않는다: 이력과 VLM 대역은 항상 --history-dir / --vlm-json / --regions-json 으로 명시한다. 예전 기본은 A 190번째 상태의 홈 전용이었다.)
DEFAULT_CK = ROOT / "checkpoints/param2deform_medium_paper_planner_range_deep150.pt"
KEY_ENVS = ("CF_VLM_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
# VLM(sam2/vlm_roi_prompt.py)의 edit_type 어휘 -> region_edit.py 편집 종류. 어휘에 없거나 reshape/round 처럼 대응이 없으면 예전처럼 deepen.
# round/reshape(예: 뾰족한 귀 -> 둥근 귀)는 규칙으로 만들 수 없는 '새 모양'이라 사용자의 build_roi_edit_target.py(VLM target_shape/drag -> 새 목표 맵 H_edit)를 그대로 쓴다.
VLM_EDIT_TO_TYPE = {"lower": "deepen", "raise": "connect", "narrow": "thin", "sharpen": "sharpen", "widen": "deepen_widen", "round": "external", "reshape": "external"}


def banner(n, text):
    print(f"\n{'=' * 78}\n[{n}] {text}\n{'=' * 78}", flush=True)


def run(cmd, log, keep=None, env=None):
    """하위 프로세스를 실행하며 출력을 화면에 흘린다. keep 정규식이 있으면 그 줄만 화면에 보이고, 전체는 로그에 남긴다."""
    p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    with open(log, "a", encoding="utf-8") as f:
        for line in p.stdout:
            f.write(line)
            if keep is None or re.search(keep, line):
                print(line.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"실패(종료 코드 {rc}): {' '.join(str(c) for c in cmd[:3])} ... 로그: {log}")


def external_edit_target(h_t, mask, prompt_json, out_dir, py, log):
    """사용자의 build_roi_edit_target.py 로 새 목표 맵(H_edit)을 만든다 (round: 둥근 귀 타원 캡, reshape: 드래그). 반환: (목표 512x512 m, 적용 영역 bool).
    적용 영역 = SAM 마스크와 목표가 현재와 0.1 mm 이상 달라진 곳의 합집합(둥글게 만들면 마스크 밖으로 깎이거나 채워지는 곳이 생긴다)."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "h_base_in.npy", np.asarray(h_t, dtype=np.float32))
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(out_dir / "mask_in.png")       # 정수 마스크는 0/255 PNG 로 줘야 한다
    run([py, SAM2_ROOT / "build_roi_edit_target.py", "--heightmap", out_dir / "h_base_in.npy", "--mask", out_dir / "mask_in.png",
         "--prompt-json", prompt_json, "--output-dir", out_dir / "out"], log, keep=r"H_edit|ROI|round|오류|Error")
    tgt = np.load(out_dir / "out" / "h_edit_512_m.npy").astype(np.float32)
    return tgt, (np.abs(tgt - np.asarray(h_t, dtype=np.float32)) > 1e-4) | (np.asarray(mask) > 0)


def render_state_image(h, path):
    """높이맵을 turbo 색으로 칠한 1024x1024 이미지. 회색 음영 이미지에서는 SAM 2 가 홈을 못 잡았다(경험)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    from PIL import Image
    bm = float(np.median(np.concatenate([h[0], h[-1], h[:, 0], h[:, -1]])))
    z = np.clip((h - (bm - 0.010)) / 0.024, 0, 1)
    rgb = (cm.turbo(z)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.BICUBIC).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", type=Path, required=True, help="조형 이력 폴더 (visual_sculpting.py --history-dir 로 만든 것). 대상은 무엇이든 된다")
    ap.add_argument("--prompt", help="수정 요청. 생략하면 터미널에서 입력받는다.")
    ap.add_argument("--vlm-json", type=Path, help="API 키가 없을 때 쓰는 단일 영역 대역 프롬프트 JSON (이 대상·이 요청에 맞게 직접 만든 것)")
    ap.add_argument("--regions-json", type=Path, help="여러 영역(편집 종류 포함) 대역 프롬프트 JSON. 예: results/case_A/vlm/prompt_multi_A.json")
    ap.add_argument("--reference-target", type=Path, help="원하는 목표 height map(512x512, m). 편집 종류 match_target 이 이것을 따른다. 기본: 이력의 목표(원래 A)")
    ap.add_argument("--types", help="Gemini 사용 시 영역별 편집 종류를 순서대로 지정. 예: trim,connect,deepen_widen (그 밖에 thin=폭 줄이기, sharpen=윤곽 선명하게, external=round/reshape 를 build_roi_edit_target.py 로)")
    ap.add_argument("--delta-mm", type=float, default=3.0, help="사람이 정하는 수정 깊이. VLM 은 정하지 않는다.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CK)
    ap.add_argument("--lookback", type=int, default=20, help="과거 beam 이 볼 최근 행동 수 (재생을 믿을 수 있는 범위, 0 = 전체)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-rollback", action="store_true", help="Genesis 롤백 검증을 건너뛴다(C* 없이 진행)")
    ap.add_argument("--no-verify", action="store_true", help="실행 전 시뮬레이터 시도(롤백) 검증을 끈다. 기본은 켜짐: 눈에 보이는 오차가 줄어드는 행동만 확정")
    ap.add_argument("--verify-k", type=int, default=4)
    ap.add_argument("--viewer", action="store_true", help="Genesis 뷰어 창으로 점토가 눌리는 것을 실시간으로 본다")
    ap.add_argument("--watch", action="store_true", help="목표/현재/차이 높이맵 창을 띄운다")
    ap.add_argument("--stop-after", choices=["spec"], help="단계별 시험용: 목표(spec) 생성까지만 하고 종료")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()

    out = (a.out_dir or ROOT / "results/live" / ("edit_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"새 폴더를 지정하세요: {out}")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "live.log"
    hist = a.history_dir.resolve()
    last = len([d for d in hist.glob("action_??????") if d.is_dir()])
    py = sys.executable

    # ---- 0) 프롬프트 ----
    prompt = a.prompt or input("수정 요청을 입력하세요 (예: A의 가운데 홈을 더 깊게 파고 싶다): ").strip()
    banner(0, f"조형 완료 상태에서 시작: 행동 {last}개 이후의 점토. 요청: \"{prompt}\"")
    h_t = np.load(hist / f"action_{last:06d}" / "transition.npz")["after_m"].astype(np.float32)
    np.save(out / "h_t.npy", h_t)

    # ---- 1) 영역 목록: VLM(Gemini) 또는 대역 ----
    banner(1, "VLM: 현재 높이맵을 색 이미지로 그려 요청과 함께 전달 -> 영역별 편집 종류·박스·점")
    img = out / "state_color.png"
    render_state_image(h_t, img)
    print(f"입력 이미지: {img}")
    key_name = next((k for k in KEY_ENVS if os.environ.get(k)), None)
    regions, used_standin = [], True
    if key_name and not a.regions_json:
        parts = [x.strip() for x in re.split(r"[;\n]", prompt) if x.strip()]
        types = [t.strip() for t in a.types.split(",")] if a.types else []
        print(f"API 키({key_name}) 발견 -> Gemini 호출 {len(parts)}회 (요청을 ';' 또는 줄바꿈으로 나눈 부분마다 1회)")
        try:
            for i, part in enumerate(parts):
                pj = out / f"regions/r{i}/prompt.json"
                pj.parent.mkdir(parents=True, exist_ok=True)
                run([py, SAM2_ROOT / "vlm_roi_prompt.py", "--image", img, "--instruction", part, "--output", pj], log, keep=r".")
                d = json.loads(pj.read_text())
                vt = d.get("edit_type")
                if i < len(types):
                    et = types[i]
                elif vt is None:
                    et = "deepen"
                elif vt in VLM_EDIT_TO_TYPE:
                    et = VLM_EDIT_TO_TYPE[vt]
                else:        # 모르는 종류를 조용히 deepen 으로 바꾸지 않는다(귀를 3mm 파 버리는 식의 잘못된 목표를 막는다)
                    raise SystemExit(f"VLM 이 낸 edit_type={vt!r} 에 대응하는 편집 종류가 없습니다(대응: {sorted(VLM_EDIT_TO_TYPE)}). --types 로 영역별 편집 종류를 직접 지정하세요.")
                print(f"  영역 {i + 1}: VLM edit_type={d.get('edit_type')!r} -> 편집 종류 {et}" + (" (--types 로 지정)" if i < len(types) else ""), flush=True)
                regions.append(dict(name=f"r{i}", edit_type=et, request_part=part, prompt=pj))
            used_standin = False
        except SystemExit as e:
            print(f"VLM 호출 실패: {e}\n-> 대역 프롬프트로 대신합니다.")
            regions = []
    if used_standin:
        src = json.loads(a.regions_json.read_text()) if a.regions_json else None
        if src is None:
            if a.vlm_json is None:
                raise SystemExit("VLM API 키가 없거나 호출이 실패했습니다. 대역을 --vlm-json(영역 하나) 또는 --regions-json(여러 영역)으로 주세요. 대상 전용 기본 대역은 없습니다.")
            src = json.loads(a.vlm_json.read_text())
            src = dict(request=src.get("request", {}), regions=[dict(src, name="groove", edit_type="deepen", request_part=src.get("request", {}).get("instruction"))])
        print("*** VLM 대역 사용: API 키가 없거나 호출이 실패했습니다. 아래 영역은 사람이 미리 써 둔 것이며, 이 요청에만 맞습니다. ***")
        print(f"    요청(대역 작성 시): {src.get('request', {}).get('instruction')}")
        regions = []
        for i, r in enumerate(src["regions"]):
            pj = out / f"regions/r{i}/prompt.json"
            pj.parent.mkdir(parents=True, exist_ok=True)
            pj.write_text(json.dumps(r, ensure_ascii=False, indent=2))
            regions.append(dict(name=r.get("name", f"r{i}"), edit_type=r.get("edit_type", "deepen"), request_part=r.get("request_part", ""), prompt=pj))
    for i, r in enumerate(regions):
        d = json.loads(Path(r["prompt"]).read_text())
        print(f"  영역 {i + 1} [{r['name']}] 편집 종류={r['edit_type']} | \"{r['request_part']}\" | 박스 {d['box_norm']} | 안쪽 점 {len(d['positive_points_norm'])}개, 바깥 점 {len(d['negative_points_norm'])}개")

    # ---- 2) SAM 2 + 영역별 목표 ----
    banner(2, "SAM 2: 영역마다 마스크 -> 편집 종류별 목표(깊이 mm 는 규칙으로 계산, VLM 이 정하지 않음) -> 합친 목표 H*")
    from region_edit import region_target
    h_star_all = h_t.astype(np.float64).copy()
    m_union = np.zeros(h_t.shape, bool)
    target_used = hist / "target_used_512_m.npy"
    for i, r in enumerate(regions):
        rd = out / f"regions/r{i}"
        run([py, ROOT / "vlm_sam_roi.py", "--image", img, "--prompt-json", r["prompt"], "--out-dir", rd / "sam"], log, keep=r"=> 선택")
        mask = np.load(rd / "sam/roi_mask_512.npy") > 0
        ref = np.load(a.reference_target if a.reference_target else target_used).astype(np.float32) if r["edit_type"] == "match_target" else None
        if r["edit_type"] == "external":
            tgt, m_applied = external_edit_target(h_t, mask, r["prompt"], rd / "edit", py, log)
        else:
            tgt, m_applied = region_target(h_t, mask, r["edit_type"], delta_mm=a.delta_mm, reference=ref)
        np.save(rd / "mask.npy", m_applied.astype(np.uint8))
        np.save(rd / "target.npy", tgt)
        r.update(dir=rd, mask_px=int(m_applied.sum()), max_change_mm=float(np.abs(tgt - h_t).max()) * 1000)
        print(f"  영역 {i + 1} [{r['name']}] 마스크 {int(mask.sum())}px -> 적용 영역 {r['mask_px']}px, 목표 최대 변화 {r['max_change_mm']:.2f}mm")
        h_star_all += (tgt.astype(np.float64) - h_t)
        m_union |= m_applied
    np.save(out / "h_star_all.npy", h_star_all.astype(np.float32))
    np.save(out / "m_union.npy", m_union.astype(np.uint8))
    spec = out / "spec"
    run([py, ROOT / "edit_spec.py", "--h-t", out / "h_t.npy", "--edit-mask", out / "m_union.npy", "--h-star", out / "h_star_all.npy",
         "--target-used", target_used, "--w-g", 1.0, "--w-b", 1.0, "--out", spec], log, keep=r".")
    print(f"목표/마스크 그림: {spec / 'spec_preview.png'}")
    if a.stop_after == "spec":
        print("(--stop-after spec) 여기서 종료합니다.")
        return

    # ---- 3~4) 영역마다: 과거 기록 beam -> 롤백 반사실 ----
    cstars = []
    for i, r in enumerate(regions):
        rd = Path(r["dir"])
        spec_r = rd / "spec"
        run([py, ROOT / "edit_spec.py", "--h-t", out / "h_t.npy", "--edit-mask", rd / "mask.npy", "--h-star", rd / "target.npy",
             "--target-used", target_used, "--w-g", 1.0, "--w-b", 1.0, "--out", spec_r], log)
        banner(f"3.{i + 1}", f"영역 {i + 1} [{r['name']}, {r['edit_type']}] 과거 기록 탐색: 행동 점수 + 계층 beam (최근 {a.lookback or '전체'}행동) -> 모델 스크리닝")
        run([py, ROOT / "attribution.py", "--history-dir", hist, "--spec-dir", spec_r, "--checkpoint", a.checkpoint,
             "--lookback", a.lookback, "--out", rd / "attribution"], log, keep=r"^L\(H_T\)|상위 묶음|선택 행동|^  #|검증:|=>")
        attr = json.loads((rd / "attribution/attribution.json").read_text())
        if attr["chosen"] and not a.skip_rollback:
            c = attr["chosen"]
            banner(f"4.{i + 1}", f"영역 {i + 1} Genesis 롤백: 행동 {c['action_number']} 직전으로 돌아가 길이×{c['lf']:g}, 깊이×{c['df']:g}만 바꿔 재생 (이후 행동은 그대로)")
            rb = rd / "rollback"
            run([py, ROOT / "rollback_cf.py", "run", "--history-dir", hist, "--start", c["action_number"],
                 "--variant", "control=1:1", "--variant", f"cf_l{c['lf']:g}_d{c['df']:g}={c['lf']}:{c['df']}", "--out", rb], log, keep=r"^\[(control|cf)")
            run([py, ROOT / "rollback_cf.py", "report", "--history-dir", hist, "--start", c["action_number"], "--spec-dir", spec_r, "--out", rb], log, keep=r"control:|^\[cf|=>")
            if (rb / "c_star_512_m.npy").exists():
                cstars.append(rb / "c_star_512_m.npy")
        elif attr["chosen"]:
            print("(--skip-rollback) Genesis 롤백 검증을 건너뜁니다.")
        else:
            print(f"영역 {i + 1}: 과거 기록에서 쓸 만한 후보가 없습니다(abstain).")
    cstar = None
    if cstars:
        cstar = out / "c_star_total.npy"
        np.save(cstar, sum(np.load(c).astype(np.float64) for c in cstars).astype(np.float32))
        print(f"유효한 C* {len(cstars)}개를 합쳐 보정에 사용합니다.")
    else:
        print("유효한 C* 없음 -> 모든 영역에서 abstain. 현재 상태 보정만 합니다(실행 전 시뮬레이터 검증 포함).")

    # ---- 5) 현재 보정 ----
    banner(5, "현재 상태 국소 보정: beam(x,y,theta) -> 자코비안 + Adam(길이,깊이) -> 1행동 실행 -> 재관측 -> Broyden" +
           (" | Genesis 뷰어 켬" if a.viewer else ""))
    watcher = None
    if a.watch:
        watcher = subprocess.Popen([py, ROOT / "watch_map.py", "--spec-dir", spec, "--run-dir", out / "correct"], cwd=ROOT)
        print("높이맵 창을 띄웠습니다(반복마다 갱신).")
    cmd = [py, ROOT / "local_correct.py", "--history-dir", hist, "--spec-dir", spec, "--checkpoint", a.checkpoint,
           "--out-dir", out / "correct", "--iters", a.iters, "--pos-samples", 48 * max(len(regions), 1)]
    if cstar:
        cmd += ["--cstar", cstar]
    if not a.no_verify:
        cmd += ["--verify", "--verify-k", a.verify_k]
    if a.viewer:
        cmd += ["--viewer"]
    run(cmd, log, keep=r"^\[(복원|C\*|시작|\d\d\]|완료)|^    ")

    # ---- 6) 결과 ----
    banner(6, "결과: 목표 height map 과의 눈에 보이는 오차 (아무것도 안 함 = 1.00, 낮을수록 목표에 가까움)")
    run([py, ROOT / "arms_table.py", "--spec-dir", spec, "--arm", f"이번 수정 ({a.iters}회까지)={out / 'correct'}", "--json", out / "result.json"], log, keep=r".")
    run([py, ROOT / "make_shaded_compare.py", "--spec-dir", spec, "--arm", f"After correction={out / 'correct'}", "--out", out / "result_shaded.png"], log, keep=r"저장")
    print(f"\n산출물: {out}\n  - result_shaded.png (목표와 결과 음영 비교)\n  - live.log (전체 로그)\n  - correct/history (실행한 행동의 전체 이력, 다시 재생 가능)")
    print("Genesis 뷰어로 수정 전/후를 번갈아 보려면:\n  python view_state.py --history-dir", hist, "--state before=" + str(hist / f"action_{last:06d}/post_state.pt"),
          "--state after=" + str(next(iter(sorted((out / 'correct/history').glob('action_*/post_state.pt'))[-1:]), "")))
    if watcher:
        print("높이맵 창을 닫으면 종료합니다.")
        watcher.wait()


if __name__ == "__main__":
    main()
