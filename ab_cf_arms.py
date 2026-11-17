#!/usr/bin/env python3
"""반사실(T 포함)이 실제로 효과가 있는지 같은 시작 상태에서 비교하는 실험을 돌린다. 한 단계(확정 1번)만 보는 비교다.

팔(arm)
  T  = 반사실 켬(J 밴드 + T, 먼 과거 모델 선별) -- 체인이 돌린 그 단계의 명령 그대로. 시드 1 은 체인 결과가 곧 T 팔이다.
  J  = 반사실 끔(--no-cf-replay --long-select rank): 같은 후보 생성, 같은 정제, 확정은 J 로만. 2026-11-16 단계 3 과 같은 방식.
  C  = T 와 똑같은 Genesis 반사실 + 확정은 J 로만(--commit-by-j). T 추적이 J 를 양보하게 하는 효과와 T 힌트가 후보를 좋게 하는 효과를 가른다. 2026-11-17 사용자 승인.
  M  = 모델 반사실만(--screen-model-only, Genesis 재생 없음): 모델 C = 모델(변형)-모델(원래)로 T 를 만든다. 단계 시간이 J 와 비슷하다. 탐색 방법을 바꿔도 효과가 같은지 보는 팔.
같은 시작 상태(이력, 영역, 보호 영역, 조정된 목표)에서 시드만 바꿔 여러 번 돌린다. 시드는 BADS 정제의 난수만 바꾼다(Genesis 비결정성은 별개).
T 팔의 재생은 캐시를 같이 써서 시드 2, 3 은 빠르다(후보 평가만). J 팔은 재생이 없다.

사용: python ab_cf_arms.py --target A --arms J:1,J:2,J:3,T:2,T:3
      (--target = A | heart, 템플릿 results/ab_cf_vs_j/template_<target>_T_seed1.json 은 체인 단계의 실제 명령을 저장한 것)
채점: python ab_cf_eval.py --target A
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "ab_cf_vs_j"


def drop(cmd, flag, nargs=1):
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] == flag:
            i += 1 + nargs
            continue
        out.append(cmd[i])
        i += 1
    return out


def set_val(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(value)
    else:
        cmd += [flag, str(value)]
    return cmd


def build(tpl, arm, seed, out_cf):
    cmd = drop(tpl, "--viewer", 0)
    cmd = set_val(cmd, "--seed", seed)
    cmd = set_val(cmd, "--out", out_cf)
    if arm == "J":
        cmd = [c for c in cmd if c != "--no-cf-replay"] + ["--no-cf-replay"]
        cmd = set_val(cmd, "--long-select", "rank")
    elif arm == "C":      # T 와 똑같이 Genesis 반사실(트리 + 모델 선별 + 검증)을 쓰되 확정은 T-추적 없이 J 로만 한다: 반사실 힌트가 후보와 정제 사전을 좋게 하는 효과만 본다
        if "--commit-by-j" not in cmd:
            cmd.append("--commit-by-j")
    elif arm == "V":      # 모델이 거른 1위 후보 하나만 Genesis 로 재생해 검증(②T 트리와 ② 최근 재생은 건너뜀): 반사실은 Genesis 로 확인하고 단계 시간만 줄인다
        cmd = set_val(cmd, "--long-select", "seq")
        cmd = set_val(cmd, "--cf-top", "0")
        cmd = set_val(cmd, "--screen-verify", "1")
    elif arm == "M":      # 모델 반사실만(Genesis 재생 없음): ②T 트리와 ② 최근 재생과 Genesis 검증을 모두 끄고 모델 C 로 T 를 만든다
        cmd = set_val(cmd, "--long-select", "seq")
        cmd = set_val(cmd, "--cf-top", "0")
        if "--screen-model-only" not in cmd:
            cmd.append("--screen-model-only")
    return cmd


def gpu_free_mb():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])


def watchdog(proc, stop_evt, item, pause_below, resume_above):
    """GPU 여유 메모리가 바닥나면(메인 체인이 후보 평가로 메모리를 더 쓸 때) 팔의 프로세스 그룹만 일시정지(SIGSTOP)하고, 여유가 다시 생기면 재개(SIGCONT)한다.
    메인 체인을 OOM 에서 지키기 위한 장치다(2026-11-17 첫 시도에서 팔과 메인이 겹쳐 31.8/32.6GB 까지 찼다). 일시정지는 팔의 결과를 바꾸지 않는다(T 팔 2,3 과 J, M 팔은 벽시계 예산이 없다)."""
    paused = False
    pgid = os.getpgid(proc.pid)
    while not stop_evt.wait(10):
        if proc.poll() is not None:
            return
        try:
            free = gpu_free_mb()
            if not paused and free < pause_below:
                os.killpg(pgid, signal.SIGSTOP)
                paused = True
                print(f"[{time.strftime('%T')}] {item} 일시정지: GPU 여유 {free}MB < {pause_below}MB", flush=True)
            elif paused and free > resume_above:
                os.killpg(pgid, signal.SIGCONT)
                paused = False
                print(f"[{time.strftime('%T')}] {item} 재개: GPU 여유 {free}MB", flush=True)
        except Exception as e:      # 감시가 죽어도 팔은 계속 돌게 둔다
            print(f"[{time.strftime('%T')}] 감시 오류 {e!r}", flush=True)
    if paused:
        try:
            os.killpg(pgid, signal.SIGCONT)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=["A", "heart"])
    ap.add_argument("--arms", default="J:1,M:1,T:2,J:2,M:2,T:3,J:3,M:3", help="쉼표로 구분한 팔:시드 목록")
    ap.add_argument("--need-mb", type=int, default=12200)
    ap.add_argument("--pause-below-mb", type=int, default=2500, help="GPU 여유가 이보다 작으면 팔을 일시정지")
    ap.add_argument("--resume-above-mb", type=int, default=7000, help="GPU 여유가 이보다 커지면 팔을 재개")
    ap.add_argument("--workers", type=int, default=0, help="모든 팔에 같은 워커 수를 쓴다(0=템플릿 그대로 3). 팔 2개를 병렬로 돌리려면 메모리 때문에 2로 낮춘다. 정제(BADS)의 배치가 워커 수라서 모든 팔이 같아야 비교가 맞다")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tpl = json.loads((BASE / f"template_{a.target}_T_seed1.json").read_text())
    for item in a.arms.split(","):
        arm, seed = item.split(":")
        d = BASE / a.target / f"{arm}_s{seed}"
        out_cf = str((d / "cf").relative_to(ROOT))
        cmd = build(tpl, arm, int(seed), out_cf)
        if a.workers > 0:
            cmd = set_val(cmd, "--workers", a.workers)
        if (d / "cf" / "summary.json").exists():
            print(f"[{item}] 이미 끝남: 건너뜀", flush=True)
            continue
        if a.dry_run:
            print(f"[{item}] " + " ".join(cmd[2:])[:600])
            continue
        while gpu_free_mb() < a.need_mb:
            time.sleep(30)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[{time.strftime('%T')}] {a.target} {item} 시작", flush=True)
        with open(d / "run.log", "a") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            stop_evt = threading.Event()
            threading.Thread(target=watchdog, args=(proc, stop_evt, item, a.pause_below_mb, a.resume_above_mb), daemon=True).start()
            rc = proc.wait()
            stop_evt.set()
        print(f"[{time.strftime('%T')}] {a.target} {item} 종료 rc={rc} ({(time.time() - t0) / 60:.0f}분)", flush=True)


if __name__ == "__main__":
    main()
