#!/usr/bin/env python3
"""대기열 진행 요약 (중간 보고용). 실행기 상태, 돌고 있는 작업의 최근 단계, 끝난 ablation 실행의 결과를 한 번에 출력한다.

사용: python progress_report.py            (끝난 실행이 있으면 ablation_report.py 도 돌려 report.md 앞부분을 보여 준다)
      python progress_report.py --no-report
대상(A, 하트 등)에 의존하지 않는다: suite_A.json 의 ablations 목록을 그대로 따른다.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEXT = ROOT / "results/case_A/next_v32"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = ("Genesis", "[push", "[PUSH", "[TIP", "[LIFT", "[END", "[HOLD", "[LOWER", "[GRIP", "[MOVE", "Running at")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def tail_stage(log, n=4):
    """로그에서 단계 표시 줄(대괄호로 시작하는 한국어 안내, 파동, 묶음, 확정 등)만 뒤에서 n개."""
    try:
        text = ANSI.sub("", Path(log).read_text(errors="ignore"))
    except FileNotFoundError:
        return []
    keep = []
    for ln in text.splitlines():
        if any(k in ln for k in NOISE):
            continue
        if re.search(r"②|파동|묶음|확정|라운드|빔|정제|BADS|SPSA|TuRBO|짜내기|쌍|Traceback|Error|재생 작업", ln):
            keep.append(ln.strip()[:230])
    return keep[-n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    print(f"=== 진행 요약 {time.strftime('%F %T')} ===")
    orch = [l.split()[0] for l in sh("ps -eo pid,args").splitlines() if l.strip().endswith("orchestrate_v2.py")]
    print(f"실행기: {orch if orch else '없음!'}" + ("  <- 둘 이상이면 중복" if len(orch) > 1 else ""))
    st = (NEXT / "status.txt").read_text().splitlines() if (NEXT / "status.txt").exists() else []
    if st:
        print(st[0])
        rows = [(l.split()[0], " ".join(l.split()[1:])) for l in st[1:] if l.strip()]
        n_run = sum(v.startswith("실행") for _, v in rows)
        n_done = sum(v.startswith("완료") or v.startswith("건너뜀") for _, v in rows)
        print(f"작업 {len(rows)}개: 실행 {n_run} / 완료 {n_done} / 대기 {len(rows) - n_run - n_done}")
        for k, v in rows:
            if v.startswith("실행") or v.startswith("완료"):
                print(f"  {k:<16} {v}")
    procs = sh("ps -eo pid,etimes,args | grep -E 'cf_compensate.py|cf_edit.py|verify_commits.py' | grep -v grep | grep -v multiprocessing")
    print("프로세스:")
    for l in procs.splitlines():
        m = re.search(r"--out (\S+)", l) or re.search(r"--cf-dir (\S+)", l)
        print(f"  pid {l.split()[0]} {int(l.split()[1]) // 60}분째 {m.group(1) if m else ''}")
    orphan = sh("ps -eo pid,ppid,args | grep 'multiprocessing.spawn' | grep -v grep").splitlines()
    parents = {l.split()[1] for l in orphan}
    alive = set(sh("ps -eo pid").split())
    lost = [p for p in parents if p not in alive]
    if lost:
        print(f"!! 부모가 죽은 워커 프로세스 있음(부모 pid {lost}) -> 정리 필요")
    nv = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader").strip()
    print(f"GPU: {nv}")

    suite = json.loads((NEXT / "suite_A.json").read_text())
    done_dirs = []
    for A in suite.get("ablations", []):
        out = ROOT / A["out"]
        print(f"\n--- 대상 {A['tag']} ({A['out']}) ---")
        for d in sorted(list(out.glob("D_s*")) + list(out.glob("[DE][0-9]_s*"))):
            summ = d / "cf" / "summary.json"
            lines = tail_stage(d / "run.log", 3)
            tb = [l for l in lines if "Traceback" in l or "Error" in l]
            tag = "완료" if summ.exists() else ("진행" if (d / "run.log").exists() else "대기(아직 시작 전)")
            print(f"[{d.name}] {tag}" + ("  !! 에러 흔적" if tb else ""))
            if not summ.exists():
                for l in lines:
                    print("    " + l)
            else:
                done_dirs.append((A, d))
    if done_dirs and not a.no_report:
        for A, d in done_dirs[-1:]:
            out = ROOT / A["out"]
            cmd = f"{sys.executable} ablation_report.py --dir {A['out']} --regions {A['regions']} --active {A['active']} --status results/case_A/next_v32/status.txt"
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
            rep = out / "report" / "report.md"
            print(f"\n=== ablation_report (rc={r.returncode}) {rep} ===")
            if rep.exists():
                print("\n".join(rep.read_text().splitlines()[:70]))
                print("그림:", [str(p.relative_to(ROOT)) for p in sorted((out / 'report').glob('*.png'))])
            else:
                print((r.stdout + r.stderr)[-800:])


if __name__ == "__main__":
    main()
