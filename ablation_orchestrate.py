#!/usr/bin/env python3
"""메모(반사실_국소보상_탐색_연구메모.pdf) 4쪽 검증 설계를 그대로 돌리는 실행기: 같은 출발점에서 조건 D, E1, E2, E3, E4 를 시드 여러 개로 반복한다(A, B, C 는 사용자 결정으로 뺐다).

출발점: A 이력 K=80 상태, 영역 groove_top(regions_all_v5 의 '위 홈'), 첫 라운드 하나(--max-commits 1). 나머지 4개 영역은 --skip-regions 로 고치지는 않고 점수만 기록한다.
공통: 영역 목표의 높이 자동 조정(--level-adjust: 베이스라인처럼 점토 총량이 보존되게 목표를 조정), 영역 밖 손상 방지(--far-limit-auto 1.0 = 과거 푸시 중 가장 큰 값을 한도로, --cbo = 제약 베이지안 최적화) + 고치지 않기로 한 영역 보호(--protect-skipped --protect-limit-auto 0.95 = 그 영역에 닿지 않은 과거 푸시가 번짐으로 바꾼 최대 변화의 95% 분위수), 순서쌍 정책 동일(--pair-branch --pairs 3 --pair-seconds 4), 조기 종료 끔(--patience 0 --refine-patience 0: 평가 n 개 시점 비교를 위해 평가 수를 고정), 쿼터 고정(--no-adaptive-quota),
      빔 쿼터 합 24 로 맞춤(실제 평가 수는 후보 가용성에 따라 달라서 보고서가 실제 개수를 같이 적는다), 시드는 --seed.

조건                설명
  D   C + 반사실 재생(#61~#80) + πBO 사전. 확정 기준에는 힌트를 쓰지 않는다(--w-cf 0: 메모 3쪽의 '확정 기준은 기존과 같다')
  E1  D + 먼 과거 계층(트리) 재생 (cf_tree, --long-select tree)
  E2  D + 개입 묶음 탐색 (cf_seq_cf, Tsirtsis 외 2023 [10] 의 구조, --long-select seq)
  E3  전부 합침: D + 먼 과거 트리(E1) + 개입 묶음 탐색(E2) + SPSA 결합 + TuRBO + 순서쌍 예산 조절
  E4  E3 + 짜내기 후보 (구덩이 둘레를 눌러 점토가 구덩이로 밀려 올라오게, cf_transport.squeeze_candidates)
  E3S E3 의 스모크 시험(예산 아주 작게, 대기열 맨 앞)

사용: python ablation_orchestrate.py [--dry-run] [--seeds 1 2 3] [--out results/case_A/ablation_v1]
산출: OUT/<조건>_s<시드>/ (cf/ 와 run.log), OUT/status.txt, 끝나면 OUT/ALL_DONE.  GPU 여유 메모리를 보고 한 번에 2개까지 시작한다.
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = "/home/airlab/research/clay_genesis/.venv/bin/python"
H = "results/A_history200_20261114_060837/history"
REGIONS = "results/case_A/regions_all_v5"
ACTIVE = "groove_top"
# 대상(이력, 시점, 영역 세트, 고칠 영역)은 spec 으로만 정한다. 알고리즘 코드는 대상에 의존하지 않는다. 기본값은 지금까지 쓴 A 설정.
DEFAULT_SPEC = dict(history_dir=H, upto=80, regions=REGIONS, active=ACTIVE, out="results/case_A/ablation_v1")


def region_names(regions_dir):
    """영역 세트 폴더의 영역 이름 목록(regions.json 우선, 없으면 *_mask.npy)."""
    import glob
    import json
    p = ROOT / regions_dir / "regions.json"
    if p.exists():
        j = json.loads(p.read_text())
        return [r["name"] for r in (j["regions"] if isinstance(j, dict) and "regions" in j else j)]
    return sorted(Path(f).name[:-len("_mask.npy")] for f in glob.glob(str(ROOT / regions_dir / "*_mask.npy")))


def others_of(spec):
    """고칠 영역(active) 말고 점수만 기록할 영역들 (--skip-regions)."""
    names = region_names(spec["regions"])
    if spec["active"] not in names:
        raise SystemExit(f"영역 '{spec['active']}' 이(가) {spec['regions']} 에 없다: {names}")
    return ",".join(n for n in names if n != spec["active"])
MARGIN_MB, SPACING_S, NEED_MB = 1200, 60, 11000

Q_D = "G1:4,CFR:4,DEEP:2,G2:4,G3:2,MDL:5,PLAN:3"
Q_E4 = "G1:3,CFR:3,DEEP:2,G2:3,G3:2,MDL:4,PLAN:3,SQ:4"     # 빔 후보 합 24: E3 와 같은 예산에서 일부를 짜내기 후보(SQ)로 바꾼다
BADS = ["--refine-mode", "bads", "--bads-evals", "20", "--diverse"]
CONDS = {
    "D": (["--deep", "--pibo", "--w-cf", "0"] + BADS, Q_D, 3 * 3600),
    "E1": (["--deep", "--pibo", "--w-cf", "0", "--long-select", "tree", "--control", "record", "--tree-budget-min", "60"] + BADS, Q_D, 5 * 3600),
    "E2": (["--deep", "--pibo", "--w-cf", "0", "--long-select", "seq", "--control", "record", "--replay-sec-per-action", "30", "--seq-pool", "6", "--seq-k", "2", "--seq-budget-min", "60"] + BADS, Q_D, 5 * 3600),
    # E3 = 메모의 모든 방법을 합친 조건: D 의 모든 것(BADS [3], 지역 벌점 [4], 순서쌍 분기 [5], πBO [2], 반사실 재생, 순위 상관 기록 [6]) + 먼 과거 트리 재생 + 개입 묶음 탐색 [10]
    #      + SPSA 결합 [7](BADS 15 + SPSA 5 = 정제 평가 약 20회. 평가가 워커 3개 단위라 D/E1/E2 의 BADS 21회와 1회 차이) + TuRBO [8] + 순서쌍 예산 조절. (순서쌍 정책이 D/E1/E2 와 달라서 그 차이가 섞인다)
    "E3": (["--deep", "--pibo", "--w-cf", "0", "--long-select", "both", "--control", "record", "--replay-sec-per-action", "30", "--tree-budget-min", "60", "--seq-pool", "6", "--seq-k", "2", "--seq-budget-min", "60",
            "--refine-mode", "bads", "--bads-evals", "15", "--spsa-after", "4", "--turbo", "--diverse", "--pair-adapt"], Q_D, 7 * 3600),
    # E4 = E3 + 짜내기 후보(cf_transport.squeeze_candidates): 채울 곳(구덩이) 둘레를 눌러 점토가 구덩이로 밀려 올라오게 한다(--squeeze 4). 빔 후보 합은 E3 와 같게(24). (수송 계획 OT 는 사용자 결정으로 대기열에서 뺐다: --ot-donors 로 켤 수는 있다)
    "E4": (["--deep", "--pibo", "--w-cf", "0", "--long-select", "both", "--control", "record", "--replay-sec-per-action", "30", "--tree-budget-min", "60", "--seq-pool", "6", "--seq-k", "2", "--seq-budget-min", "60",
            "--refine-mode", "bads", "--bads-evals", "15", "--spsa-after", "4", "--turbo", "--diverse", "--pair-adapt", "--squeeze", "4"], Q_E4, 7 * 3600),
    # E3S = E3 의 스모크 시험(예산을 아주 작게): 모든 코드 경로(②T, ②S, turbo, spsa, pair-adapt)가 Genesis 에서 실제로 도는지 먼저 본다. 재생 결과는 캐시에 남아 뒤의 실행이 빨라진다.
    "E3S": (["--deep", "--pibo", "--w-cf", "0", "--long-select", "both", "--control", "record", "--tree-budget-min", "5", "--seq-pool", "2", "--seq-k", "2", "--seq-budget", "1",
             "--refine-mode", "bads", "--bads-evals", "4", "--spsa-after", "2", "--turbo", "--diverse", "--pair-adapt"], Q_D, 3 * 3600),
}
# E5 = E3 + '힌트가 확정을 이끈다': 사용자가 원한 목적 = 과거 행동을 이만큼 눌렀다면 목표였을 텐데(반사실) -> 지금 그 자리와 비교해 얼마나 더 올리고/눌러야 하는지(D = T - 현재) -> 그만큼 맞추는 푸시를 찾아 확정.
#      --cf-both-dirs  과거 행동을 덜/더 누른 변형을 양쪽 다 재생(w 가 0 에 가까우면 방향이 불확실)
#      --cf-target fixed  반사실 목표 T 를 고정하고 확정 후보를 T 에 가까운 것으로 고른다(J 가 허용치 --t-slack 안일 때. 쌍도 T 거리로 판단)
#      --t-metric blob    T 까지 거리를 '더 올리거나 눌러야 하는 덩어리'에서만 잰다(창 전체 평균은 부수 변화가 이득보다 크게 잡혀 v20/v21 에서 T 에 가까워진 후보가 거의 없었다)
#      --hint-match 4     D 를 모델 예측으로 가장 잘 맞추는 푸시(세기까지)를 후보(HM)로 4개 올린다. 빔 후보 합은 E3 와 같게 24(TRK 3 + HM 4 를 위해 G2, G3, MDL, PLAN 에서 덜어 냄)
#      유효한 반사실(q>0)이 있는 영역에서만 T 가 만들어진다. 없으면 이 조건은 E3 와 같게 돈다(보고서가 features.hm 으로 구분).
Q_E5 = "G1:3,CFR:3,DEEP:2,G2:2,G3:2,MDL:3,PLAN:2,TRK:3,HM:4"
CONDS["E5"] = (CONDS["E3"][0] + ["--cf-both-dirs", "--cf-target", "fixed", "--t-metric", "blob", "--hint-match", "4"], Q_E5, 7 * 3600)

# ---- 힌트 기여를 분리하는 대조 3종(메모 4쪽의 D 대 C 설계). 같은 정제(BADS 20), 같은 후보 수(24), 가까운 과거(#61~#80) 재생만(먼 과거 트리/묶음 없음), 한 라운드.
#      D0 = 힌트 전부 끔(--no-hints: 과거 파생 후보와 모든 과거 재생 없음). 후보 24 를 맞추려고 힌트 계열(G1 4 + CFR 4 + DEEP 2 = 10칸)을 MDL +6, PLAN +3, G3 +1 로 돌렸다(이 배분은 제가 정한 값)
#      D  = 힌트를 후보 + πBO 사전으로만 쓴다(확정은 J)
#      D5 = D + 힌트가 확정을 이끈다(E5 와 같은 4개 플래그: 양쪽 방향 재생, T 고정 추적, blob 거리, 모델이 D 를 맞추는 푸시)
#      유효한 힌트(q>0)가 있는 영역에서만 D 와 D0 가 달라진다 -> 영역은 힌트가 있는 곳(A 의 fill3: v30 에서 q +1.48)으로 고른다.
Q_D0 = "G2:4,G3:3,MDL:11,PLAN:6"
Q_D5 = "G1:3,CFR:3,DEEP:2,G2:2,G3:2,MDL:3,PLAN:2,TRK:3,HM:4"
CONDS["D0"] = (["--no-hints", "--w-cf", "0"] + BADS, Q_D0, 3 * 3600)
CONDS["D5"] = (CONDS["D"][0] + ["--cf-both-dirs", "--cf-target", "fixed", "--t-metric", "blob", "--hint-match", "4"], Q_D5, 3 * 3600)
# 처음 써 보는 연결(E2)과 가장 오래 걸리는 조건을 시드 1 에서 먼저 시작해 문제를 일찍 본다.
ORDER = ["E3", "E4", "E2", "D", "E1"]          # 사용자 결정(11-16): 조건은 D, E1, E2, E3 (A, B, C 는 뺀다). E3(전부 합침)를 가장 먼저 시작해 문제를 일찍 본다


def cmd_for(cond, seed, out, spec=None):
    spec = spec or DEFAULT_SPEC
    flags, quota, _ = CONDS[cond]
    common = [PY, "-u", "cf_compensate.py", "--history-dir", spec["history_dir"], "--upto", str(spec["upto"]), "--regions", spec["regions"], "--skip-regions", others_of(spec),
              "--out", f"{out}/{cond}_s{seed}/cf",
              "--workers", "3", "--gate", "local", "--w-tail", "1", "--ring-budget-mm", "*=50", "--far-limit-mm", "50", "--far-limit-auto", "1.0", "--cbo", "--protect-skipped", "--protect-limit-auto", "0.95", "--credit-side", "1.0", "--level-adjust", "--own-score",
              "--max-commits", "1", "--per-region-commits", "1", "--patience", "0", "--refine-patience", "0",
              "--pairs", "3", "--pair-branch", "--pair-seconds", "4", "--refine-top", "2", "--plan-cands", "4", "--jac-top", "0", "--long-top", "0",
              "--no-adaptive-quota", "--quota", quota, "--seed", str(seed), "--budget-min", "240", "--strict"]
    return common + flags


def gpu_free_mb():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="results/case_A/ablation_v1")
    ap.add_argument("--conds", nargs="+", default=ORDER)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = a.out
    jobs = [(f"{c}_s{s}", c, s) for s in a.seeds for c in a.conds]
    if a.dry_run:
        for name, c, s in jobs:
            print(f"[{name}] timeout {CONDS[c][2] // 60}분\n   " + " ".join(cmd_for(c, s, out)))
        return
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    state, lock = {n: "대기" for n, _, _ in jobs}, threading.Lock()

    def run_job(name, c, s):
        d = ROOT / out / name
        d.mkdir(parents=True, exist_ok=True)
        t0, code = time.time(), 0
        with open(d / "run.log", "a") as f:
            cmd = cmd_for(c, s, out)
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            while p.poll() is None:
                if time.time() - t0 > CONDS[c][2]:
                    p.kill()
                    f.write("\n[시간 초과로 중단]\n")
                    code = 124
                    break
                time.sleep(10)
            code = code or p.returncode
        with lock:
            state[name] = f"완료 rc={code} ({(time.time() - t0) / 60:.0f}분)"

    def write_status():
        (ROOT / out / "status.txt").write_text("\n".join([f"{time.strftime('%F %T')}  GPU 여유 {gpu_free_mb()} MiB"] + [f"{k:<10} {v}" for k, v in state.items()]) + "\n")

    last = 0.0
    while True:
        with lock:
            pending = [j for j in jobs if state[j[0]] == "대기"]
            running = [k for k, v in state.items() if v.startswith("실행")]
        if not pending and not running:
            break
        if pending and time.time() - last >= SPACING_S and len(running) < 2 and gpu_free_mb() - MARGIN_MB >= NEED_MB:
            name, c, s = pending[0]
            if (ROOT / out / name / "cf" / "summary.json").exists():
                with lock:
                    state[name] = "건너뜀 (이미 결과 있음)"
                continue
            with lock:
                state[name] = f"실행 중 ({time.strftime('%T')}~)"
            threading.Thread(target=run_job, args=(name, c, s), daemon=True).start()
            last = time.time()
        write_status()
        time.sleep(20)
    write_status()
    (ROOT / out / "ALL_DONE").write_text(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
