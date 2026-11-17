"""보상 실행들이 남긴 높이맵과 '80번째 이후 총 몇 행동을 썼는지'를 한 파일로 정리한다.
확정 한 번 = 행동 1개, 2-step(PAIR) = 행동 2개. 단계별 높이맵은 <run>/cf/states/iter_XXX.npy, 최종은 <run>/cf/final_m.npy.
이어 조형 팔은 <arm>/states/step_XXX.npy(행동마다)와 final_m.npy.
사용: python compensation_manifest.py --base-k 80 --run v8=results/case_A/cf_edit_A80_v8[:start_total] --run v10=... [--arm B1=results/case_A/ab_B1:85]
"""
import argparse, json, re
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-k", type=int, required=True)
    ap.add_argument("--run", action="append", default=[], help="이름=cf_edit 출력 폴더[:시작 시점의 총 행동 수(기본 base-k)]")
    ap.add_argument("--arm", action="append", default=[], help="이름=continue_sculpt 출력 폴더:시작 시점의 총 행동 수")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        path, _, start = rest.partition(":")
        k = int(start) if start else a.base_k
        root = Path(path)
        log = Path(str(root) + ".log")
        stages = [dict(stage="start", total_actions=k, heightmap=str(root / "cf/states/iter_000.npy"))]
        if log.exists():
            for line in log.read_text(errors="ignore").splitlines():
                m = re.match(r"\s+확정 (\d+): (\S+)", line)
                if m:
                    n = int(m.group(1))
                    k += 2 if " + x=" in line else 1
                    stages.append(dict(stage=f"commit {n} ({m.group(2)})", pushes=2 if " + x=" in line else 1, total_actions=k, heightmap=str(root / f"cf/states/iter_{n:03d}.npy")))
        final = root / "cf/final_m.npy"
        out[name] = dict(start_total_actions=int(start) if start else a.base_k, final_total_actions=k, final_heightmap=str(final) if final.exists() else "(실행이 끝나면 생성)", stages=stages)
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        path, _, start = rest.partition(":")
        k0 = int(start)
        steps = sorted(Path(path, "states").glob("step_*.npy"))
        out[name] = dict(start_total_actions=k0, final_total_actions=k0 + len(steps), final_heightmap=str(Path(path) / "final_m.npy"), per_action_heightmaps=f"{path}/states/step_001..{len(steps):03d}.npy")
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for n, v in out.items():
        print(f"{n:<10} 시작 {v['start_total_actions']} -> 최종 {v['final_total_actions']}행동 | 최종 높이맵 {v['final_heightmap']}")

if __name__ == "__main__":
    main()
