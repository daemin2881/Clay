# 체인 변형 (J / C / T)

체인(`cf_chain.py`)이 한 단계마다 어떻게 확정할지 고르는 세 가지 방법. 실행 스크립트 `run_A.sh`, `run_heart.sh` 맨 위의
`VARIANT=` / `VARIANT_FLAGS=` 두 줄만 변형마다 다르다(`git diff variant/C-commit-by-j variant/T-original -- chain_variants` 로 바로 비교).

| 변형 | 의미 | VARIANT_FLAGS | 브랜치 | 단계 시간 |
|---|---|---|---|---|
| **J** | 반사실 끔. 후보 생성·정제는 같고 확정은 J(영역 오차 점수)로만 | `--no-cf-replay --long-select rank` | main 작업 트리(커밋 안 함) | 1x |
| **C** | T 와 같은 Genesis 반사실(힌트는 후보·사전에 쓰임), 확정은 J 로만 | `--commit-by-j` | `variant/C-commit-by-j` | 약 3x |
| **T** | 원래 방법. 반사실 + T 추적 확정(J 밴드 0.1 후 T 최소) | (없음) | `variant/T-original` | 약 3x |

## 2026-11-17 A/B 결과 (한 단계, 시작 상태 대상마다 1개, 시드 3개)
주 영역 MAE 감소율 평균(%). A = cut2(시작 3.299 mm), 하트 = cut1(시작 3.965 mm). 채점: `python ab_cf_eval.py --target A|heart`.

| 방법 | A | 하트 |
|---|---|---|
| J | **15.1** (13.4/13.3/18.5) | **16.4** (16.9/16.6/15.7) |
| C | 10.0 (10.0/14.2/5.7) | **16.4** (17.3/17.2/14.7) |
| T | 7.7 (2.3/7.5/13.3) | 5.5 (5.5 x3: 시드 3개 모두 CFR#181 확정) |
| V (Genesis 1개만 검증) | 7.0 | 6.6 |
| M (모델 반사실만) | 3.6 | 8.0 |

사전 등록 기준(T 가 J +2.0 %p 이상, 같은 시드에서 T>=J 가 3개 중 2개 이상, 다른 영역 증가 J+0.05 mm 이하) 불충족 → "검증 안 됨".
하트에서 C == J 이고 T 보다 10.9 %p 높아 T 의 약점은 T-추적 확정 규칙 쪽으로 보인다. A 에서는 C 가 T 와 J 사이이고 시드 표준편차(약 5 %p)가 커서 가르기 어렵다.
**한계**: 시작 상태 대상마다 1개, 한 단계만 봄. 반사실의 이득이 여러 단계에 쌓이는 경우는 못 잡았을 수 있다. 사용자가 J 로 전환(2026-11-17), C 와 T 는 이 브랜치들로 보존.

## 실행
```
bash chain_variants/run_A.sh       # A 체인 (뷰어 켬: DISPLAY=:1, 끄려면 VIEWER=0)
bash chain_variants/run_heart.sh   # 하트 체인
```
- 기본으로 CUDA MPS(`results/mps_smoke`)를 쓴다(MPS 데몬이 떠 있어야 함, 끄려면 `USE_MPS=0`). 높이맵 가속(`vs_heightmap_fast.py`)은 `cf_compensate.py` 가 자동으로 켠다(`VS_FAST_HEIGHT=0` 으로 끔, 결과는 비트 단위로 같다).
- 두 체인을 함께 돌리면 GPU 메모리가 32.6 GB 중 약 32 GB 찬다(체인당 약 15 GB).
- 체인은 디스크에서 이어 돌린다(끝난 단계 재사용). 변형을 바꾸려면 체인을 멈추고 다른 브랜치의 스크립트를 가져온다:
  `git show variant/C-commit-by-j:chain_variants/run_A.sh > chain_variants/run_A.sh`
- **체인이 도는 작업 트리에서 `git checkout` 하지 말 것**: 단계마다 `cf_compensate.py` 등을 새로 읽는다.
- 공통 플래그: `--skip-stuck`(--rotate 와 함께: 멈춘 영역 건너뜀), `--min-commit-gain 0.01`(예측 J 이득이 이보다 작으면 확정 안 함), A `--ctrl-valid-len 79`, 하트 `--ctrl-valid-len 190`.
