# PushT deployment: 기본 / Red T / Salt-and-pepper

기본 PushT 데이터로 학습한 **하나의 Flow-JEPA 체크포인트를 고정**하고 세 시각
조건에서 MPC 제어를 비교합니다. 조건마다 모델을 다시 학습하지 않습니다.

| 조건 | 현재 관측 이미지 | 목표 이미지 |
|---|---|---|
| `default` | 기본 렌더링 | 기본 렌더링 |
| `red_t` | 실제 T 블록을 RGB (255, 0, 0)으로 렌더링 | 동일한 빨간 T로 렌더링 |
| `salt_pepper` | 매 프레임 픽셀 5% 선택, 흰색/검정 각각 절반 | 별도의 독립 마스크로 동일 적용 |

T의 모양·질량·충돌·성공 판정과 agent/배경/초록 목표 영역의 색은 동일합니다.
Salt-and-pepper는 선택된 픽셀의 RGB 세 채널을 함께 0 또는 255로 바꿉니다.
5%는 **선택 확률**입니다. 흰 배경에 흰색이 선택되면 겉으로 달라지지 않을 수
있으므로 실제 달라진 픽셀 비율은 5%보다 작을 수 있습니다.

## 공통 평가 프로토콜

- 기본 50개의 시작·목표 쌍을 seed=42로 한 번 뽑고 세 조건에서 그대로 사용합니다.
- 시작·목표 물리 상태는 `pusht_expert_train.h5`에서 가져옵니다.
- 세 조건의 **현재·목표 이미지 모두 시뮬레이터에서 다시 렌더링**합니다.
  이 방식으로 목표 이미지의 T 색도 정확히 일치시킵니다. 기존 `eval.py`의
  저장된 데이터 이미지 복사 방식과 구분되는 별도 프로토콜입니다. 기본 조건
  점수도 이번 suite에서 측정하며, 기존 평가 점수와 직접 혼합하지 마세요.
- 목표 위치를 나타내는 초록 영역은 해당 목표 상태의 T 위치/각도에 맞춥니다.
- 성공 판정은 설치된 PushT의 `eval_state`를 그대로 사용합니다.
- 동일한 체크포인트, 평가 샘플, flow/CEM seed, horizon과 예산을 사용합니다.
- 조건마다 planner를 새로 만들고 flow RNG를 초기화합니다.
- SP 마스크는 매 정책 호출 시 새로 뽑습니다. goal은 고정 물리 목표에
  노이즈만 다시 적용합니다. receding-horizon MPC가 캐시한 action을 실행하는
  동안에도 관측 노이즈는 새로 뽑힙니다.

## 코드 push와 서버 pull

로컬 PowerShell에서:

```powershell
cd D:\Flow-JEPA
git add .
git commit -m "Add paired PushT deployment evaluation in three visual conditions"
git push
```

CIRCE의 본인 Flow-JEPA clone에서:

```bash
git pull --ff-only
source scripts/circe.env
```

전체 학습 모델의 `.pt`와 옆의 `config.json`이 필요합니다. 짧은 학습 검증용
`flow-jepa-pusht-smoke` 모델은 정상 동작 확인에만 사용할 수 있습니다.

## CIRCE smoke test

이전 학습 스크립트의 경로/계정 설정을 읽고, 별도 평가 ID를 지정합니다.
`FLOWJEPA_RUN_ID`는 학습 때 사용한 ID입니다.

```bash
source scripts/circe.env
export FLOWJEPA_EVAL_CHECKPOINT="$FLOWJEPA_RUNS_ROOT/$FLOWJEPA_RUN_ID/worldmodel/checkpoints/$FLOWJEPA_RUN_ID/weights_epoch_20.pt"
export FLOWJEPA_EVAL_ID=flowjepa-eval-pusht-NEW-UNIQUE-ID
export FLOWJEPA_EVAL_TIME=00:15:00
export FLOWJEPA_EVAL_SEED=42

# 명령·요청 자원 미리보기. 여기서는 제출하지 않습니다.
bash scripts/circe_eval_submit.sh smoke
bash scripts/circe_eval_submit.sh smoke --check

# 위 명령과 계정 권한을 확인한 뒤 실제 제출
eval_job=$(bash scripts/circe_eval_submit.sh smoke --submit)
eval_job=${eval_job%%;*}
echo "$eval_job"
```

smoke에서는 각 조건에 동일한 1개 샘플을 사용하며, horizon=1, action_block=5,
budget=5, CEM samples=4 / iterations=1로 줄여 실행과 영상 저장을 확인합니다.
이 점수는 성능 측정용이 아닙니다.

## 정식 평가

한 GPU 작업이 세 조건을 **순차 실행**합니다. 요청 자원은 GPU 1 / CPU 8 /
RAM 64 GB이며, 이는 보수적인 요청값입니다. 실제 VRAM 사용량과 실행 시간은
CIRCE GPU에서 확인해야 합니다. partition/QoS는 현재 계정에서 허용된 설정을
사용합니다. `muma_2021 / preempt_short / 48시간`은 확인 전에는 확정값이 아닙니다.

48시간이 현재 QoS에서 허용된다고 확인한 경우의 명령:

```bash
export FLOWJEPA_EVAL_TIME=2-00:00:00
bash scripts/circe_eval_submit.sh full
bash scripts/circe_eval_submit.sh full --check
# 전체 학습이 끝났고 smoke가 성공했으며 자원/명령을 검토한 뒤 실행
eval_job=$(bash scripts/circe_eval_submit.sh full --submit)
eval_job=${eval_job%%;*}
```

full은 `config/eval/pusht.yaml`과 `solver/cem.yaml`의 기본 평가 설정을 사용합니다:
50개 평가 시작점, goal offset=25, budget=50, horizon=5, action_block=5,
CEM candidates=300 / iterations=30 / topk=30. 매우 짧은 smoke와 실행 시간이
크게 다를 수 있습니다. 완료 시간을 보장하는 요청은 아닙니다.

## 상태·로그·결과

```bash
squeue -j "$eval_job"
sacct -j "$eval_job" --format=JobID,State,ExitCode,Elapsed,MaxRSS
# smoke 로그를 보려면 ID 뒤에 -smoke를 붙입니다.
tail -f "$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID/$FLOWJEPA_EVAL_ID-$eval_job.out"
tail -f "$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID/$FLOWJEPA_EVAL_ID-$eval_job.err"
tail -f "$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID/gpu-$eval_job.csv"
scancel "$eval_job" # 해당 작업을 취소할 때만 실행
```

결과는 `$FLOWJEPA_RUNS_ROOT/evaluations/$FLOWJEPA_EVAL_ID/` 아래 저장됩니다.
`FLOWJEPA_EVAL_ROOT`를 별도 절대경로로 지정할 수도 있습니다. smoke 결과 ID에는
`-smoke`가 붙습니다. 기존 결과가 있으면 덮어쓰지 않고 중단합니다.

```text
summary.csv                  세 조건의 성공률(%)과 시간
trials.json                  공통 row/episode/start/goal 및 물리 상태·seed
checkpoint.json              사용한 체크포인트 절대경로와 SHA-256
config.yaml                  평가 설정
default/                     기본 조건
red_t/                       빨간 T 조건
salt_pepper/                 5% SP 조건
  metrics.json               조건별 성공률과 에피소드별 성공 여부·step
  episodes.csv               공통 평가 샘플별 결과
  initial_states.npy         reset 직후 시뮬레이터 물리 상태
  initial_observation.png    첫 샘플의 실제 정책 입력
  goal.png                   첫 샘플의 목표 정책 입력
  final_observation.png      마지막 action 실행 뒤 첫 샘플의 관측
  videos/env_N.mp4           각 샘플의 관측·목표 영상
```

영상은 정책에 전달된 현재/목표 이미지를 기록하고 마지막 action 뒤의 관측을
마지막 프레임으로 덧붙입니다. 기존 학습 체크포인트나 W&B run은 수정하지 않습니다.
평가가 중단되면 새 평가 ID로 다시 실행하세요. 동일 seed/설정/체크포인트를
사용하면 같은 샘플을 선택합니다. 학습 체크포인트 재개와는 별도입니다.

## 직접 실행

서버에서는 GPU를 Slurm으로 할당받은 작업 안에서만 실행하세요:

```bash
"$FLOWJEPA_PYTHON" "$FLOWJEPA_REPO/eval_pusht_suite.py" \
  "policy=$FLOWJEPA_EVAL_CHECKPOINT" "cache_dir=$FLOWJEPA_DATA_ROOT" \
  "output.dir=$FLOWJEPA_RUNS_ROOT/evaluations/$FLOWJEPA_EVAL_ID"
```

salt 양을 바꾸는 실제 Hydra 인수는 `suite.salt_pepper.amount=0.05`입니다.
현재 기본값은 요청한 5%이며 SP와 Red T를 섞지 않은 세 조건입니다.
