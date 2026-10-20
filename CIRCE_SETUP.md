# Flow-JEPA + PushT: CIRCE 운영 절차

AdaJEPA와 별도 실험입니다. 기존 AdaJEPA 저장소, 환경, 데이터, 체크포인트,
W&B run은 참조하거나 수정하지 않습니다. CIRCE에는 아직 설치하지 않았으며,
이 문서의 서버 경로는 로그인 후 정해야 합니다. 전체 학습을 제출하지 않았습니다.

## 조사한 실제 코드

| 항목 | 이 저장소에서 확인한 내용 |
|---|---|
| 진입점 | `train.py`, Hydra `--config-name=pusht_circe` |
| 모델 | `jepa.JEPA`, `module.FlowTrajectoryPredictor`, ViT-tiny scratch encoder |
| 데이터 | `config/train/data/pusht.yaml`: `pusht_expert_train.h5`, pixels/action/proprio/state |
| 데이터 루트 | `LOCAL_DATASET_DIR/datasets/pusht_expert_train.h5` |
| 기본 학습 | 20 epoch, AdamW lr=5e-5, num_preds=5, frameskip=5 |
| CIRCE 시작값 | 1 GPU, batch=8, worker=6, FP16 mixed precision |
| Python | 별도 Linux Python 3.10 환경. Windows `.venv` 재사용 불가 |
| 의존성 | torch 2.6.0/cu124, torchvision 0.21.0, stable-worldmodel 0.1.1, stable-pretraining 0.1.8 |
| 평가용 가중치 | `SaveCkptCallback`: epoch마다 `.pt`와 `config.json` |
| 재개용 상태 | stable-pretraining의 full-state `last.ckpt`; CIRCE 설정은 100 step마다 저장 |
| 재개 설정 | 이번 변경으로 `checkpoint.resume=/absolute/path/last.ckpt`를 추가; full state로 로드 |
| W&B | 원래 비활성. 사용 시 별도 project `flow-jepa-pusht`와 새 `flowjepa-...` ID |

요청 자원은 GPU 1개 / CPU 8개 / RAM 64 GB입니다. 이는 보수적인 시작 요청이며
실측 필요량은 아닙니다. 로컬 RTX 4050에서는 batch=8 학습이 통과했지만 CIRCE GPU는
아직 모릅니다. GPU VRAM, 속도, peak RAM은 서버 smoke test로 확인합니다.
batch=8은 논문의 기본 batch=128과 다릅니다. 20 epoch 소요 시간은 아직 산정하지 않았습니다.
초기 precision은 `16-mixed`입니다. 할당 GPU가 BF16을 지원한다고 확인한 후
`FLOWJEPA_PRECISION=bf16-mixed`로 바꿀 수 있으며, 작업 내부에서 지원 여부를 검사합니다.

## 서버가 비어 있는 현재 단계

PowerShell에서 `ssh YOUR_NETID@circe.rc.usf.edu`로 접속합니다. 이후 서버 Bash에서:

```bash
git clone https://github.com/jinu7513/Flow-JEPA.git "$HOME/Flow-JEPA-circe"
cd "$HOME/Flow-JEPA-circe"
pwd -P
module avail
```

현재 CIRCE 스크립트와 코드 수정은 **로컬에만** 있습니다. 아직 GitHub에 push하지
않았으므로 clone만 하면 포함되지 않습니다. 제공한 소스 ZIP을 전송·덮어씌우거나,
변경 내용을 본인 포크의 새 브랜치에 올린 뒤 서버에서 받아야 합니다.
ZIP에는 Python 환경, 데이터, 체크포인트가 포함되지 않습니다.

소스 ZIP을 사용하는 경우 로컬 PowerShell에서:

```powershell
scp D:\Flow-JEPA\storage\flow-jepa-circe-source.zip YOUR_NETID@circe.rc.usf.edu:~/
```

서버에서 본인 Flow-JEPA clone에만 적용:

```bash
cd "$HOME/Flow-JEPA-circe"
unzip -o "$HOME/flow-jepa-circe-source.zip" -d "$PWD"
```

## 현재 계정과 Slurm 제한 확인

아래는 조회 명령이며 작업을 제출하지 않습니다.

```bash
id -un
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS
scontrol show partition muma_2021
sacctmgr show qos preempt_short
sinfo -p muma_2021 -N -o "%N %G %m %c %t"
```

과거 사례 `muma_2021 / preempt_short / 2-00:00:00`은 후보일 뿐입니다.
현재 association, AllowQos, MaxTime/MaxWall, GPU GRES, 선점 정책을 확인하세요.
권한상 조회 불가하면 USF RC에 확인합니다. 실제 GPU 모델은 작업 내부의
`nvidia-smi`와 PyTorch CUDA device 출력으로 확인하며 강제로 GPU 번호를 지정하지 않습니다.

## 독립된 Linux 환경과 경로

`module avail`에서 확인한 Python 3.10 모듈을 로드한 후:

```bash
python --version
python -m venv "$PWD/.venv-circe"
"$PWD/.venv-circe/bin/python" -m pip install --upgrade pip
"$PWD/.venv-circe/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PWD/.venv-circe/bin/python" -m pip install -r requirements-pusht.txt
"$PWD/.venv-circe/bin/python" -m pip check
cp scripts/circe.env.example scripts/circe.env
```

`scripts/circe.env`의 모든 경로를 실제 Linux **절대경로**로 편집합니다.
`FLOWJEPA_PYTHON`은 위 환경의 `bin/python`, 데이터는 새 Flow-JEPA 작업 저장소,
`FLOWJEPA_RUNS_ROOT`와 `FLOWJEPA_LOG_ROOT`는 보존 가능한 별도 저장소로 지정합니다.
scratch는 영구 보관 저장소가 아닙니다. `/work`와 `/work_bgfs`의 접근 범위와 삭제 정책을
확인하고 보관할 체크포인트와 로그는 영구 저장소로 지정하거나 별도로 백업합니다.
`FLOWJEPA_RUN_ID`는 `flowjepa-`로 시작하는 새 ID를 사용하세요.
공유 AdaJEPA 환경을 활성화하거나 그 디렉터리를 출력 경로로 지정하지 마세요.

```bash
source scripts/circe.env
"$FLOWJEPA_PYTHON" --version
mkdir -p "$FLOWJEPA_DATA_ROOT/datasets" "$FLOWJEPA_RUNS_ROOT" "$FLOWJEPA_LOG_ROOT/data"
```

작업 노드에서도 Python 모듈이 필요하면 `circe_train.sbatch`와
`circe_prepare.sbatch`의 `set -euo pipefail` 다음에 확인된 `module load ...`를
추가합니다. 임의의 모듈명은 가정하지 않습니다.

데이터 다운로드/압축 해제도 로그인 노드에서 장시간 실행하지 않습니다.
CPU partition/QoS를 별도로 확인하고 `CPU_PARTITION`, `CPU_QOS`를 바꾸세요:

```bash
sbatch --job-name=flowjepa-data --chdir="$FLOWJEPA_REPO" \
  --partition=CPU_PARTITION --qos=CPU_QOS --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=8G --time=01:00:00 --export=ALL \
  --output="$FLOWJEPA_LOG_ROOT/data/%x-%j.out" \
  --error="$FLOWJEPA_LOG_ROOT/data/%x-%j.err" \
  "$FLOWJEPA_REPO/scripts/circe_prepare.sbatch"
```

인터넷이 안 되는 계산 노드라면 로컬의 13.14 GB `.h5.zst`를
`$FLOWJEPA_DATA_ROOT/datasets/`에 먼저 scp로 전송하고 같은 CPU 작업으로
해시 검증·압축 해제합니다. 추출 후 46.30 GB이며 압축 파일도 보관합니다.
단일 준비 작업만 실행하고, 완료 후 다음 명령을 확인하세요:

```bash
ls -lh "$FLOWJEPA_DATA_ROOT/datasets/pusht_expert_train.h5"
```

## 1. 제출 전 명령·자원 확인, smoke 및 정식 학습

다음 스크립트는 기본적으로 **미리보기만** 수행합니다. 경로와 명령을 검토한 후
`--check`로 Slurm 검증을 하고 `--submit`을 명시해야 실제 제출됩니다.
`--test-only`는 실행 성공이나 즉시 자원 배정을 보장하지 않습니다.

```bash
source scripts/circe.env
export FLOWJEPA_TIME=00:15:00
bash scripts/circe_submit.sh smoke
bash scripts/circe_submit.sh smoke --check
smoke_job=$(bash scripts/circe_submit.sh smoke --submit)
smoke_job=${smoke_job%%;*}
echo "$smoke_job"
```

smoke는 별도 `RUN_ID-smoke` 실험에서 2 training batch + 2 validation batch를
실행하며, 매 step full-state checkpoint를 저장합니다. W&B는 끕니다.
정식 학습은 smoke가 성공한 후에만 진행하세요:

```bash
# 48시간이 현재 계정/QoS에서 허용된 경우에만 지정
export FLOWJEPA_TIME=2-00:00:00
bash scripts/circe_submit.sh train
bash scripts/circe_submit.sh train --check
# 명령과 자원을 확인하고 학습을 시작할 때만 실행
job=$(bash scripts/circe_submit.sh train --submit)
job=${job%%;*}
echo "$job"
```

로그와 Python 경로는 절대경로이고, sbatch에 `--chdir`, `--output`, `--error`를
명시합니다. sbatch가 받은 작업은 SSH/노트북 연결과 독립적으로 실행됩니다.
배치 안에서만 `srun`으로 학습하고 Slurm이 설정한 GPU 가시성을 유지합니다.

## 2. job ID와 상태

```bash
squeue -u "$USER"
squeue -j "$job" -o "%.18i %.12P %.30j %.10T %.10M %.10l %R"
scontrol show job "$job"
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

`PENDING` 이유가 자원 부족인지 QoS/시간 제한인지 확인합니다. 쉘의 `job` 값은
새 접속에 자동 복원되지 않으므로 출력된 ID를 따로 기록하세요.

## 3. 로그와 GPU

```bash
exp="$FLOWJEPA_RUNS_ROOT/$FLOWJEPA_RUN_ID"
tail -f "$FLOWJEPA_LOG_ROOT/$FLOWJEPA_RUN_ID/$FLOWJEPA_RUN_ID-$job.out"
tail -f "$FLOWJEPA_LOG_ROOT/$FLOWJEPA_RUN_ID/$FLOWJEPA_RUN_ID-$job.err"
tail -f "$exp/gpu-$job.csv"
```

로그 첫 부분에 실제 GPU 종류/메모리, CUDA device, Python 절대경로와 최종
학습 명령이 나옵니다. GPU 사용량 CSV는 30초마다 기록됩니다. `nvidia-smi`가
노드 전체 GPU를 보여주더라도 PyTorch가 사용하는 GPU는 CUDA device 출력으로
확인합니다. smoke 로그/경로는 `FLOWJEPA_RUN_ID-smoke`를 사용합니다.

## 4. 주기적 저장과 중단 후 수동 재개

- `.pt`/`config.json`: `$exp/worldmodel/checkpoints/$FLOWJEPA_RUN_ID/`
- optimizer/scheduler/진행 상태: `$exp/trainer/runs/.../checkpoints/last.ckpt`
- 기본 주기: 100 training steps. 저장 주기는 `FLOWJEPA_CHECKPOINT_STEPS`로 조절합니다.
- 비정상 종료 시 마지막으로 완전히 저장된 체크포인트 이후 작업은 잃을 수 있습니다.
- mid-epoch의 sample 순서까지 정확히 이어진다고 보장하지 않습니다.

이 제출 스크립트는 `--no-requeue`를 사용합니다. Slurm 선점/시간 제한 후 자동
재제출을 가정하지 않으며, 종료 확인 후 **같은 실험의** 체크포인트로 새 job을
제출합니다. 동시 쓰기는 실험별 flock으로 차단합니다.

```bash
exp="$FLOWJEPA_RUNS_ROOT/$FLOWJEPA_RUN_ID"
find "$exp/trainer" -type f -name last.ckpt -printf '%T@ %p\n' | sort -nr
# 위 결과에서 원하는 체크포인트의 실제 절대경로를 지정
export FLOWJEPA_RESUME=/ABSOLUTE/PATH/TO/THIS/EXPERIMENT/trainer/runs/.../checkpoints/last.ckpt
bash scripts/circe_submit.sh resume
bash scripts/circe_submit.sh resume --check
job=$(bash scripts/circe_submit.sh resume --submit)
job=${job%%;*}
```

실제 Hydra 인수는 `checkpoint.resume=...`이며 `weights_only=False`로 읽습니다.
출처를 신뢰하는 이 실험의 `.ckpt`만 사용하세요. 기존 실험에 새 학습 모드로
다시 쓰려 하면 중단하고, 다른 실험의 체크포인트도 거부합니다.

## 5. 취소

```bash
scancel "$job"
sacct -j "$job" --format=JobID,State,ExitCode
```

본인의 해당 job ID만 취소합니다. 취소 자체가 새 체크포인트 저장을 보장하지
않으므로 최근 `last.ckpt` 갱신 시각을 먼저 확인하세요.

## 6. W&B

서버에서 Flow-JEPA의 Python으로 로그인하고, API 키를 코드나 로그에 넣지 마세요.
첫 정식 제출 **전**에 활성화합니다:

```bash
"$FLOWJEPA_PYTHON" -m wandb login
export FLOWJEPA_WANDB=1
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_MODE=online
```

새 project는 `flow-jepa-pusht`, run ID는 `FLOWJEPA_RUN_ID`입니다. 새 실행은
`resume=never`, 같은 실험의 재개는 `resume=must`로 W&B run을 명시적으로 선택합니다.
기존 AdaJEPA run ID는 사용하지 않습니다. 웹 주소:
`https://wandb.ai/<ENTITY>/flow-jepa-pusht/runs/<FLOWJEPA_RUN_ID>`

`train/loss`, `train/pred_loss`, `train/sigreg_loss`와 `validate/...`를 확인하세요.
W&B는 모델 체크포인트를 대신하지 않습니다. 계산 노드 외부 통신이 안 되면:

```bash
export WANDB_MODE=offline
# 종료 후 네트워크가 되는 노드에서 해당 실험의 offline-run 경로만 동기화
"$FLOWJEPA_PYTHON" -m wandb sync "$exp/wandb/wandb/offline-run-..."
```

offline 모드에서는 online과 같은 resume 동작을 가정하지 마세요. 각 offline
segment의 경로를 로그에서 확인하고 해당 segment만 동기화하세요.

## 검증 범위와 출처

CIRCE의 실제 계정·Python·데이터 경로는 아직 미확정입니다. Slurm 제출이나
서버 GPU 테스트를 수행하지 않았습니다. 스크립트 문법과 로컬 모델 실행만
확인할 수 있으며 서버 smoke가 필요합니다.

로컬에서는 별도 테스트 디렉터리에 full-state checkpoint를 저장하고 새 프로세스로
불러와 global step 2→3 재개를 확인했습니다. 저장 파일에 optimizer state 1개와
LR scheduler state 1개가 들어 있습니다. 제출 경계 검사 6개(미리보기 무제출,
Slurm 검사 인수, 새 실험 충돌, 외부 체크포인트 거부, 같은 실험 재개 허용,
job ID 반환)도 가짜 scheduler로 통과했습니다. 실제 CIRCE 작업은 제출하지 않았습니다.

- https://wiki.rc.usf.edu/index.php/SLURM_Partitions
- https://wiki.rc.usf.edu/index.php/SLURM_Active_QOS%27
- https://wiki.rc.usf.edu/index.php/CIRCE_Layout
- https://slurm.schedmd.com/sbatch.html
- https://docs.wandb.ai/models/runs/resuming
