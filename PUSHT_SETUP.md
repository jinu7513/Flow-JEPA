# Windows PushT 설정

작업 폴더: `D:\Flow-JEPA`

현재 origin은 본인 포크인 `https://github.com/jinu7513/Flow-JEPA.git`입니다.
Python 환경은 `.venv`, 데이터와 체크포인트는 Git에서 제외한 `storage`에 저장합니다.

## 설치 재현

```powershell
uv venv --python=3.10 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe -r requirements-pusht.txt
.venv/Scripts/python.exe scripts/download_pusht.py
```

공식 데이터: https://huggingface.co/datasets/quentinll/lewm-pusht

다운로드 13.14 GB, 압축 해제 후 46.30 GB입니다. 다운로드는 중단 후 다시 실행하면
이어서 받고, SHA-256 검증 후 압축을 풉니다. 압축 파일도 보관합니다.

## 학습

```powershell
cd D:\Flow-JEPA
# 짧은 검증: 학습 2배치 + 검증 2배치
.\scripts\train_pusht.ps1 -Smoke
# 전체 20 epoch 학습
.\scripts\train_pusht.ps1
```

`pusht_local.yaml`은 RTX 4050 6 GB용 시작 설정입니다. 배치 크기 8, BF16 mixed
precision, 데이터 로더 worker 0을 사용합니다. 논문의 배치 크기 128과 다르므로
논문 성능의 정확한 재현 설정은 아닙니다.

학습 실행기는 UTF-8 로그와 `storage/training` 출력 경로를 설정합니다.
`train.py`에는 stable-pretraining 0.1.8의 Windows 미지원 signal 진단을
우회하는 수정과 설정된 seed 적용이 들어 있습니다.

전체 학습 모델: `storage/checkpoints/flow-jepa-pusht-local/weights_epoch_20.pt`

## 평가

```powershell
$env:STABLEWM_HOME = "$PWD/storage"
$env:LOCAL_DATASET_DIR = $env:STABLEWM_HOME
$env:SDL_VIDEODRIVER = 'dummy'
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe eval.py --config-name=pusht policy="$env:STABLEWM_HOME/checkpoints/flow-jepa-pusht-local/weights_epoch_20.pt" eval.num_eval=1 solver.num_samples=30 solver.topk=5 solver.n_steps=5
```

위 명령은 적은 CEM 샘플로 실행을 확인합니다. 정식 평가는
`config/eval/pusht.yaml`과 `config/eval/solver/cem.yaml`의 기본 평가 설정을 사용하세요.
`.pt` 파일 옆에 `config.json`이 있어야 합니다. Smoke 체크포인트는 실행 확인용이며
성능 평가용으로 학습된 모델이 아닙니다.

## 검증 결과

- CUDA에서 RTX 4050을 인식했고 PushT reset/step 검사를 통과했습니다.
- `uv pip check`로 설치 패키지 의존성을 확인했습니다.
- 데이터 아카이브의 SHA-256 검사를 통과했습니다.
- 배치 크기 8로 학습 2배치와 검증 2배치를 완료했습니다.
- 테스트 모델과 `config.json`이 `storage/checkpoints/flow-jepa-pusht-smoke`에 저장됐습니다.
- 검증 로그: `storage/pusht-smoke.log`
- 전체 20 epoch 학습은 아직 시작하지 않았습니다.
