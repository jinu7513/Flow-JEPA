param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$env:STABLEWM_HOME = Join-Path $repoRoot 'storage'
$env:LOCAL_DATASET_DIR = $env:STABLEWM_HOME
$env:SDL_VIDEODRIVER = 'dummy'
$env:SPT_CACHE_DIR = Join-Path $env:STABLEWM_HOME 'training'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$datasetPath = Join-Path $env:STABLEWM_HOME 'datasets\pusht_expert_train.h5'
if (-not (Test-Path -LiteralPath $datasetPath)) {
    throw "Dataset missing: $datasetPath. Run scripts/download_pusht.py first."
}
$trainArgs = @('train.py', '--config-name=pusht_local')
if ($Smoke) {
    $trainArgs += @('+trainer.fast_dev_run=2', 'output_model_name=flow-jepa-pusht-smoke', 'subdir=flow-jepa-pusht-smoke')
}
# Detect native failures by exit code, allowing Python stderr warnings in logs.
$ErrorActionPreference = 'Continue'
& $pythonExe @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
