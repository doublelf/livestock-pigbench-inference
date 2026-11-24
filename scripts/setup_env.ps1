# 创建独立 Python 环境并安装 PigBench 检测所需依赖。
# 说明：本脚本只安装软件包（pip），不下载任何数据集或模型权重。
# 版本尽量与官方 _setup/setup.sh 对齐：Python 3.10 / torch 2.0.0 / mmcv 2.0.0 / mmdet 3.3.0 / mmyolo 0.6.0。
# 本机无 NVIDIA GPU，因此安装 CPU 版 torch。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$log = Join-Path $root "results\setup_env.log"
$cnf = Join-Path $root "scripts\openssl_tls12.cnf"
if (-not (Test-Path $cnf)) { throw "缺少 OpenSSL 配置：$cnf" }

# 本机 Python 在 TLS 1.3 握手时会被重置（SSLEOFError），限定 TLS 1.2 后可稳定访问 PyPI；
# 同时给 pip 增加重试次数与超时。
$env:OPENSSL_CONF = $cnf
$env:PIP_DEFAULT_TIMEOUT = "120"
$env:PIP_RETRIES = "10"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
Write-Output "=== [setup] OPENSSL_CONF=$cnf"

function Invoke-Pip {
    param([string[]]$PipArgs)
    & $py -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed (exit $LASTEXITCODE): pip $($PipArgs -join ' ')"
    }
}

Start-Transcript -Path $log -Append | Out-Null
try {
    Write-Output "=== [setup] project root: $root"
    Write-Output "=== [setup] python launcher: $(python --version 2>&1)"

    if (-not (Test-Path $py)) {
        Write-Output "=== [setup] creating venv at $venv"
        python -m venv $venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $py)) { throw "venv creation failed" }
    } else {
        Write-Output "=== [setup] venv already exists"
    }

    Write-Output "=== [setup] upgrading pip (best effort; 失败不阻断)"
    & $py -m pip install -U pip
    if ($LASTEXITCODE -ne 0) { Write-Output "=== [setup] WARN: pip upgrade skipped" }

    Write-Output "=== [setup] installing CPU torch 2.0.0 / torchvision 0.15.0"
    Invoke-Pip @("install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.0.0+cpu", "torchvision==0.15.0+cpu")

    Write-Output "=== [setup] installing numpy / mmengine"
    Invoke-Pip @("install", "numpy==1.26.4", "mmengine==0.10.4")

    Write-Output "=== [setup] installing mmcv 2.0.0 (Windows CPU wheel)"
    Invoke-Pip @("install", "mmcv==2.0.0", "-f", "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html")

    Write-Output "=== [setup] installing mmdet 3.3.0 / mmyolo 0.6.0"
    Invoke-Pip @("install", "mmdet==3.3.0", "mmyolo==0.6.0")

    Write-Output "=== [setup] installing runtime helpers"
    Invoke-Pip @("install", "opencv-python==4.9.0.80", "pycocotools==2.0.7", "matplotlib==3.7.5", "pandas==2.0.3", "tqdm==4.66.5", "pyyaml==6.0.2", "scipy==1.10.1", "terminaltables==3.1.10")

    Write-Output "=== [setup] versions"
    & $py -c "import sys, torch, mmcv, mmdet, mmyolo, numpy, cv2; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); print('mmyolo', mmyolo.__version__); print('numpy', numpy.__version__); print('opencv', cv2.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "import check failed" }
    Write-Output "=== [setup] done"
}
finally {
    Stop-Transcript | Out-Null
}
