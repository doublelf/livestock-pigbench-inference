# 环境记录（environment）

- 记录时间：2026-11-24（Asia/Shanghai）
- 主机：Windows 11 家庭版 中文版（10.0.26200），13th Gen Intel Core i5-13500H（12 核 16 线程），32 GB RAM
- GPU：Intel Iris Xe 核显；**无 NVIDIA GPU**
- 推理设备：**CPU**
- 虚拟环境：项目根目录 `.venv`（Python 3.10.11）
- 完整依赖清单：`results/pip_freeze.txt`

## 版本对照

| 组件 | 本机实际 | 官方 `_setup/setup.sh` 要求 | 差异说明 |
| --- | --- | --- | --- |
| Python | 3.10.11 | 3.10 | 一致 |
| PyTorch | **2.0.0+cpu** | 2.0.0（CUDA 11.8 版） | 版本号一致，但本机为 CPU 构建 |
| torchvision | 0.15.0+cpu | 0.15.0 | 同上 |
| CUDA | **不可用**（`torch.cuda.is_available() == False`，`torch.version.cuda == None`） | CUDA 11.8 | 本机无 NVIDIA GPU，无法安装/使用 CUDA |
| mmengine | 0.10.4 | 未锁小版本 | 兼容 mmdet 3.3.0 |
| mmcv | 2.0.0（Windows CPU wheel） | 2.0.0 | 一致；使用 `mmcv-2.0.0-cp310-cp310-win_amd64.whl` |
| mmdet | 3.3.0 | 3.3.0 | 一致 |
| mmyolo | 0.6.0 | 0.6.0 | 一致 |
| numpy | 1.26.4 | 1.26.4 | 一致 |
| opencv-python | 4.9.0.80 | 未锁版本 | 兼容 numpy 1.26 |

## 安装说明（只装检测所需，不装 MOTIP/MOTRv2）

- 官方 `_setup/setup.sh` 是“检测 + SORT 类跟踪”的最小环境；MOTIP/MOTRv2 的自定义 CUDA 算子
  本期不需要，也未安装。
- 本机 Windows 用不了官方的 conda + CUDA 11.8 组合，改为：
  `python -m venv .venv` + pip 安装 CPU 版 torch + OpenMMLab Windows CPU wheel。
- 安装脚本：`scripts/setup_env.ps1`（可重跑，日志 `results/setup_env.log`）。

## 安装过程中实际踩到的坑

1. **PyPI TLS 握手失败**：本机 Python 访问 `pypi.org` / `download.pytorch.org` 时，
   默认 TLS 1.3 握手被重置（`SSLEOFError: EOF occurred in violation of protocol`），
   而 `curl` 正常。解决：用 `scripts/openssl_tls12.cnf` 把 OpenSSL 限定到 TLS 1.2
   （`$env:OPENSSL_CONF`），并给 pip 增加重试与超时。这属于**依赖安装**问题，不涉及数据/权重下载。
2. **PowerShell 5.1 读 UTF-8 脚本**：无 BOM 的中文注释会乱码，安装脚本已转为 UTF-8 with BOM。
3. **控制台中文乱码**：Python 输出重定向到日志时默认用 GBK，后续统一加 `PYTHONUTF8=1` 写 UTF-8 日志。

## 冒烟测试（不加载权重）

`scripts/smoke_test.py` 用官方 `configs/yolov8/yolov8_s.py` 构建模型（`checkpoint=None`，随机初始化），
在仓库自带的 `data/example_images/demo_image1.jpg` 上跑通官方 `inference_detector`：

```
[smoke] 模型构建成功，参数量 11.14 M，耗时 3.32 s
[smoke] 推理完成：300 个候选框（随机权重，数量无意义），耗时 0.93 s
[smoke] 图片尺寸: 1280x800
[smoke] PASS：环境、config、推理 API 均可用
```

完整日志：`results/smoke_test.log`。随机权重输出没有语义意义，不作为任何推理结果。
