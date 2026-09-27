# SAM3 Makeup and Jewelry Segmentation

基于 **SAM3** 和 **MediaPipe Face Mesh** 的人像化妆与饰品分割工具。输入图片目录、单张图片或路径清单，输出与原图尺寸一致的二值 mask，适用于批量数据预处理和自动标注。

- 根据输入文件夹名称自动选择 `makeup` 或 `jewelry` 任务。
- 默认递归处理全部图片，无数量限制。
- 每张图片只保存一张二值 PNG mask。
- 保留输入子目录结构，区分同名、不同扩展名的图片。
- 输入、输出、SAM3 源码和权重位置均可配置。

## 安装

### 1. 克隆本项目

```bash
git clone https://github.com/kabuda-pixel/sam3-makeup-jewelry.git
cd sam3-makeup-jewelry
```

后续命令默认在本项目根目录执行。示例中的 `/path/to/...` 均需替换为当前机器的实际路径。

### 2. 准备 SAM3 环境

按照 [SAM3 官方安装说明](https://github.com/facebookresearch/sam3#installation) 安装 SAM3 和适配服务器的 PyTorch/CUDA 环境。官方当前要求 Python 3.12+、PyTorch 2.7+ 和支持 CUDA 12.6+ 的 GPU 环境；具体兼容性以所使用的 SAM3 版本为准。

在已激活的 SAM3 环境中安装本项目依赖：

```bash
python -m pip install -r requirements.txt "mediapipe==0.10.21"
```

本项目调用 `mediapipe.solutions.face_mesh`，上述 MediaPipe 版本与仓库的环境记录一致。`requirements.txt` 不安装 SAM3 或 PyTorch。

[environment.yml](environment.yml) 提供开发环境的完整依赖记录，其中包含特定 CUDA/PyTorch 版本；迁移到其他服务器时，应先确认目标环境的兼容性。

### 3. 准备权重

按照 [SAM3 官方权重获取说明](https://github.com/facebookresearch/sam3#getting-started) 申请访问并下载 checkpoint，将其保存到服务器本地，例如 `/path/to/models/sam3.pt`。

化妆和饰品任务共用同一份 SAM3 checkpoint。本仓库不包含模型权重；运行时必须通过 `--checkpoint-path` 指定本地权重文件。

如果使用本地 SAM3 源码，可通过 `--sam3-repo` 指定其根目录，该目录应包含 `sam3/model_builder.py`。此参数用于选择源码位置，SAM3 所需依赖仍需预先安装。

## 快速开始

### 自动识别化妆任务

输入目录的名称为 `makeup` 时，自动运行化妆分割：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_masks.py \
  --input /path/to/data/makeup \
  --output-dir /path/to/results/makeup_masks \
  --checkpoint-path /path/to/models/sam3.pt
```

### 自动识别饰品任务

输入目录的名称为 `jewelry` 时，自动运行饰品分割：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_masks.py \
  --input /path/to/data/jewelry \
  --output-dir /path/to/results/jewelry_masks \
  --checkpoint-path /path/to/models/sam3.pt
```

### 使用其他名称的数据目录

通过 `--task` 显式选择任务，例如为 CelebA-HQ 生成化妆 mask：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_masks.py \
  --input /path/to/CelebA-HQ \
  --task makeup \
  --output-dir /path/to/results/celebahq_makeup_masks \
  --checkpoint-path /path/to/models/sam3.pt \
  --sam3-repo /path/to/sam3
```

SAM3 已安装且可以直接导入时，可省略 `--sam3-repo`。GPU 通过 `CUDA_VISIBLE_DEVICES` 选择；每个进程运行一个任务，不会自动分配到多张 GPU。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 图片目录、单张图片或 `.txt` 图片路径清单 |
| `--output-dir` | 必填 | 二值 mask 输出目录，自动创建 |
| `--checkpoint-path` | 必填 | 本地 SAM3 checkpoint 文件 |
| `--task` | `auto` | `auto`、`makeup` 或 `jewelry` |
| `--sam3-repo` | 不指定 | 本地 SAM3 源码根目录；省略时使用当前 Python 环境中的 SAM3 |
| `--recursive` | 启用 | 扫描输入目录及其子目录 |
| `--no-recursive` | 不启用 | 只扫描输入目录第一层 |
| `--config` | 根据任务选择 | 自定义提示词与阈值配置 JSON |
| `--device` | `cuda` | 推理设备 |
| `--dtype` | `bfloat16` | `auto`、`float32`、`bfloat16` 或 `float16` |

默认配置为 [makeup_complex.json](configs/makeup_complex.json) 和 [jewelry.json](configs/jewelry.json)，根据项目位置自动定位。

统一入口会把任务专属参数传入对应的推理脚本。例如，可添加 `--pixel-threshold 0.5` 调整二值化阈值。查看完整参数：

```bash
python scripts/infer_masks.py --help
python scripts/infer_makeup.py --help
python scripts/infer_jewelry.py --help
```

### 自动选择任务的规则

- 目录输入：读取输入目录本身的名称，忽略大小写。
- 单张图片或 `.txt` 清单输入：读取该文件所在目录的名称。
- 自动识别只接受 `makeup` 和 `jewelry`；其他名称需显式指定 `--task`。
- 显式指定的 `--task` 优先，输出目录名称不参与判断。

## 输入与输出

### 输入规则

支持 `.jpg`、`.jpeg`、`.png`、`.webp` 和 `.bmp`，扩展名不区分大小写。

使用文本清单时，每行填写一个图片路径：

```text
images/001.jpg
images/002.png
/path/to/another/image.jpg
```

```bash
python scripts/infer_masks.py \
  --input /path/to/images.txt \
  --task jewelry \
  --output-dir /path/to/results/jewelry_masks \
  --checkpoint-path /path/to/models/sam3.pt
```

- 命令行相对路径相对于启动时的工作目录。
- 清单内的相对路径相对于清单所在目录；空行忽略，重复图片只处理一次。
- 路径支持 `~` 和环境变量展开；包含空格的命令行路径需使用引号。
- 模型加载前会检查输入图片、配置和权重文件是否存在。
- 目录输入时，输出目录必须位于输入目录之外，避免后续扫描读入已生成的 mask。

### 二值 mask

每张图片生成一张单通道 PNG，保持原图宽高：

| 像素值 | 含义 |
| --- | --- |
| `0` | 背景 |
| `255` | 预测的化妆或饰品区域 |

输出直接写入 `--output-dir`，保留子目录结构。文件名使用 **完整原文件名 + `.png`**，因此 `001.jpg` 和 `001.png` 的输出不会相互覆盖：

```text
输入 makeup/                  输出 makeup_masks/
├── 001.jpg                   ├── 001.jpg.png
├── 001.png                   ├── 001.png.png
└── person_a/                 └── person_a/
    └── 001.jpg                   └── 001.jpg.png
```

单图输入直接保存到输出目录。清单输入的子目录结构以所有列出图片的共同父目录为基准。

输出仅包含最终二值 mask；化妆任务中的眼部排除仍参与最终 mask 的计算。重复运行会覆盖对应的 mask，输出目录中的历史文件不会自动删除。需要一份干净结果时，请使用新的输出目录。

读取 mask：

```python
import numpy as np
from PIL import Image

mask = np.asarray(Image.open("/path/to/results/001.jpg.png").convert("L")) > 0
```

## Shell 启动方式

统一启动脚本接收与 Python 入口相同的参数：

```bash
bash scripts/run_masks_server.sh \
  --input /path/to/data/makeup \
  --output-dir /path/to/results/makeup_masks \
  --checkpoint-path /path/to/models/sam3.pt
```

也可以通过环境变量配置：

```bash
INPUT_DIR=/path/to/data/jewelry \
OUTPUT_DIR=/path/to/results/jewelry_masks \
CHECKPOINT=/path/to/models/sam3.pt \
SAM3_REPO=/path/to/sam3 \
CUDA_DEVICE=0 \
bash scripts/run_masks_server.sh
```

| 环境变量 | 用途 |
| --- | --- |
| `INPUT_DIR` / `OUTPUT_DIR` | 输入和输出路径 |
| `CHECKPOINT` / `SAM3_REPO` | 权重文件和 SAM3 源码根目录 |
| `TASK` / `CONFIG` | 任务类型和配置文件 |
| `CUDA_DEVICE` | 设置 `CUDA_VISIBLE_DEVICES`；省略时保留当前设置 |
| `PYTHON` | Python 解释器，默认 `python` |
| `CONDA_ENV` | 可选 Conda 环境名；省略时使用当前 Python 环境 |

环境变量由 Shell 启动脚本读取；显式命令行参数优先。
`run_makeup_server.sh` 和 `run_jewelry_server.sh` 分别预设对应任务，其余参数转发到统一入口。
不再使用 `LIMIT` 或旧版阈值环境变量；分割阈值通过命令行参数传入。

## 项目结构

```text
sam3-makeup-jewelry/
├── configs/
│   ├── makeup_complex.json          # 化妆提示词与配置
│   └── jewelry.json                 # 饰品提示词与配置
├── scripts/
│   ├── infer_masks.py               # 统一入口，自动选择任务
│   ├── infer_makeup.py              # 化妆分割
│   ├── infer_jewelry.py             # 饰品分割
│   ├── run_masks_server.sh          # 通用服务器启动脚本
│   ├── run_makeup_server.sh         # 化妆任务启动脚本
│   └── run_jewelry_server.sh        # 饰品任务启动脚本
├── src/sam3_face_attributes/
│   ├── core.py                      # 模型加载与公共处理逻辑
│   └── paths.py                     # 路径校验、图片扫描与输出命名
├── tests/
├── requirements.txt
├── environment.yml
├── SAM3_COMMIT.txt
└── SAM3_CHECKPOINT_SHA256.txt
```

## 测试与复现

运行测试：

```bash
python -m pytest -q tests
```

测试覆盖候选筛选与融合、任务选择、路径解析、超过 5 张图片的全量处理、二值输出、重名文件及启动参数。批量推理测试使用模拟模型，不需要真实 checkpoint 或 GPU，也不代表真实模型的分割精度。

复现时记录本项目版本、SAM3 源码版本、checkpoint 校验值、依赖环境、配置、运行参数及 GPU 信息。仓库提供以下参考记录：

- [SAM3_COMMIT.txt](SAM3_COMMIT.txt)：SAM3 源码提交记录。
- [SAM3_CHECKPOINT_SHA256.txt](SAM3_CHECKPOINT_SHA256.txt)：`sam3.pt` 的 SHA-256 校验记录。
- [environment.yml](environment.yml)：依赖版本记录。

在权重所在目录核对文件：

```bash
cd /path/to/models
sha256sum -c /path/to/sam3-makeup-jewelry/SAM3_CHECKPOINT_SHA256.txt
```

生成的 mask 属于自动预测结果。更换数据集后，建议先人工检查代表性样本，再用于后续训练或定量分析。
