<div align="center">

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Abstract%20digital%20art%20banner%20for%20video%20frame%20analysis%20plugin%2C%20film%20strip%20with%20keyframe%20markers%2C%20camera%20motion%20arrows%2C%20scene%20detection%20visualization%2C%20dark%20blue%20and%20teal%20color%20scheme%2C%20modern%20tech%20style%2C%20clean%20minimal%20design&image_size=landscape_16_9" width="100%" alt="Banner" />

# ComfyUI Video Frame Yunjii

[![GitHub](https://img.shields.io/badge/GitHub-yunjii--cn%2Fcomfyui--video--frame--yunjii-181717?logo=github)](https://github.com/yunjii-cn/comfyui-video-frame-yunjii)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-FF6600?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0iI0ZGNjYwMCIvPjwvc3ZnPg==)](https://www.comfy.org/)
[![License](https://img.shields.io/badge/License-Copyright%202026-blue)]()

**云集智能视频帧处理插件** | Yunjii Video Frame Processing Plugin

自动提取视频关键帧、检测镜头切换、分析运镜方式，并生成结构化提示词<br>为 AI 视频生成工作流提供精准的帧级控制

Automatically extract keyframes, detect scene cuts, analyze camera motion, and generate structured prompts<br>providing precise frame-level control for AI video generation workflows

</div>

---

## 🌟 功能特性 / Features

<table>
<tr>
<td width="50%">

### 🎬 镜头检测 / Scene Detection
基于 HSV 直方图相关性的自动镜头边界检测，精准识别画面切换点

</td>
<td width="50%">

### 📐 三种分段模式 / Segmentation Modes
自然镜头 / 均匀时长 / 固定段数，灵活适配不同视频内容

</td>
</tr>
<tr>
<td width="50%">

### 🎥 运镜分析 / Motion Analysis
基于 Farneback 光流的运镜识别：平移、俯仰、静态、对角线等

</td>
<td width="50%">

### 👤 人物检测 / Person Detection
基于边缘密度的画面人物判断，自动追加质量提示词

</td>
</tr>
<tr>
<td width="50%">

### 🖼 关键帧提取 / Keyframe Extraction
自动选取每段中间帧，支持手动选帧微调和帧偏移

</td>
<td width="50%">

### 📝 提示词控制 / Prompt Control
前缀 / 后缀 / 替换 / 合并四种模式，灵活组合分段提示词

</td>
</tr>
<tr>
<td width="50%">

### 🕺 姿态提取 / Pose Extraction
基于 MediaPipe 的视频姿态提取，内置时序平滑和帧间插值

</td>
<td width="50%">

### 🎭 模仿提示词 / Mimic Prompt
根据运镜分析自动生成优化的视频生成提示词

</td>
</tr>
<tr>
<td width="50%">

### 📤 视频上传 / Video Upload
前端一键上传视频到 ComfyUI input 目录

</td>
<td width="50%">

### 🔍 手动选帧浏览器 / Frame Browser
可视化时间轴浏览视频，点击标注/取消关键帧

</td>
</tr>
</table>

---

## 📦 安装 / Installation

### 方式一：通过 ComfyUI-Manager（推荐 / Recommended）

在 ComfyUI-Manager 中搜索 `comfyui-video-frame-yunjii` 并安装。

Search for `comfyui-video-frame-yunjii` in ComfyUI-Manager and install.

### 方式二：手动安装 / Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yunjii-cn/comfyui-video-frame-yunjii.git
```

### 依赖 / Dependencies

| 依赖 | 说明 |
|------|------|
| `opencv-python` (cv2) | 视频读取、光流计算、图像处理 |
| `numpy` | 数值计算 |
| `torch` | 图像张量输出 |

> ComfyUI 环境通常已包含以上依赖，无需额外安装。

---

## 🧩 节点说明 / Node Reference

所有节点位于 **`Yunjii/Video`** 分类下。

All nodes are under the **`Yunjii/Video`** category.

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Three%20connected%20nodes%20in%20a%20node-based%20editor%20UI%2C%20dark%20theme%2C%20showing%20video%20analysis%20workflow%2C%20motion%20analysis%20node%20connected%20to%20prompt%20control%20and%20keyframe%20preview%2C%20professional%20schematic%20style&image_size=landscape_16_9" width="100%" alt="Node Overview" />

---

### 1️⃣ 运动分析 🔍 (Yunjii) — `MotionAnalysisNode`

> 核心节点。对视频进行镜头检测、运镜分析、人物判断和关键帧提取。
>
> The core node. Performs scene detection, motion analysis, person detection, and keyframe extraction.

#### 输入 / Inputs

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 视频文件 | COMBO | — | 选择 `input/` 目录中的视频文件 |
| 分段模式 | COMBO | 自然镜头 | `自然镜头` / `均匀时长` / `固定段数` |
| 限制最长秒数 | BOOLEAN | True | 超过最大秒数的段自动拆分 |
| 每段最大秒数 | FLOAT | 10.0 | 单段最大时长（2.0–60.0 秒） |
| 限制最短秒数 | BOOLEAN | True | 短于最小秒数的段合并到相邻段 |
| 每段最小秒数 | FLOAT | 2.0 | 单段最小时长（0.5–10.0 秒） |
| 固定段数 | INT | 5 | 固定段数模式下的段数（1–50） |
| 灵敏度 | FLOAT | 0.3 | 镜头检测灵敏度（0.1–1.0），越高越容易检测到切换 |
| 保存关键帧 | BOOLEAN | True | 是否将关键帧保存到 `output/keyframes_*/` |
| 视频路径 | STRING | "" | 可选，直接指定视频完整路径 |
| 合并指令 | STRING | "" | 如 `1+2` 合并第1和第2段，多条用逗号分隔 |
| 拆分指令 | STRING | "" | 如 `3:2` 把第3段拆成2段，多条用逗号分隔 |
| 帧偏移 | STRING | "" | 微调关键帧位置，逗号分隔，如 `0,-5,10` |
| 人物质量词 | STRING | "eyes open, clear face, sharp focus" | 有人镜头自动追加的质量提示词 |
| 手动选帧 | STRING | "" | 手动选择的帧号，逗号分隔，如 `45,120,195` |

#### 输出 / Outputs

| 输出 | 类型 | 说明 |
|------|------|------|
| 运动提示词 | STRING | 各段运镜提示词，用 `\|\|\|` 分隔 |
| 分段信息 | STRING | 完整的分析报告文本 |
| 镜头数 | INT | 检测到的镜头/段数量 |
| 关键帧 | IMAGE | 所有关键帧的图像张量 (B, H, W, 3) |
| 帧信息 | STRING | 每帧的摘要信息，换行分隔 |

#### 分段模式说明 / Segmentation Modes

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Diagram%20showing%20three%20video%20segmentation%20modes%20side%20by%20side%3A%20natural%20scene%20detection%20with%20varying%20segment%20lengths%2C%20even%20duration%20with%20equal%20segments%2C%20fixed%20count%20with%20specified%20number%20of%20segments%2C%20clean%20infographic%20style%2C%20dark%20background%2C%20teal%20and%20orange%20accents&image_size=landscape_16_9" width="100%" alt="Segmentation Modes" />

| 模式 | 说明 |
|------|------|
| 🎬 **自然镜头 / Natural Scene** | 基于 HSV 直方图相关性检测镜头切换边界，自动合并过短段、拆分过长段 |
| ⏱️ **均匀时长 / Even Duration** | 按最大秒数均分视频，优先在检测到的镜头边界处切分 |
| 🔢 **固定段数 / Fixed Count** | 指定段数均分视频，优先在检测到的镜头边界处切分 |

#### 合并/拆分指令 / Merge & Split Commands

```
合并: 1+2,4+5    → 合并第1段和第2段，合并第4段和第5段
拆分: 3:2,7:3    → 第3段拆成2段，第7段拆成3段
```

---

### 2️⃣ 提示词控制 📝 (Yunjii) — `PromptControlNode`

> 对运动分析节点生成的分段提示词进行灵活组合和覆盖。
>
> Flexibly combine and override segment prompts generated by the Motion Analysis node.

#### 输入 / Inputs

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 自动提示词 | STRING | "" | 由运动分析节点生成的提示词，用 `\|\|\|` 分隔 |
| 自定义前缀 | STRING | "" | 每段提示词的前缀，用 `\|\|\|` 分隔 |
| 自定义后缀 | STRING | "" | 每段提示词的后缀，用 `\|\|\|` 分隔 |
| 模式 | COMBO | 合并 | `合并` / `替换` / `前缀` / `后缀` |
| 分段数 | INT | 0 | 指定段数，0=自动检测 |

#### 输出 / Outputs

| 输出 | 类型 | 说明 |
|------|------|------|
| 分段提示词 | STRING | 处理后的分段提示词，用 `\|\|\|` 分隔 |
| 提示词详情 | STRING | 每段提示词的详细展示 |

#### 模式说明 / Mode Description

| 模式 | 逻辑 | 示例 |
|------|------|------|
| 🔗 **合并 / Merge** | `前缀 + 自动 + 后缀`，空值跳过 | `cinematic, static shot, 4k` |
| 🔄 **替换 / Replace** | 有前缀则用前缀替换自动提示词 | `epic landscape` → 替换 `static shot` |
| ⬅️ **前缀 / Prefix Only** | `前缀 + 自动` | `cinematic, static shot` |
| ➡️ **后缀 / Suffix Only** | `自动 + 后缀` | `static shot, 4k quality` |

---

### 3️⃣ 关键帧预览 🖼 (Yunjii) — `KeyframePreviewNode`

> 在节点上直接显示关键帧的帧信息卡片，方便确认提取结果。
>
> Displays a frame info card directly on the node for quick verification of extracted keyframes.

#### 输入 / Inputs

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 关键帧 | IMAGE | — | 来自运动分析节点的关键帧图像 |
| 帧信息 | STRING | "" | 来自运动分析节点的帧信息文本 |

#### 输出 / Outputs

| 输出 | 类型 | 说明 |
|------|------|------|
| 关键帧 | IMAGE | 透传关键帧图像，可连接到 PreviewImage 等节点 |
| 帧信息 | STRING | 透传帧信息文本 |

---

### 4️⃣ 姿态提取 🕺 (Yunjii) — `VideoPoseExtractor`

> 从参考视频中逐帧提取人体姿态，输出 OpenPose 格式姿态图序列。基于 MediaPipe 自研实现，内置时序平滑和帧间插值。
>
> Extract body poses frame-by-frame from a reference video, outputting OpenPose-format pose image sequences. Self-developed based on MediaPipe with built-in temporal smoothing and frame interpolation.

#### 输入 / Inputs

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 视频文件 | COMBO | — | 选择参考视频文件 |
| 目标帧数 | INT | 81 | 输出帧数（4k+1格式，如81, 85, 89），匹配视频生成模型 |
| 检测身体 | BOOLEAN | True | 检测身体骨架 |
| 检测手部 | BOOLEAN | True | 检测手部关键点 |
| 检测面部 | BOOLEAN | True | 检测面部关键点 |
| 时序平滑 | BOOLEAN | True | 对关键点进行时序平滑，减少帧间抖动 |
| 平滑窗口 | INT | 5 | 平滑窗口大小（3-15），越大越平滑 |
| 输出分辨率 | INT | 512 | 姿态图分辨率 |
| 视频路径 | STRING | "" | 可选，直接指定视频完整路径 |

#### 输出 / Outputs

| 输出 | 类型 | 说明 |
|------|------|------|
| 姿态图 | IMAGE | OpenPose 格式姿态图序列 (T, H, W, 3) |
| 姿态数据 | STRING | 每帧关键点 JSON 数据 |
| 帧数 | INT | 实际输出帧数 |

#### 自研优势 / Self-Developed Advantages

| 优化 | 说明 |
|------|------|
| 🎯 **时序平滑** | 加权置信度平滑算法，消除逐帧检测抖动 |
| 🔗 **帧间插值** | 采样帧之间自动线性插值，生成流畅姿态序列 |
| 🎨 **OpenPose 渲染** | 标准 OpenPose 配色方案，兼容 Wan2.1 ControlNet |
| ⚡ **速度优势** | MediaPipe 比 DWPose 快 5-10 倍 |

---

### 5️⃣ 模仿提示词 🎭 (Yunjii) — `MimicPromptGenerator`

> 根据运镜分析结果和人物描述，生成优化的视频生成提示词（正面+负面）。
>
> Generate optimized video generation prompts (positive + negative) based on motion analysis and character description.

#### 输入 / Inputs

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 运动提示词 | STRING | "" | 由运动分析节点生成的提示词，用 \|\|\| 分隔 |
| 人物描述 | STRING | "a person" | 目标人物的外貌描述（英文） |
| 风格关键词 | STRING | "cinematic, high quality, 4k" | 视频风格关键词 |
| 质量增强 | BOOLEAN | True | 自动追加质量增强关键词 |
| 自定义负面提示词 | STRING | "" | 追加到默认负面词后 |

#### 输出 / Outputs

| 输出 | 类型 | 说明 |
|------|------|------|
| 正面提示词 | STRING | 组合后的正面提示词 |
| 负面提示词 | STRING | 组合后的负面提示词 |

---

## 🔄 典型工作流 / Typical Workflow

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Flowchart%20diagram%20of%20video%20processing%20workflow%2C%20showing%20video%20input%20going%20into%20motion%20analysis%20node%2C%20then%20branching%20to%20prompt%20control%20and%20keyframe%20preview%2C%20clean%20modern%20design%2C%20dark%20background%2C%20neon%20blue%20connections%2C%20professional%20technical%20illustration&image_size=landscape_16_9" width="100%" alt="Workflow Diagram" />

```
┌─────────────────┐     运动提示词      ┌─────────────────┐
│                 │────────────────────▶│                 │
│  运动分析 🔍     │     分段信息        │  提示词控制 📝    │──▶ 下游视频生成节点
│  MotionAnalysis │                     │  PromptControl  │
│                 │────── 镜头数 ──────▶│                 │
│                 │                     └─────────────────┘
│                 │     关键帧
│                 │────────────────────┐
│                 │     帧信息         │
│                 │────────────────────┤
└─────────────────┘                    ▼
                              ┌─────────────────┐     关键帧     ┌──────────────┐
                              │                 │──────────────▶│              │
                              │  关键帧预览 🖼    │               │ PreviewImage │
                              │  KeyframePreview│               │              │
                              └─────────────────┘               └──────────────┘
```

### 步骤 / Steps

1. **运动分析 🔍** — 处理视频，输出分段提示词、关键帧和帧信息
2. **提示词控制 📝** — 组合/覆盖提示词，连接到下游视频生成节点
3. **关键帧预览 🖼** — 展示帧信息卡片，同时透传关键帧到 PreviewImage 预览

---

## 🎬 手动选帧浏览器 / Frame Browser

运动分析节点内置「🎬 手动选帧」按钮，点击后打开全屏帧浏览器：

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Video%20frame%20browser%20interface%20mockup%2C%20dark%20theme%2C%20showing%20video%20preview%20area%20with%20crosshair%20cursor%2C%20timeline%20with%20film%20strip%20thumbnails%20and%20scene%20markers%2C%20mark%20unmark%20clear%20buttons%2C%20professional%20video%20editing%20tool%20UI&image_size=landscape_16_9" width="100%" alt="Frame Browser" />

| 操作 | 说明 |
|------|------|
| 🖱️ **时间轴拖拽** | 滑动时间轴浏览视频帧 |
| 👆 **左键点击画面** | 标注当前帧为关键帧 |
| 👇 **右键点击画面** | 取消标注当前帧 |
| 🎞️ **胶片条** | 时间轴上方显示视频缩略图条 |
| 🏷️ **镜头标记** | 自动显示各镜头范围和中间帧位置 |
| ✅ **应用到节点** | 将手动选帧写回节点参数 |

---

## 📁 文件结构 / File Structure

```
comfyui-video-frame-yunjii/
├── __init__.py          # 插件入口，注册节点和视频上传路由
├── nodes.py             # 节点定义和核心算法
├── js/
│   └── widgets.js       # 前端交互：手动选帧浏览器、帧信息卡片
├── workflows/           # 示例工作流
│   └── 云集智能视频模仿_20260518_0714.json
└── .gitignore
```

---

## ⚙️ 技术细节 / Technical Details

### 🎬 镜头检测算法 / Scene Detection Algorithm

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Technical%20diagram%20of%20HSV%20histogram%20correlation%20for%20scene%20detection%2C%20showing%20frame%20comparison%20pipeline%2C%20histogram%20overlay%20on%20video%20frames%2C%20correlation%20graph%20with%20threshold%20line%2C%20scientific%20visualization%20style%2C%20dark%20background&image_size=landscape_16_9" width="100%" alt="Scene Detection Algorithm" />

基于 HSV 色彩空间直方图相关性比较：

1. 将每帧缩放至 **160×90**
2. 计算 **32×32** 的 H-S 二维直方图
3. 使用 `cv2.compareHist` (HISTCMP_CORREL) 与前一帧比较
4. 当相关系数低于阈值 `1.0 - sensitivity × 0.5` 时，判定为镜头切换

### 🎥 运镜分析算法 / Motion Analysis Algorithm

使用 **Farneback 光流法** (`cv2.calcOpticalFlowFarneback`) 计算相邻帧间的像素运动：

| 光流特征 | 判定结果 |
|----------|----------|
| 水平光流占优 (fx > fy) | `camera panning left/right` |
| 垂直光流占优 (fy > fx) | `camera tilting up/down` |
| 双向光流 (fx > 0 且 fy > 0) | `camera moving diagonally` |
| 光流极小 (fx < 0.3 且 fy < 0.3) | `static shot` |
| 帧间变化 > 50% | `major scene change` |
| 帧间变化 > 25% | `significant movement` |

### 👤 人物检测算法 / Person Detection Algorithm

基于边缘密度启发式方法：

1. 对每段取 **1/4、1/2、3/4** 位置的三帧
2. 计算 **Canny 边缘密度**
3. 当下半区域边缘密度 > **5%** 且整体边缘密度 > **4%** 时，判定为有人物画面
4. 有人物镜头自动追加 `人物质量词`（如 "eyes open, clear face, sharp focus"）

> ⚠️ 此方法为轻量级近似，适用于快速预筛选，不替代专业人物检测模型。

---

## 🛠️ 开发 / Development

### 版本号规则 / Versioning

采用 `YYYYMMDD_HHMM` 格式（如 `20260518_0714`）。每次修改插件或工作流时，必须以新版本号创建新文件，不覆盖旧版本。

### 本地开发 / Local Development

```bash
cd ComfyUI/custom_nodes/comfyui-video-frame-yunjii
# 修改代码后重启 ComfyUI 即可生效
```

### 验证导入 / Verify Import

```bash
python -c "import sys; sys.path.insert(0, r'ComfyUI路径'); import importlib; mod = importlib.import_module('custom_nodes.comfyui-video-frame-yunjii.nodes'); print('OK')"
```

---

## 📄 许可 / License

Copyright © 2026 Yunjii (云集智能). All rights reserved.
