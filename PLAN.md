# 云集智能 · 视频生视频引擎 — 开发计划

> **产品代号**: Yunjii V2V Engine
> **产品定位**: 打破图生视频的边界，定义视频生视频的新范式
> **核心理念**: 视频不是终点，而是新视频的起点
> **产品形态**: 跨平台核心引擎 → 多形态部署（插件/模块/独立应用/线上平台）

---

## 零、实战进度同步（代码真实状态，2026-07-26 更新）

> 本计划文档是产品愿景；下方为工程真实进度，避免"计划"与"代码"脱节。

### 已落地（可用）
- **V2V 引擎主链路**：`planner → runner(DirectAdapter 内联执行) → stitcher` 全通，已注册 8 个节点（5 分析 + 3 引擎）。原文档把 Phase 1 标为 🔴 未完成，实际**已完成**（链路能跑真实 WanVideo 生成）。
- **链式执行死锁**：原文档列为"关键未解问题"，已由 `DirectAdapter`（内联 `PromptExecutor`，绕过 HTTP 队列）解决。
- **两个严重 bug 已修复**：
  - S1 首段过短静默丢帧（planner.py）→ 改为向前扩展保留片头。
  - S2 超时并发结果损坏（direct.py）→ 超时时 `interrupt_processing` 打断残留线程并重建独立 executor。
- **SCAIL-2 无骨架动作迁移后端（接口联通，2026-07-26）**：
  - 新增 `engine/adapters/scail.py`（`SCAILAdapter` 继承 `DirectAdapter`，复用执行核心），Runner 新增「生成后端」参数可在 **骨骼路线(WanVideo)** / **SCAIL-2 路线** 间切换。
  - 新增 `workflows/scail2_template.json`（SCAIL-2 节点模板）。
  - 已用桩注入做接口联通验证：discover_nodes 识别全部 SCAIL 节点；段间**角色身份参考图始终锁定用户参考图（无身份漂移）**，`previous_frames` 正确串联，驱动视频按段偏移——即「长视频完美模仿」的基础已打通。
  - **待真模型验证**：本机需下载 SCAIL-2 14B 权重 + SAM3，在真实 ComfyUI 中跑一次确认 widget 名与 14B 推理链路。

### 长视频完美模仿路线（当前目标）
- 复用现有 `planner`（按镜头/运动切段）+ `runner`（分段链式）+ `stitcher`（交叉淡化拼接）。
- SCAIL-2 后端：每段 = 一次 `WanSCAILToVideo`（驱动视频 `skip_first_frames` 偏移 + `previous_frames` 串联 + 角色参考图恒定），stitcher 做段间平滑。
- 与 SCAIL-2 原生 Base/Extend 长视频分块互补：我们的分段负责"镜头/风格编排"，SCAIL 的原生分块负责"单镜头内的时长延展"。

### 仍待办（P 级）
- **P1**：清理死代码（`effects/mimic.py` 空实现、`discovery.py`/`adapters/comfyui.py` 孤儿）、把本计划与两份技术文档对齐成真实状态。
- **P2**：归档根目录 ~30 个临时调试脚本与 `app/` 下 10 个 ps1、24 个 workflow 版本 JSON。
- **P3**：落地 Phase 2 效果模块（运镜/创意/增强），把"帧分析插件"真正升级成 V2V 引擎。

---

## 一、产品愿景

### 1.1 一句话定义

**云集智能 V2V Engine** — 从视频中理解运动、结构、节奏与意图，智能编排生成全新视频。

### 1.2 为什么是"视频生视频"而不是"模仿视频"

```
图生视频 (Image → Video) 的局限:
  · 一张图 → 只能生成几秒的短视频
  · 无法理解运动意图和节奏
  · 无法控制运镜和转场
  · 无法生成长视频
  · 用户需要反复手动调参

视频生视频 (Video → Video) 的革命:
  · 视频包含完整的时间维度信息
  · 运动意图、节奏、运镜、转场全部可提取
  · 可以理解"这个人想做什么"而不仅仅是"这个人长什么样"
  · 可以智能编排：模仿、创意、增强、变换
  · 可以自动生成长视频（链式生成 + 无缝拼接）
  · 用户只需说"我要什么效果"，引擎自动完成
```

### 1.3 效果模块体系

模仿视频只是触发点。V2V Engine 提供 4 大可组合效果模块，模块之间可以叠加、互补、任意组合：

```
┌─────────────────────────────────────────────────────────────────┐
│                  Yunjii V2V 效果模块体系（可叠加组合）             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🎭 模仿模块 (Mimic) — 控制动作来源                              │
│  ├── 完全模仿：1:1 还原原视频的动作和运镜                        │
│  ├── 动作迁移：提取原视频动作，迁移到新人物+新场景               │
│  ├── 人物替换：保持动作和场景，仅替换人物                        │
│  └── 风格迁移：保持动作和结构，改变视觉风格                      │
│                                                                 │
│  🎬 运镜模块 (Cinematic) — 控制镜头语言                          │
│  ├── 炫酷运镜：自动添加推拉摇移跟等电影级运镜                    │
│  ├── 转场特效：场景间自动添加创意转场（缩放/旋转/模糊/故障）      │
│  ├── 一镜到底：零接缝连续长镜头                                 │
│  └── 多角度：同一动作生成多个视角                                │
│                                                                 │
│  ✨ 创意模块 (Creative) — 控制创意表达                           │
│  ├── 动态增强：在原动作基础上增加夸张/特效动作                   │
│  ├── 节奏重构：根据音乐节拍重新编排动作节奏                      │
│  ├── 混合融合：多个视频的动作/场景/风格交叉融合                  │
│  └── 超现实：在保持结构的前提下创造超现实效果                    │
│                                                                 │
│  🔧 增强模块 (Enhance) — 控制输出品质                            │
│  ├── 画质提升：低分辨率→高分辨率，保持动作一致                   │
│  ├── 帧率提升：低帧率→高帧率，补间平滑                          │
│  ├── 去除抖动：稳定画面，去除手持抖动                            │
│  └── 背景替换：保持人物动作，替换背景场景                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

模块组合示例：

  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 动作迁移  │ + │ 炫酷运镜  │ + │ 节奏重构  │ + │ 画质提升  │
  │ (模仿)   │   │ (运镜)   │   │ (创意)   │   │ (增强)   │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  提取动作姿态    注入推拉运镜     对齐音乐节拍    超分辨率输出
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                            │
                            ▼
              新人物在新场景，炫酷运镜，踩着节拍，高清输出

  其他组合：
  · 人物替换 + 转场特效 + 帧率提升
  · 完全模仿 + 一镜到底 + 去除抖动
  · 动作迁移 + 动态增强 + 超现实 + 背景替换
  · 风格迁移 + 炫酷运镜 + 节奏重构
```

**模块叠加的执行管线**:

```
输入视频
   │
   ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: 模仿模块 (决定动作来源)                 │
│  从参考视频中提取/变换姿态序列                     │
│  输出：目标姿态序列 + 参考图策略                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: 创意模块 (决定动作表达)                 │
│  对姿态序列进行增强/重映射/融合                    │
│  输出：变换后的姿态序列 + 创意提示词               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: 运镜模块 (决定镜头语言)                 │
│  注入运镜参数 + 编排转场                          │
│  输出：运镜提示词 + 转场方案 + 分段策略           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 4: 链式生成 (执行生成)                     │
│  综合所有模块的输出，逐段生成                      │
│  输出：多个视频片段                                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 5: 增强模块 (决定输出品质)                 │
│  拼接 + 超分/插帧/稳定/换背景                     │
│  输出：最终成品视频                                │
└─────────────────────────────────────────────────┘
```

### 1.4 产品形态矩阵

```
                    ┌──────────────────────────────────┐
                    │     Yunjii V2V Core Engine        │
                    │     (跨平台 Python 核心引擎)       │
                    │                                  │
                    │  · 视频分析 (Analysis)            │
                    │  · 智能分段 (Segmentation)        │
                    │  · 链式生成 (Chain Generation)    │
                    │  · 无缝拼接 (Stitching)           │
                    │  · 效果编排 (Effect Orchestration) │
                    └──────────┬───────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │  ComfyUI 适配器 │  │  创意站模块    │  │  独立应用      │
  │               │  │               │  │               │
  │  自定义节点    │  │  后端服务模块  │  │  桌面应用      │
  │  前端 Widget   │  │  API 路由     │  │  Web UI       │
  │  Queue API    │  │  Pipeline集成 │  │  CLI 工具      │
  └───────────────┘  └───────────────┘  └───────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
  ComfyUI 工作流      云集智能视频创意站    独立桌面/线上平台
  (当前阶段)          (模块集成)           (未来产品化)
```

### 1.5 关键指标

| 指标 | 目标 |
|------|------|
| 用户操作 | 上传视频 → 选效果模式 → 点生成 |
| 自动化率 | 95%+（引擎自动分析、分段、编排、生成） |
| 效果模块 | 4 大类 13+ 子效果，可任意叠加组合 |
| 段间接缝 | 一镜到底：零接缝；其他模式：亚帧级过渡 |
| 动态适应 | 根据运动复杂度自动调整每段长度和参数 |
| 错误恢复 | 失败段自动重试 3 次，断点续跑 |
| 跨平台 | Windows / macOS / Linux |
| 部署形态 | ComfyUI 插件 / Python 库 / 桌面应用 / Web API |

---

## 二、核心引擎架构：Yunjii V2V Core

> **设计原则**: 核心引擎与任何特定平台解耦，通过适配器层对接不同部署形态

### 2.1 引擎分层

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Layer 5: 效果编排层 (Effect Orchestration) — 模块可叠加组合      │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Mimic   │ │ Cinematic│ │ Creative │ │ Enhance  │           │
│  │ 模仿模块 │ │ 运镜模块  │ │ 创意模块  │ │ 增强模块  │           │
│  │ Stage 1 │ │ Stage 3  │ │ Stage 2  │ │ Stage 5  │           │
│  └─────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  执行管线: 模仿(动作来源) → 创意(动作表达) → 运镜(镜头) → 生成 → 增强(品质) │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 4: 智能调度层 (Intelligent Orchestration)                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ SegmentPlanner│ │SegmentRunner │ │SegmentStitcher│            │
│  │ 动态分段规划   │ │ 链式执行引擎  │ │ 无缝拼接器    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 3: 视频分析层 (Video Analysis)                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ Motion     │ │ Pose       │ │ Content    │ │ Rhythm     │  │
│  │ 运动分析    │ │ 姿态提取    │ │ 内容理解    │ │ 节奏分析    │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 2: 生成适配层 (Generation Adapter)                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ ComfyUI    │ │ LTX Studio │ │ Direct     │ │ Cloud API  │  │
│  │ 适配器     │ │ 适配器      │ │ Pipeline   │ │ 适配器      │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 基础设施层 (Infrastructure)                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ Video I/O  │ │ Image Proc │ │ Audio Proc │ │ Checkpoint │  │
│  │ 视频读写    │ │ 图像处理    │ │ 音频处理    │ │ 断点管理    │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心引擎接口定义

```python
class V2VEngine:
    """视频生视频引擎 — 平台无关的核心接口"""

    def analyze(self, video_path: str) -> VideoAnalysis:
        """分析视频：运动、姿态、内容、节奏"""

    def plan(self, analysis: VideoAnalysis, effects: list[EffectModule]) -> SegmentPlan:
        """规划分段：根据效果模块组合动态分段"""

    def generate(self, plan: SegmentPlan, adapter: GenerationAdapter) -> list[SegmentResult]:
        """链式生成：逐段调度，上下文传递"""

    def stitch(self, results: list[SegmentResult], effects: list[EffectModule]) -> str:
        """无缝拼接：按效果模块组合决定拼接策略"""

    def run(self, video_path: str, effects: list[EffectModule], adapter: GenerationAdapter) -> str:
        """一键运行：分析 → 规划 → 生成 → 拼接"""


class EffectModule(Protocol):
    """效果模块基类 — 所有模块可叠加组合"""

    @property
    def stage(self) -> int:
        """执行阶段: 1=模仿 2=创意 3=运镜 5=增强"""

    def transform_poses(self, poses: PoseSequence, context: EffectContext) -> PoseSequence:
        """变换姿态序列"""

    def transform_prompts(self, prompts: PromptList, context: EffectContext) -> PromptList:
        """变换提示词"""

    def transform_params(self, params: SegmentParams, context: EffectContext) -> SegmentParams:
        """变换生成参数"""

    def transform_stitch(self, stitch_plan: StitchPlan, context: EffectContext) -> StitchPlan:
        """变换拼接方案"""


class EffectContext:
    """模块间传递的上下文 — 前序模块的输出作为后序模块的输入"""
    video_analysis: VideoAnalysis
    pose_sequence: PoseSequence
    prompts: PromptList
    params: SegmentParams
    stitch_plan: StitchPlan
    metadata: dict


class GenerationAdapter(Protocol):
    """生成适配器 — 对接不同的生成后端"""

    def submit(self, task: GenerationTask) -> str:
        """提交生成任务，返回任务ID"""

    def wait(self, task_id: str) -> GenerationResult:
        """等待任务完成"""

    def extract_frame(self, result: GenerationResult, frame_idx: int) -> str:
        """从生成结果中提取指定帧"""

    def discover_nodes(self, workflow: dict) -> dict:
        """自动发现工作流中的关键节点"""
```

### 2.3 生成适配器

```python
class ComfyUIAdapter(GenerationAdapter):
    """ComfyUI 适配器 — 通过 Queue API 调度"""

    def __init__(self, host="127.0.0.1:8188"):
        self.host = host

    def submit(self, task):
        workflow = self._build_workflow(task)
        return queue_prompt(workflow, self.host)

    def wait(self, task_id):
        return wait_for_completion(task_id, self.host)


class LTXStudioAdapter(GenerationAdapter):
    """LTX Studio 适配器 — 对接云集智能视频创意站"""

    def __init__(self, base_url="http://127.0.0.1:7860"):
        self.base_url = base_url

    def submit(self, task):
        return requests.post(f"{self.base_url}/api/generation/video", json=task.to_dict())

    def wait(self, task_id):
        return poll_generation_status(task_id, self.base_url)


class DirectPipelineAdapter(GenerationAdapter):
    """直连 Pipeline 适配器 — 直接调用模型推理"""

    def submit(self, task):
        return self.pipeline.generate(**task.to_dict())


class CloudAPIAdapter(GenerationAdapter):
    """云端 API 适配器 — 调用远程生成服务"""

    def submit(self, task):
        return requests.post(f"{self.cloud_url}/v2v/generate", json=task.to_dict())
```

---

## 三、效果模块详解

### 3.1 🎭 模仿模块 (Mimic) — Stage 1: 决定动作来源

| 子模式 | 输入 | 输出 | 核心逻辑 |
|--------|------|------|----------|
| 完全模仿 | 参考视频 | 同动作同场景同风格新视频 | 1:1 提取姿态+运镜，忠实还原 |
| 动作迁移 | 参考视频 + 人物图 + 场景描述 | 新人物在新场景做同样动作 | 姿态提取 + 新人物IP-Adapter + 新场景提示词 |
| 人物替换 | 参考视频 + 人物图 | 同动作同场景换人 | 姿态驱动 + IP-Adapter 人物一致 |
| 风格迁移 | 参考视频 + 风格描述 | 同动作同场景新风格 | 姿态驱动 + 风格 LoRA + 提示词引导 |

**动作迁移 vs 人物替换 的区别**:

```
人物替换 (Person Swap):
  原视频:  张三在篮球场打篮球
  输出:    李四在篮球场打篮球        ← 场景不变，只换人

动作迁移 (Motion Transfer):
  原视频:  张三在篮球场打篮球
  输出:    李四在舞台上跳舞(同样动作)  ← 场景也变了，动作迁移到全新语境

  动作迁移的核心价值:
  · 提取"动作语义"而非"像素级动作"
  · 同一个挥臂动作，在篮球场是投篮，在舞台可以是舞蹈
  · 用户可以自由定义新场景，动作自动适配
  · 这才是 V2V 真正的革命性：视频不再是模仿的对象，而是创意的起点
```

**生成策略**: 一镜到底链式生成（保持动作连贯性）

### 3.2 🎬 运镜模块 (Cinematic) — Stage 3: 决定镜头语言

| 子模式 | 输入 | 输出 | 核心逻辑 |
|--------|------|------|----------|
| 炫酷运镜 | 参考视频 + 运镜描述 | 电影级运镜视频 | 运镜参数注入 + 姿态保持 |
| 转场特效 | 参考视频 | 带创意转场的视频 | 镜头边界检测 + 转场模板匹配 |
| 一镜到底 | 参考视频 | 零接缝长视频 | 末帧链式传递 + 无缝拼接 |
| 多角度 | 参考视频 + 视角设定 | 多视角视频 | 3D 姿态重建 + 视角投影 |

**生成策略**: 智能分段 + 转场编排

**运镜参数注入**:
```python
CAMERA_MOTION_PRESETS = {
    "push_in":    {"zoom": 1.3, "speed": "slow", "prompt_suffix": "camera slowly pushing in"},
    "pull_out":   {"zoom": 0.7, "speed": "slow", "prompt_suffix": "camera slowly pulling out"},
    "pan_left":   {"pan": -1.0, "speed": "medium", "prompt_suffix": "camera panning left"},
    "pan_right":  {"pan": 1.0, "speed": "medium", "prompt_suffix": "camera panning right"},
    "orbit":      {"orbit": 360, "speed": "slow", "prompt_suffix": "camera orbiting around subject"},
    "crane_up":   {"tilt": -1.0, "speed": "slow", "prompt_suffix": "camera crane shot moving up"},
    "dolly_zoom": {"zoom": 1.5, "dolly": -1.0, "prompt_suffix": "dolly zoom effect"},
    "shake":      {"shake": 0.3, "speed": "fast", "prompt_suffix": "handheld camera shake"},
    "follow":     {"track": 1.0, "speed": "match", "prompt_suffix": "camera following subject"},
}
```

**转场模板库**:
```python
TRANSITION_PRESETS = {
    "fade_black":      {"type": "dissolve", "color": "black", "duration": 8},
    "fade_white":      {"type": "dissolve", "color": "white", "duration": 8},
    "cross_dissolve":  {"type": "blend", "duration": 12},
    "zoom_in":         {"type": "scale", "direction": "in", "duration": 6},
    "zoom_out":        {"type": "scale", "direction": "out", "duration": 6},
    "spin":            {"type": "rotate", "duration": 8},
    "glitch":          {"type": "effect", "effect": "glitch", "duration": 4},
    "blur":            {"type": "effect", "effect": "gaussian_blur", "duration": 6},
    "whip_pan":        {"type": "motion_blur", "direction": "horizontal", "duration": 3},
    "match_cut":       {"type": "smart", "match": "shape", "duration": 2},
}
```

### 3.3 ✨ 创意模块 (Creative) — Stage 2: 决定动作表达

| 子模式 | 输入 | 输出 | 核心逻辑 |
|--------|------|------|----------|
| 动态增强 | 参考视频 | 动作更夸张的视频 | 姿态幅度放大 + 物理约束 |
| 节奏重构 | 参考视频 + 音乐 | 节拍同步视频 | 节拍检测 + 时间重映射 |
| 混合融合 | 多个参考视频 | 融合视频 | 多源姿态/场景/风格交叉组合 |
| 超现实 | 参考视频 + 创意描述 | 超现实视频 | 结构保持 + 超现实提示词 + 风格 LoRA |

**节奏重构核心**:
```python
def remap_to_beats(video_analysis, audio_beats):
    """
    将视频动作重映射到音乐节拍
    """
    original_timestamps = video_analysis.motion_peaks  # 原始动作峰值时间
    beat_timestamps = audio_beats                       # 音乐节拍时间

    # 计算时间映射函数
    time_map = compute_time_warp(original_timestamps, beat_timestamps)

    # 重新采样姿态序列
    remapped_poses = resample_poses(video_analysis.poses, time_map)

    # 重新生成提示词（节奏关键词）
    rhythm_prompts = generate_rhythm_prompts(beat_timestamps, video_analysis.motion_types)

    return remapped_poses, rhythm_prompts
```

### 3.4 🔧 增强模块 (Enhance) — Stage 5: 决定输出品质

| 子模式 | 输入 | 输出 | 核心逻辑 |
|--------|------|------|----------|
| 画质提升 | 低分辨率视频 | 高分辨率视频 | 超分 + 姿态一致性约束 |
| 帧率提升 | 低帧率视频 | 高帧率视频 | 补间插帧 + 运动补偿 |
| 去除抖动 | 抖动视频 | 稳定视频 | 运动估计 + 反向补偿 |
| 背景替换 | 视频 + 新背景 | 换背景视频 | 前景分割 + 背景合成 |

---

## 四、智能调度引擎：YunjiiSegmentEngine

### 4.1 三种分段模式

```
━━━ 模式 1：一镜到底 (One Shot) ━━━

  适用：单镜头连续运动（舞蹈/动作/运镜）
  特点：前段末帧→后段参考图，零接缝

  Seg1: [0────80]  ── 末帧传递 ──→  Seg2: [81────160]  ── 末帧传递 ──→  Seg3: [161────240]
  ↑ 用户参考图                         ↑ Seg1末帧                          ↑ Seg2末帧

━━━ 模式 2：智能分段 (Smart Split) ━━━

  适用：多镜头视频
  特点：按镜头边界分段，段间转场编排

  Seg1: [0────80]  ── 转场特效 ──  Seg2: [73────160]  ── 转场特效 ──  Seg3: [153────240]
  ↑ 用户参考图                      ↑ 自动选取参考图                     ↑ 自动选取参考图

━━━ 模式 3：滑动窗口 (Sliding Window) ━━━

  适用：超长视频（>60秒）
  特点：固定窗口滑动，重叠区域加权融合

  Seg1: [0────────80]  ── 重叠融合 ──  Seg2: [64────────144]  ── 重叠融合 ──  Seg3: [128────────208]
```

### 4.2 动态分段算法

```python
def dynamic_segment_plan(analysis: VideoAnalysis, effects: list[EffectModule]) -> SegmentPlan:
    """
    根据视频分析结果 + 效果模块组合，动态规划最优分段方案
    """
    segments = []

    for scene in analysis.scenes:
        complexity = scene.motion_complexity  # 0.0 ~ 1.0

        # 基础分段参数（根据复杂度自适应）
        if complexity > 0.7:
            seg_frames, overlap, steps, cfg = 41, 8, 30, 6.5
        elif complexity > 0.3:
            seg_frames, overlap, steps, cfg = 61, 6, 25, 5.5
        else:
            seg_frames, overlap, steps, cfg = 81, 4, 20, 5.0

        # 效果模块覆盖参数
        for effect in effects:
            if isinstance(effect, OneShotCinematic):
                overlap = max(overlap, 8)   # 一镜到底需要更大重叠
            elif isinstance(effect, TransitionCinematic):
                overlap = max(overlap, 12)  # 转场需要更多过渡帧
            elif isinstance(effect, RhythmCreative):
                steps = max(steps, 30)      # 节奏重构需要更多步数

        sub_segments = split_with_overlap(scene, seg_frames, overlap)
        segments.extend(sub_segments)

    return SegmentPlan(segments=segments, effects=effects)
```

### 4.3 链式执行引擎

```python
def execute_segment_chain(plan, adapter, user_ref_image, effects):
    """
    链式执行：每段依赖前一段的输出
    效果模块按 Stage 顺序叠加作用
    """
    results = []
    prev_context = None

    for seg in plan.segments:
        # 1. 确定参考图策略（由模仿模块决定）
        if seg.index == 0:
            ref_image = user_ref_image
        elif any(isinstance(e, OneShotCinematic) for e in effects):
            ref_image = prev_context.last_frame
        else:
            ref_image = auto_select_ref_frame(seg, prev_context)

        # 2. 构建生成任务（所有模块的输出汇聚）
        task = GenerationTask(
            ref_image=ref_image,
            poses=seg.pose_slice,
            prompt=seg.prompt,
            params=seg.params,
            prev_context=prev_context,
        )

        # 3. 提交并等待
        task_id = adapter.submit(task)
        result = adapter.wait(task_id)

        if not result.success:
            result = retry_with_fallback(adapter, task, max_retries=3)

        # 4. 提取上下文（供下一段使用）
        prev_context = SegmentContext(
            last_frame=adapter.extract_frame(result, -1),
            style_embedding=result.style_embedding,
            color_palette=result.color_palette,
        )

        # 5. 保存断点
        save_checkpoint(seg.index, prev_context)

        results.append(result)

    return results
```

### 4.4 无缝拼接器

| 拼接模式 | 适用场景 | 原理 |
|----------|----------|------|
| 硬切 | 场景切换 | 直接拼接 |
| 交叉淡化 | 场景渐变 | alpha 加权混合 |
| 潜在空间融合 | 一镜到底 | VAE latent 空间加权融合 |
| 转场特效 | 运镜模式 | 插入预设转场动画 |
| 智能选择 | 自动 | 根据段间差异自动匹配 |

---

## 五、适配器层设计

### 5.1 ComfyUI 适配器（当前阶段）

```
comfyui-video-frame-yunjii/
├── __init__.py              # 注册节点 + API 路由
├── nodes.py                 # 分析节点（现有）
├── engine/                  # V2V 核心引擎
│   ├── __init__.py
│   ├── core.py              # V2VEngine 主类
│   ├── planner.py           # 动态分段规划器
│   ├── runner.py            # 链式执行引擎
│   ├── stitcher.py          # 无缝拼接器
│   ├── effects/             # 效果编排
│   │   ├── mimic.py         # 模仿模式
│   │   ├── cinematic.py     # 运镜模式
│   │   ├── creative.py      # 创意模式
│   │   └── enhance.py       # 增强模式
│   ├── adapters/            # 生成适配器
│   │   ├── comfyui.py       # ComfyUI Queue API
│   │   └── base.py          # 适配器基类
│   ├── discovery.py         # 自动节点发现
│   ├── checkpoint.py        # 断点续跑
│   └── queue_api.py         # Queue API 封装
├── utils/
│   ├── __init__.py
│   ├── scene_detect.py
│   ├── motion_analyze.py
│   ├── pose_extract.py
│   ├── prompt_utils.py
│   ├── ref_image.py
│   ├── video_io.py
│   └── audio_utils.py
├── js/
│   ├── widgets.js
│   └── engine_dashboard.js
└── workflows/
```

### 5.2 云集智能视频创意站 模块集成

```
云集智能视频创意站/
├── dev/app/resources/backend/
│   ├── services/
│   │   ├── v2v_engine/              # [NEW] V2V 引擎模块
│   │   │   ├── __init__.py
│   │   │   ├── v2v_engine.py        # V2VEngine 实例
│   │   │   ├── ltx_adapter.py       # LTX Studio 适配器
│   │   │   └── effect_presets.py    # 效果预设
│   │   └── ... (现有服务)
│   ├── _routes/
│   │   ├── v2v.py                   # [NEW] V2V API 路由
│   │   └── ... (现有路由)
│   └── handlers/
│       ├── v2v_handler.py           # [NEW] V2V 请求处理
│       └── ... (现有处理器)
└── dev/app/resources/ui/
    ├── index.js                     # [UPDATE] 增加 V2V 面板
    └── index.html
```

**创意站集成 API**:
```python
# POST /api/v2v/analyze
# 分析视频，返回运动/姿态/内容/节奏信息

# POST /api/v2v/plan
# 根据效果模式生成分段计划

# POST /api/v2v/generate
# 执行链式生成

# POST /api/v2v/stitch
# 无缝拼接

# POST /api/v2v/run
# 一键运行（分析→规划→生成→拼接）

# GET /api/v2v/status/{run_id}
# 查询运行状态

# POST /api/v2v/cancel/{run_id}
# 取消运行

# POST /api/v2v/resume/{run_id}
# 断点续跑
```

### 5.3 独立桌面应用

```
YunjiiV2V/
├── main.py                 # PyQt6 主窗口
├── engine/                 # V2V 核心引擎（同一套代码）
├── adapters/
│   ├── local_pipeline.py   # 本地 Pipeline 适配器
│   └── cloud_api.py        # 云端 API 适配器
├── ui/
│   ├── main_window.py      # 主窗口
│   ├── video_panel.py      # 视频预览面板
│   ├── effect_panel.py     # 效果模式选择面板
│   ├── timeline.py         # 时间轴编辑器
│   └── progress.py         # 进度面板
└── build/
    └── build.py            # PyInstaller 打包
```

### 5.4 线上平台

```
YunjiiV2V-Cloud/
├── api/
│   ├── main.py             # FastAPI 主入口
│   ├── routes/
│   │   ├── v2v.py          # V2V API
│   │   ├── auth.py         # 认证
│   │   └── billing.py      # 计费
│   └── engine/             # V2V 核心引擎（同一套代码）
├── worker/
│   ├── gpu_worker.py       # GPU 生成 Worker
│   └── task_queue.py       # 任务队列
├── web/
│   ├── index.html          # Web 前端
│   └── app.js
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 六、当前状态：已实现功能

### 6.1 已有节点（5 个）

| 节点 | 目录 | 状态 | 功能 |
|------|------|------|------|
| MotionAnalysisNode | `Yunjii/Video` | ✅ | 镜头检测、运镜分析、人物检测、关键帧提取 |
| PromptControlNode | `Yunjii/Video` | ✅ | 提示词前缀/后缀/合并/替换控制 |
| KeyframePreviewNode | `Yunjii/Video` | ✅ | 关键帧预览与帧信息展示 |
| VideoPoseExtractor | `Yunjii/Video/Pose` | ✅ | MediaPipe 姿态提取，OpenPose 格式输出 |
| MimicPromptGenerator | `Yunjii/Video/Mimic` | ✅ | 基于运镜分析的提示词生成 |

### 6.2 已有前端功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 视频上传路由 | ✅ | 一键上传视频到 input 目录 |
| 手动选帧浏览器 | ✅ | 全屏帧浏览器，左右键标注/取消关键帧 |
| 帧信息卡片 | ✅ | 节点上实时显示镜头信息和手动选帧 |
| 场景分段可视化 | ✅ | 时间轴上的镜头范围标记 |

### 6.3 已有分析能力

| 能力 | 算法 | 说明 |
|------|------|------|
| 镜头检测 | HSV 直方图相关性 | `cv2.compareHist(HISTCMP_CORREL)` |
| 运镜分析 | Farneback 光流 | 识别平移/俯仰/静态/场景切换 |
| 人物检测 | Canny 边缘密度 | 轻量级近似判断 |
| 姿态提取 | MediaPipe → OpenPose | 时序平滑 + 帧间插值 |
| 提示词生成 | 运镜关键词组合 | 带人物描述和风格关键词 |

---

## 七、开发路线图

### Phase 1: V2V 核心引擎 + ComfyUI 适配 🔴

```
目标：在 ComfyUI 中实现完整的 V2V 工作流

Step 1   engine/core.py              V2VEngine 主类 + 接口定义
Step 2   engine/adapters/base.py     GenerationAdapter 基类
Step 3   engine/adapters/comfyui.py  ComfyUI Queue API 适配器
Step 4   engine/planner.py           动态分段规划器
Step 5   engine/runner.py            链式执行引擎
Step 6   engine/stitcher.py          无缝拼接器
Step 7   engine/effects/mimic.py     模仿模式编排
Step 8   engine/discovery.py         自动节点发现
Step 9   engine/checkpoint.py        断点续跑
Step 10  __init__.py                 注册新节点
Step 11  端到端测试：模仿模式一键运行
```

### Phase 2: 效果模块扩展 🟡

```
目标：实现运镜模块和创意模块，建立模块叠加管线

Step 1   engine/effects/base.py          EffectModule 基类 + EffectContext
Step 2   engine/effects/mimic.py         模仿模块（重构为模块化）
Step 3   engine/effects/cinematic.py     运镜模块
Step 4   engine/effects/creative.py      创意模块
Step 5   engine/effects/enhance.py       增强模块
Step 6   engine/pipeline.py             模块叠加管线（Stage 1→2→3→4→5）
Step 7   运镜预设库 + 转场模板库
Step 8   节奏分析模块（音频节拍检测）
Step 9   前端效果模块选择器（支持多选叠加）
```

### Phase 3: 前端控制面板 🟢

```
目标：完善的用户交互界面

Step 1   js/engine_dashboard.js       引擎控制面板
Step 2   效果模式选择器
Step 3   实时进度面板
Step 4   段落缩略图预览
Step 5   失败段重试 + 断点续跑 UI
Step 6   运镜/转场可视化编辑器
```

### Phase 4: 创意站模块集成 🟣

```
目标：将 V2V Engine 集成到云集智能视频创意站

Step 1   LTX Studio 适配器
Step 2   V2V API 路由 + Handler
Step 3   创意站前端 V2V 面板
Step 4   Pipeline 共享（避免重复加载模型）
Step 5   与现有生成流程的融合
```

### Phase 5: 独立产品化 🔵

```
目标：V2V Engine 成为独立产品

Step 1   独立桌面应用（PyQt6）
Step 2   本地 Pipeline 适配器（不依赖 ComfyUI）
Step 3   CLI 工具
Step 4   云端 API 适配器
Step 5   Web 前端
Step 6   Docker 容器化部署
Step 7   多 GPU / 分布式生成
```

---

## 八、与现有 SQR 的关系

| 策略 | 说明 |
|------|------|
| **独立运行** | V2V Engine 完全独立，不依赖 comfyui_segment_queue |
| **可共存** | 两个插件可以同时安装 |
| **不修改** | 不修改 SQR 的任何代码 |
| **超越** | V2V Engine 的能力远超 SQR（效果模式/动态分段/链式生成/无缝拼接） |
| **API 隔离** | 使用 `/yunjii/` 前缀 |

---

## 九、风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 核心引擎跨平台兼容性 | 不同环境行为不一致 | 核心引擎纯 Python，无平台依赖 |
| 生成后端差异大 | 适配器开发成本高 | 统一 GenerationAdapter 接口，隔离差异 |
| 一镜到底末帧质量差 | 下一段参考图不佳 | 末帧质量检查 + 回退 + 用户参考图校正 |
| 链式误差累积 | 越往后段偏差越大 | 每 N 段插入校正帧 |
| 效果模式参数复杂 | 用户难以选择 | 预设模板 + 智能推荐 |
| 创意站集成冲突 | Pipeline 加载冲突 | 模型共享机制 + GPU 内存管理 |
| 长视频生成耗时 | 用户等待时间长 | 实时进度 + 断点续跑 + 预估时间 |

---

## 十、核心引擎包结构（pip install yunjii-v2v）

```
yunjii_v2v/                    # PyPI 包名: yunjii-v2v
├── __init__.py                # from yunjii_v2v import V2VEngine
├── core.py                    # V2VEngine 主类
├── types.py                   # 所有数据类型定义
├── analysis/                  # 视频分析模块
│   ├── __init__.py
│   ├── motion.py              # 运动分析
│   ├── pose.py                # 姿态提取
│   ├── content.py             # 内容理解
│   └── rhythm.py              # 节奏分析
├── engine/                    # 智能调度引擎
│   ├── __init__.py
│   ├── planner.py             # 动态分段规划器
│   ├── runner.py              # 链式执行引擎
│   ├── stitcher.py            # 无缝拼接器
│   └── checkpoint.py          # 断点管理
├── effects/                   # 效果编排
│   ├── __init__.py
│   ├── base.py                # EffectMode 基类
│   ├── mimic.py               # 模仿模式
│   ├── cinematic.py           # 运镜模式
│   ├── creative.py            # 创意模式
│   └── enhance.py             # 增强模式
├── adapters/                  # 生成适配器
│   ├── __init__.py
│   ├── base.py                # GenerationAdapter 基类
│   ├── comfyui.py             # ComfyUI 适配器
│   ├── ltx_studio.py          # LTX Studio 适配器
│   ├── direct_pipeline.py     # 直连 Pipeline 适配器
│   └── cloud_api.py           # 云端 API 适配器
└── infra/                     # 基础设施
    ├── __init__.py
    ├── video_io.py            # 视频读写
    ├── image_proc.py          # 图像处理
    ├── audio_proc.py          # 音频处理
    └── utils.py               # 通用工具
```

---

*文档版本: v3.1 · 2026.05.26 · 云集智能 · V2V Engine 产品设计 — 模块化架构*
