# 原生 SCAIL-2 长视频节点驱动 · 切换计划

> 状态：**预留方案，默认未启用**（2026-08-14 评估结论）
> 当前 yunjii 引擎主路径仍是「自分段 + 无缝拼接」自有管线（A/B/C 三档连贯方案）。
> 原生节点切换需本机装包 + GPU 实跑验证，**沙箱无法验证**，故不在导入期引用、默认关闭。

## 1. 背景：为什么考虑驱动原生节点

FaboroHacks 工作流（`F:\ComfyUI_heihe\ComfyUI\user\default\workflows\FaboroHacks`）用开源包
`comfyui_scail2_multi_cond` 的 **`SCAIL2ScheduledLongVideo`**（主调度器，外部 mask 版）实现了「一镜到底动作模仿」：
- `pose_video`：驱动视频的姿态序列（动作模仿核心输入，IMAGE）
- `reference_{i}`：分段参考图（链式条件化锚点，防长程漂移），配合 `reference_{i}_mask`
- `segment_plan`：由 `SCAIL2SegmentPlanBuilder` 生成的调度字符串（各段帧数/参考/提示/边界重叠）
- `max_chunk_frames`(17~81, 对齐 4n+1) / `overlap_frames`：单块帧数 + 块间重叠，整片在同一条去噪轨迹连续

示例调度 599 帧 ≈ 37s，追加调度段可到 **1 分钟+ 且不劣化**。该包还提供更优的
**`SCAIL2ScheduledLongVideoInternalSAM`**（内置 SAM 便捷变体，自动出 mask，**切换时优先评估**）/ 两阶段工作流。

> 注：上述类名/端口均来自 2026-08-15 对仓库 `nodes.py` 的源码核对（此前文档里的
> `SCAIL2ScheduledLongVideoWithSAM` / `context_frames` 为早期推测名，已更正）。

这与本引擎 B 方案（单遍连续采样 + context 滑窗真·无缝）目标一致，但由原生节点原生实现、更省心。
故把「驱动原生节点」作为自有管线的**战略补充/备选**，而非立即替换。

## 2. 现状约束（决定本步只做脚手架）

- 本机当前**未安装** `comfyui_scail2_multi_cond`（沙箱无网、且需本机操作安装）。
- 沙箱**无 GPU / 无运行中的 ComfyUI**，无法实跑验证原生节点行为。
- 盲改现有可用管线风险高（可能破坏 A/B/C 三档正在工作的输出）。

因此本次只落地：**
- `engine/adapters/scail2_native.py`：可用性探测 `is_native_scail2_available()` + 接线配方
  `describe_native_scail2_wiring()`（源码级端口名）+ **已可产出准确 prompt 的**
  `build_native_graph(plan, ctx)`（用 `SCAIL2SegmentPlanBuilder` 生成 segment_plan，再喂
  `SCAIL2ScheduledLongVideo`；loader 由 ctx 注入）。全部 lazy import，**导入期绝不引用原生节点**；
  `SCAIL2_NATIVE_ENABLED=False` 默认关闭（调用即抛 RuntimeError 守卫）。
- 本文件：切换前的本机验证清单。

## 3. 本机切换步骤（用户执行，需 RTX 3090）

1. **装包**（绕过失效 ghproxy 镜像，用代理直连 github）：
   ```powershell
   cd F:\ComfyUI_heihe\ComfyUI\custom_nodes
   $env:GIT_CONFIG_GLOBAL = $null
   git -c http.proxy=http://127.0.0.1:7890 clone https://github.com/TTPlanetPig/comfyui_scail2_multi_cond.git
   # 重启 ComfyUI，确认节点 SCAIL2ScheduledLongVideo / SCAIL2ScheduledLongVideoInternalSAM 出现
   ```
2. **指权重**：基座 `wan2.1_14B_SCAIL_2_fp8_scaled` + 蒸馏 LoRA
   `lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16`（本机 fp8_scaled 是唯一能放下 24GB 的精度）。
3. **硬刷新 + 跑 FaboroHacks 参考工作流**，验证「1 分钟+ 不劣化动作模仿」。
4. **fp8 软回退护栏**：启动 ComfyUI 前设
   `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`（防 `CUDA error 214` 竞争）。
5. **蒸馏 LoRA 检测**：确认代码扫描所有输入值含 `distill`，覆盖多 LoRA 选择节点，否则
   steps 不匹配会结构性崩坏。

## 4. 切换落地（验证通过后）

- `build_native_graph(plan, ctx)` 已写好源码级准确接线（2026-08-15 核对），本机装包后只需核对：
  ① `SCAIL2ScheduledLongVideoInternalSAM` 真实类名（截断推断，需再确认）；② ctx 里 loader 节点
  （model/clip/vae/sampler/sigmas/clip_vision）的实际节点 id 与输出索引；③ 段数 >8 时截断策略。
- 置 `SCAIL2_NATIVE_ENABLED=True`，并在 composer/runner 增加「使用原生 SCAIL-2 长视频节点」开关；
- 保留自有 A/B/C 管线作为默认与回退，原生节点作为「超长 + 真动作模仿」增强路径；
- 更新 `无缝拼接_根治方案.md` / `一镜到底_配置与验证清单.md` 的对比表。

## 5. 与本次合并的关系

本次合并（三控合一「连贯方案」+ 选项名小白化 + 标准 IMAGE 输出节点）均在**自有管线**内，
不依赖原生节点，可独立提交、独立生效。原生节点切换是**正交的后续增强**，不阻塞上述改动。
