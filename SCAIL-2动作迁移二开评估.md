# SCAIL-2 动作迁移二开评估（2026-07-26）

> 背景：用户问"是否要把动作迁移技术用 SCAIL-2 来二开（称其为目前最好的开源动作迁移项目）"。
> 结论先行：**SCAIL-2 确实存在、是当下 SOTA 级开源动作迁移框架，值得引入；但应"集成"而非"从零二开"模型本身。**

---

## 一、SCAIL-2 是什么（事实核查）

- **出品方**：智谱 AI（Z.ai / Zhipu）+ 清华大学联合开源，论文《SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning》（arXiv 2606.10804）。
- **协议 / 可得性**：**Apache 2.0**，权重在 Hugging Face（zai-org/SCAIL-2）与 ModelScope；代码 github.com/zai-org/SCAIL-2；**已接进 ComfyUI**（Comfy-Org 构建 + Kijai / RunningHub 社区工作流）。
- **规模 / 底座**：14B 参数，构建于 **Wan 视频生态**（复用 Wan 的 VAE 与 T5 文本编码器），与本插件当前的 WanVideo 后端**同生态**，亲和度高。
- **核心范式（与你们相反）**：**彻底抛弃骨架/姿态中间表示**，端到端把"驱动视频"的动作直接迁移到"参考图"上。用两类掩码通道（环境切换掩码 + 角色绑定掩码，由 SAM3 自动提取）+ 模式专属 RoPE 统一处理：
  - Human2Any（人驱动任意角色）
  - Any2Any（任意角色互驱）
  - 角色替换（Replacement）
  - 多角色交互、跨物种（动物/卡通/四足）、零样本驱动
- **质量**：Studio-Bench 上单角色动作一致性胜率 68.3%（vs 前代 SCAIL）、65.0%（vs Wan-Animate）；多角色身份隔离 90.0%；部分指标接近/超过商业系统 Kling 3.0。
- **局限**：仅 **512p / 704p**（尺寸需被 32 整除），定位 previs / 预演，**非最终 4K 渲染**；强依赖高质量配对数据。

---

## 二、与本插件现状的对比

| 维度 | 本插件现状 | SCAIL-2 |
|------|-----------|---------|
| 动作表示 | **骨架/姿态驱动**：`VideoPoseExtractor`（MediaPipe）→ 姿态图 → WanVideoSampler 链式分段 | **无骨架**：参考图 + 驱动视频 + SAM3 掩码，端到端 |
| 管线 | 分析 → 规划 → 分段链式生成 → 拼接（你们的 IP 在"分段+拼接"） | 单模型一次生成（支持 infinite-length 社区工作流） |
| 跨物种/多角色 | 不支持（骨架为人形，四足/尾巴会糊） | 原生支持 |
| 手部/精细动作 | 姿态估计易丢手部细节 | 偏差感知 DPO 专门修手部 |
| 分辨率 | 取决于 WanVideo 后端（可高清） | 上限 704p |
| ComfyUI 集成 | 自研 `DirectAdapter` 内联执行 | 官方/社区 ComfyUI 节点已存在 |

**关键结论**：两者是**不同范式**。你们的"分段规划 + 拼接"是 V2V 长视频的独门价值；SCAIL-2 的"端到端无骨架"是动作迁移的质量与泛化价值。它们**互补**，不是替代。

---

## 三、"用 SCAIL-2 二开动作迁移"的三种路线

### ❌ 路线 A：fork SCAIL-2 模型从零二开
- 改 14B 权重 / 训练 / DPO，成本极高（训练集群、MotionPair-60K 合成流水线、Bias-Aware DPO）。
- 官方已给 Apache 2.0 权重 + ComfyUI 节点，**没必要重复造轮子**。
- **不推荐。**

### ✅ 路线 B（推荐）：把 SCAIL-2 作为本插件的"新生成后端/适配器"
- 保留你们的核心 IP：**planner（分段规划）+ runner（链式执行）+ stitcher（拼接）**。
- 新增 `engine/adapters/scail.py`（`SCAILAdapter`），实现与 `DirectAdapter` **同一套接口**（`init_executor / execute_inline / cleanup_executor / discover_nodes / modify_workflow_for_segment`）。
- runner 在 `engine/adapters/direct.py` 之外多一个后端选项；用户可切换"骨骼路线（现有）"或"SCAIL-2 端到端路线"。
- 这样 Phase 2 计划里的 **motion_transfer 效果模块**可直接架在 SCAIL-2 上，白拿跨物种/多角色/精细手部能力。
- **优点**：改动小、复用最大化、风险低、立刻拿到 SOTA 质量。

### 🟡 路线 C：用 SCAIL-2 直接替换整条管线
- 抛弃你们的分段/拼接，全交给 SCAIL-2 的 infinite-length 工作流。
- 会**丢失你们的长视频分段控制与参考图串联能力**，且 SCAIL-2 仅 704p。
- 仅当目标就是"短预演动作迁移"时考虑。

---

## 四、路线 B 集成方案草图

```
engine/
  adapters/
    direct.py      # 现有：WanVideo 骨骼路线（保留为 fallback）
    scail.py       # 新增：SCAILAdapter —— 包装 SCAIL-2 的 ComfyUI 工作流
runner.py          # 增加后端选择：DirectAdapter / SCAILAdapter
nodes.py           # 新增"动作迁移(SCAIL)"模式：输入 参考图 + 驱动视频(+掩码)
```

- `SCAILAdapter` 内部复用 `DirectAdapter` 的 `PromptExecutor` 内联执行骨架（同 diesee），只是 `modify_workflow_for_segment` 改为注入 SCAIL-2 的工作流节点 + SAM3 掩码。
- planner 的 `ref_strategy` 可映射为 SCAIL-2 的 Animation / Replacement 模式（首段=Animation，后续段=用前段末帧做 Replacement 串联）。
- stitcher 不变（拼接两路后端产出的视频段）。

---

## 五、前提与风险（落地前必须知道）

1. **显存门槛高**：14B Wan 模型，建议 **24GB+** 显存；704p 比 512p 更吃资源。
2. **分辨率上限 704p**：定位 previs，不是最终成片。若需要高清成片，仍走现有 WanVideo 骨骼路线（或 SCAIL-2 出低清 + 超分后处理）。
3. **需 SAM3 掩码**：参考图与驱动视频都要 SAM3 提取掩码 → 需额外接入 SAM3 节点/模型。
4. **权重需本机下载**：14B 权重约数十 GB，从我这侧无法代为下载，需你在本机 ComfyUI 环境拉取（`hf download zai-org/SCAIL-2` 或 ModelScope）。
5. **并存与回退**：两套后端并存，保留现有骨骼路线作为高清 fallback，避免 SCAIL-2 704p 限制成为唯一路径。

---

## 六、是否"需要"的判断框架

- **需要引入（强烈建议）**，当目标是：更好的动作迁移质量、跨物种/多角色、摆脱骨架在四足/尾巴/手部上的失真、统一的"动画/替换/互驱"框架。
- **不必强迁**，当目标只是维持现有高清骨骼路线、且不关心跨物种/多角色泛化。

> 我的建议：**走路线 B（集成而非二开）**。这既兑现了 PLAN 的 Phase 2"效果模块"方向，又用最小代价把动作迁移质量拉到开源 SOTA，同时不丢掉你们分段拼接的长视频能力。

---

## 七、建议的下一步（待你确认）

1. 我先把 `engine/adapters/scail.py` 的**适配器骨架** + 一个 **SCAIL-2 工作流 JSON** 搭出来，验证能否接进现有 `runner` 的分段/拼接链路（不下载模型，仅验证接口联通）。
2. 你在本机下载 SCAIL-2 权重与 SAM3，做真实出图验证。
3. 视效果决定是否把 `mimic` 效果模块整体迁到 SCAIL-2 路线。

是否要我开始第 1 步（搭骨架 + 工作流 JSON）？
