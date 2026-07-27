# SCAIL-2 真模型端到端验证清单

> 目的:在本机真实 ComfyUI 里把 SCAIL-2 后端跑通第一段,确认真实节点/widget 名,再把 `engine/adapters/scail.py` 与 `workflows/scail2_template.json` 的字段对齐。
> 生成日期:2026-07-26 · 依据官方 `docs.comfy.org/tutorials/video/zai/scail2` 与社区教程核对。

---

## 〇、一句话结论(先看这个)

SCAIL-2 **已是 ComfyUI 原生内置节点**(`WanSCAILToVideo` / `SCAIL2ColoredMask` / `SAM3_VideoTrack`),**无需装自定义节点**,只要:①ComfyUI 升级到 **Nightly/master** 分支;②下载 6 类模型权重;③装 VideoHelperSuite + KJNodes。
官方工作流用 **Base 子图(第1段)+ Extend 子图(2+段)** 手动逐段跑——**官方明确说"WanSCAILToVideo 无法自动排队所有段,需手动逐段运行"**。这正是我们 `runner + DirectAdapter` 的价值:**把逐段生成程序化自动串起来**,官方的最大痛点恰好是我们的强项。

---

## 一、需要下载的模型(6 类)

全部来自 Comfy-Org 官方 repackaged 仓库,按下表放到 `F:\ComfyUI_heihe\ComfyUI\models\` 对应子目录:

| 类型 | 文件(推荐) | 目标目录 | 来源 |
|------|-----------|---------|------|
| 扩散主模型 | `wan2.1_14B_SCAIL_2_fp8_scaled.safetensors`(24G+显存) 或 `wan2.1_14B_SCAIL_2_mxfp8.safetensors`(更省) | `models\diffusion_models\` | HF: `Comfy-Org/SCAIL-2` |
| 文本编码器 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models\text_encoders\` | HF: `Comfy-Org/Wan_2.1...` |
| CLIP Vision | `clip_vision_h.safetensors` | `models\clip_vision\` | 同上 split_files |
| VAE | `wan_2.1_vae.safetensors`(即 Wan2_1_VAE_bf16) | `models\vae\` | 同上 split_files |
| SAM3 掩码 | `sam3.1_multiplex_fp16.safetensors` | `models\sam\`(⚠见下) | HF: `Comfy-Org/sam3.1` |
| LoRA(修手脸,可选但强烈建议) | `Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`(低显存用 rank64,高显存 rank128) | `models\loras\` | HF: `lightx2v/...` |
| DPO LoRA(可选) | `wan2.1_SCAIL_2_DPO_lora_bf16.safetensors` | `models\loras\` | HF: `Comfy-Org/SCAIL-2` |

**⚠️ SAM3 目录有分歧**:官方文档写 `models\sam\`,部分社区教程写 `models\checkpoints\`。以你本机 `SAM3_VideoTrack` 节点下拉框实际能读到的目录为准——两个目录各放一份软链/副本最稳。

**省显存替代**:显存不足 24G 时可用 GGUF 量化版(`realrebelai/SCAIL-2_GGUF` 或 `Abiray/SCAIL-2-GGUF`,Q4_K_M≈10-11.5G),放 `models\unet\`,用 `Unet Loader (GGUF)`(需装 city96 的 `ComfyUI-GGUF`)。

---

## 一点五、精度选择(本机 RTX 3090 / 24G 显存)——结论:不用 BF16/FP16,用 fp8_scaled

**为什么不是 BF16/FP16**:
- 14B DiT 的 BF16/FP16 完整权重约 **28-32G,单卡 24G 根本放不下**——即便靠 block-swap/offload 硬塞,速度会掉到不可用。
- BF16 与 FP16 二选一的问题只对 **权重能装下的模型** 才有意义;对 SCAIL-2 14B,这两个都不是选项。

**3090 的正确配置**:

| 组件 | 推荐文件 | 精度说明 |
|------|---------|---------|
| 扩散主模型 | `wan2.1_14B_SCAIL_2_fp8_scaled.safetensors`(≈16-17G) | **fp8 只是存储格式**。3090(Ampere)没有 fp8 硬件计算单元(那是 40 系 Ada/Hopper 的),ComfyUI 会自动以 **fp16/bf16 反量化计算**——完全能跑,只是没有 40 系的 fp8 加速,质量几乎无损 |
| 主模型替代 | GGUF `Q8_0`(18.1G,最接近 fp16)或 `Q6_K`(14.2G,余量更大) | Q8_0 配 3090 是"最高保真"选项;若跑 704p 长段爆显存,降到 Q6_K/Q5_K_M |
| 文本编码器 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors`(≈6G) | 官方即 fp8,编码完成即卸载,无影响 |
| VAE | `wan_2.1_vae.safetensors`(bf16,0.3G) | ✅ 这里用 BF16 没问题——VAE 很小 |
| LoRA | lightx2v distill(bf16, rank64) | ✅ bf16,照官方用 |
| 计算 dtype | ComfyUI 自动(Ampere 默认 fp16 累加/bf16 均支持) | 无需手动改;若出现数值 NaN 可加启动参数 `--bf16-unet` 强制 bf16 计算 |

**一句话**:下 **fp8_scaled 主模型**(或 GGUF Q8_0),其余按官方组合;"BF16 还是 FP16"只在 VAE/LoRA 层面出现,均按官方默认即可。3090 预计 480p 每段(81帧)约 5-10 分钟,704p 会更久且需注意显存水位(必要时开 block swap 或降 GGUF 档位)。

---

## 二、前置依赖

1. **ComfyUI 升级到最新 Nightly**:`SCAIL2ColoredMask` 需要 master 分支,稳定版会报节点缺失。
   - 便携版:`git pull` + 更新依赖;桌面版:等自动更新或手动切 master。
2. **自定义节点**:`ComfyUI-VideoHelperSuite`(VHS_LoadVideo/VHS_VideoCombine)、`ComfyUI's KJNodes`。经 Manager 安装后重启。
3. 显存建议 **24G+**(fp8);低显存走 GGUF。

---

## 三、第一段验证(先用官方原生工作流,别用我们的 runner)

**目标:先证明模型本身能跑,并抄下真实 widget 名。**

1. ComfyUI → Workflow Templates 搜 "SCAIL-2",或从 `docs.comfy.org` 下官方 JSON,拖入画布。
2. Step1 组里选好 5 个模型(diffusion/lora/clip/text_encoder/vae)。
3. 上传:**驱动视频**(提供动作)+ **参考角色图**(定义身份)。
4. 分辨率:宽高**都必须能被 16 整除**(如 896×512 / 704×1280),否则张量不匹配报错。
5. SAM3 追踪:`sam3_video_object` 和 `sam3_image_object` 填开放词汇文本,默认 `human`;同一主体两边填一样。
6. 模式:`replace_mode = false` = Animation(参考角色做动作,黑底掩码);`true` = Replacement(把视频里被追踪的人换成参考角色,白底掩码)。
7. 只跑 **Base 子图第 1 段**(默认 81 帧),确认能出 4 秒片段且身份/动作正确。

---

## 四、⚠️ 关键:真实字段名 vs 我们模板/适配器的差异(务必核对并回填)

我 7-26 建的 `scail2_template.json` 与 `scail.py` 字段是**基于早期文档推测**,与官方最新节点有出入。跑通后按下表把两处改对:

| 我们当前用的名 | 官方真实名 | 位置 | 说明 |
|---------------|-----------|------|------|
| `driving_video` | **`pose_video`** | WanSCAILToVideo 输入 | 驱动视频端口名 |
| `ref_image` | **`reference_image`** | WanSCAILToVideo 输入 | 参考角色图端口名 |
| (无) | **`previous_frame_count`** | WanSCAILToVideo | 段间重叠帧,默认 **5** |
| `frame_count`=target_frames | `frame_count` 默认 **81** | 一致,但注意默认值 | 每段帧数 |
| SAM 用 `threshold/max_objects` | **`sam3_video_object`/`sam3_image_object`**(文本) | SAM3_VideoTrack | 改成开放词汇文本输入 |
| 驱动偏移 `skip_first_frames=start_frame` | **Pose offset = 76 ×(segment_index−1)** | — | **分块步长是 76,不是 81**;planner 切段步长应对齐 76 |
| `replace_mode` | `replace_mode` | 一致 | ✅ |
| `segment_index` | `segment_index`(1-based) | 一致 | ✅ |
| `width`/`height` | `width`/`height`(需 %16==0) | 一致 | ✅ 但要校验 16 整除 |

**两个需要改代码的点(跑通确认后再改,避免盲改):**
- **`scail.py`**:把 `SCAIL_*` 端口常量 `driving_video→pose_video`、`ref_image→reference_image`;`modify_workflow_for_segment` 的驱动偏移由 `start_frame` 改为 `76×(index)`(步长 76),并写入 `previous_frame_count=5`。
- **`planner.py`**:SCAIL 路线下**分段步长/重叠**应对齐"每段 81 帧、段间重叠 5、有效步进 76",与现在骨骼路线的 4k+1 帧规则不同——建议给 planner 增加"后端感知"的分段参数(或在 runner 里按后端换算)。

> 结论:**接口联通已验证(静态+桩),但真实 widget 名需本机跑通后回填这张表**——这是唯一的"最后一公里",且是纯字段对齐,改动量很小。

### 四·补、M0 代码骨架（2026-07-27 已落地，待真模型回填字段）

为把"最后一公里"变成可机械执行，已做两件事：

1. **`engine/adapters/scail.py` 字段集中化**：新增 `SCAIL_FIELD_MAP`（pose_video / reference_image / previous_frames / previous_frame_count / segment_index / frame_count / prompt / replace_mode / width / height）+ `SCAIL_SEG_LEN=81 / SCAIL_OVERLAP=5 / SCAIL_STEP=76` 常量。所有核心节点字段赋值都走这张表，**回填时只改 `SCAIL_FIELD_MAP` 一处，不动逻辑**。同时补上了此前缺失的 `previous_frame_count`（= seg.overlap_prev 或 5），这是段间 5 帧重叠连续串联必需的字段。
   - 驱动偏移维持 `skip_first_frames = seg.start_frame`：planner 在 SCAIL-2 路线下已算出 `sub_start = start + 76×sub_idx`，即 `seg.start_frame` 本就是 76×index 步进——数学已对齐，无需改。
2. **`M0_validate_scail_fields.py`（仓库根目录）**：在本机 ComfyUI 跑一次，自动拉取 `WanSCAILToVideo` 等节点的真实 INPUT_TYPES，与 `SCAIL_FIELD_MAP` 逐项比对，精确标出哪些字段存在、哪些改名。把"猜字段"变成"跑脚本拿真实清单"。
   - 用法：`python M0_validate_scail_fields.py --url http://127.0.0.1:8188`
   - 跑通后：把脚本输出的真实字段清单与 `SCAIL_FIELD_MAP` 对齐 → 端到端验证 runner（SCAIL-2 路线）→ 通过即把 SCAIL 路线设为默认后端。
   - **沙箱结构验证已通过**：新增 `.workbuddy/verify_m0_scail.py`，桩掉 cv2/numpy/folder_paths 后真导入 `SCAILAdapter`，构造 7 类节点模板 + 首段/后续段 `SegmentInfo`，断言核心节点字段映射、首段身份注入、后续段 `previous_frames` 串联、`previous_frame_count=5`、驱动偏移 `skip_first_frames=76×index`、深拷贝不污染原 workflow——**35/35 全 PASS**。证明字段集中化与偏移数学结构正确（字段*值*仍待真模型回填，但脚手架逻辑已验证）。

---

## 五、接入我们 runner 的第二步验证(字段对齐后)

1. 把官方跑通的模型选择(diffusion/lora/clip/vae/text_encoder)固化进 `scail2_template.json` 对应 loader 节点。
2. ComfyUI 里:分析→规划(planner)→选「生成后端 = SCAIL-2 路线」→ runner 自动逐段跑 → stitcher 拼接。
3. 重点验证:**段间身份不漂移**(每段都注入参考图)+ **动作连贯**(previous_frames 串联)+ **步长 76 无跳帧/无重影**。
4. 通过后:把 SCAIL 路线设为「模仿模块」默认后端。

---

## 六、长视频完美模仿 = 三层叠加(回顾)

1. **planner** 按镜头/运动切段(风格编排)——步长需按后端(骨骼 4k+1 / SCAIL 76)自适应;
2. **SCAIL-2** 原生 Base/Extend 单镜头时长延展(每段 81 帧、重叠 5);
3. **stitcher** 段间交叉淡化拼接。

现在 1、3 已就绪,2 的接口已联通,只差真模型跑通回填字段。

---

## 附:最小下载体积估算
- fp8 主模型 ≈ 16–17G;GGUF Q4 ≈ 10G;text_encoder ≈ 6G;clip_vision ≈ 2G;vae ≈ 0.3G;sam3 ≈ 3G;lora ≈ 0.6–1.2G。
- **合计约 28–35G**(fp8)/ 22G 左右(GGUF Q4)/ 30G 左右(GGUF Q8_0 组合)。
- ~~F 盘当前仅剩 27G,需先清理腾空间~~ → **已于 2026-07-26 清理完成,F 盘可用 182G**,空间充足,可直接下载 fp8 全套。
