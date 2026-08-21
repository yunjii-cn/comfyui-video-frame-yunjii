# 肥猴(SQR)方案工作流失败 — 
 彻底修复

## 问题
用户实测「多段队列·硬冻结接段（肥猴 SQR）」方案跑工作流，4 次重试全部失败，报错确定性崩溃：

```
RuntimeError: Given groups=1, weight of size [5120, 20, 1, 2, 2],
expected input[1, 36, 21, 90, 160] to have 20 channels, but got 36 channels instead
```

崩溃点位于 `WanAnimatePlus SamplerFrom 1260`).

## 根因
- SQR / 暖启动(Tier2) 路由在 `engine/runner.py` 默认优先选用 **Animate 模板**
  (`Tier2_WanAnimatePlus_Animate_template.json`)，其 `WanAnimatePlus AnimateEmbeds` 是 **36 通道**。
- 适配器 `_pin_distill_lora_and_model` 会把所有 ModelLoader 强制钉成本机唯一基座
  `wan2.1_14B_SCAIL_2_fp8_scaled`（patch_embedding 仅 **20 通道**）。
- 36≠20 → 卷积输入通道不匹配 → 确定性崩溃。本机磁盘**不存在** Wan2.2-Animate-14B（36 通道）基座。

## 修复（已改文件）
1. **`engine/runner.py`** — SQR / 暖启动 路由翻转优先级：改用 `AP_WORKFLOW_DEFAULT`
   （`Tier2_WanAnimatePlus_SCAIL2_template.json`，20 通道）为首选，Animate 仅作兜底。
   `WanAnimatePlus SCAIL_2 Embeds` 同为 20 通道、与 SCAIL_2 基座配对，且原生支持
   `transition_video` / `prefix_frames` 续写，冻结接段机制等价「肥猴」方案。
2. **`engine/adapters/animateplus.py`** — `_inject_transition_video` 去掉硬编码 `format:"AnimateDiff"`，
   改为与驱动视频节点一致（默认），避免 Wan 基底下加载上段视频格式错配。
3. 顶部注释与 routing 注释同步更新，记录通道约束根因。

## 关键验证
- `python -m py_compile` 两文件通过。
- SCAIL2 模板核查：ModelLoader(37)=`wan2.1_14B_SCAIL_2_fp8_scaled`、Embeds(347)=
  `WanAnimatePlus SCAIL_2 Embeds`（含 `transition_video`/`prefix_frames`），含步数蒸馏 LoRA →
  强制 4 步（即好片 00017 配置）；旧「画质崩坏」已在 4 步蒸馏路线修复。
- 已清理 `engine/__pycache__`，重启 ComfyUI 即生效。

## 待用户本机验证
- 重启用 ComfyUI，重跑「多段队列·硬冻结接段」方案，确认段间不再报通道错误、成片连续无缝。
  如有新报错请把最新 `output/yunjii_logs/*.log` 发回。
