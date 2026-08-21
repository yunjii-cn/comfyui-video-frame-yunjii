#!/usr/bin/env bash
# ============================================================================
# 下载 SCAIL-2 Remix 画质 LoRA（肥猴分段队列工作流必需，根治画质粗糙）
# ----------------------------------------------------------------------------
# 背景：
#   肥猴(SQR)方案的画质完全依赖 Scail-2_Remix_LoRA_non_Relighting_rank256
#   这个画质 LoRA。本机 models/loras/wan/ 此前缺失该文件，导致 _fix_model_names
#   回退 none -> 输出画质粗糙。本脚本把它下载到位即可根治。
#
# 用法（在 Git Bash 里运行）：
#   ./download_remix_lora.sh            # 直连 hf-mirror（推荐，绕过 7890 代理限速）
#   ./download_remix_lora.sh --proxy   # 走 http://127.0.0.1:7890 代理（若你本机代理更快）
#
# 注意：
#   - 文件 3.68GB，务必在本机（有正常外网）执行，不要在沙箱跑（沙箱实测 2KB/s）。
#   - 支持断点续传：中断后重跑同一条命令即可从断点继续。
#   - 下载完成后无需改代码，重启 ComfyUI 即可生效。
# ============================================================================
set -euo pipefail

PROXY_FLAG="--noproxy *"
if [[ "${1:-}" == "--proxy" ]]; then
  PROXY_FLAG="-x http://127.0.0.1:7890"
  echo "[info] 使用代理 127.0.0.1:7890"
else
  echo "[info] 直连 hf-mirror（绕过代理限速）"
fi

URL="https://hf-mirror.com/INFOMSG/Scail-2_Remix_LoRA/resolve/main/Scail-2_Remix_LoRA_non_Relighting_rank256.safetensors"
OUT_DIR="F:/ComfyUI_heihe/ComfyUI/models/loras/wan"
OUT_FILE="$OUT_DIR/Scail-2_Remix_LoRA_non_Relighting_rank256.safetensors"
EXPECTED_SIZE=3680638752

mkdir -p "$OUT_DIR"

echo "[info] 目标: $OUT_FILE"
echo "[info] 预期大小: $EXPECTED_SIZE 字节 (3.68GB)"

# 断点续传下载
curl $PROXY_FLAG -C - -L --retry 5 --retry-delay 3 -o "$OUT_FILE" "$URL"

# 校验
ACTUAL=$(stat -c%s "$OUT_FILE" 2>/dev/null || echo 0)
if [[ "$ACTUAL" -eq "$EXPECTED_SIZE" ]]; then
  echo "[OK] 下载完整，大小校验通过 ($ACTUAL 字节)"
  echo "[done] 重启 ComfyUI 后，肥猴(SQR)方案将自动挂载 Remix 画质 LoRA"
else
  echo "[WARN] 大小不符: 实际 $ACTUAL / 预期 $EXPECTED_SIZE"
  echo "[WARN] 可能未下完或被重定向。请重跑本脚本续传，或手动核对。"
  exit 1
fi
