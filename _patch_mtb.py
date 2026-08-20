# -*- coding: utf-8 -*-
"""补丁：MTB 输入/输出侧栏支持视频显示（comfy_mtb 在插件工作目录外，用脚本修改）。

改动：
1. endpoint.py ACTIONS_getUserImageFolders:
   - output 子目录列表追加二级（"一级/二级"），按 mtime 降序限量
     （每父目录最多8个、总量40），让 output/yunjii_v2v/<run_id>/ 的成片可直达。
2. endpoint.py ACTIONS_getUserImages:
   - supported 白名单追加 mp4/webm/mov/mkv；
   - 视频条目 URL 走官方 /view（支持 Range，<video> 可流式播放；
     /mtb/view 用 PIL 打开文件，视频必炸）。
3. web/mtb_input_output_sidebar.js getImgsFromUrls:
   - 按 URL 扩展名动态选择 <video>/<img> 元素；video 加 autoplay/muted/loop。
运行: f:\\ComfyUI_heihe\\python_embeded\\python.exe _patch_mtb.py
"""
import io
import os

MTB = r"f:\ComfyUI_heihe\ComfyUI\custom_nodes\comfy_mtb"
ENDPOINT = os.path.join(MTB, "endpoint.py")
SIDEBAR_JS = os.path.join(MTB, "web", "mtb_input_output_sidebar.js")


def patch(path, replacements):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    changed = False
    for old, new in replacements:
        if new in text and old not in text:
            print(f"[SKIP] 已应用过 in {path}")
            continue
        if old not in text:
            raise SystemExit(f"[FAIL] 未找到目标片段 in {path}:\n---\n{old[:300]}\n---")
        if text.count(old) != 1:
            raise SystemExit(f"[FAIL] 片段不唯一({text.count(old)}次) in {path}:\n---\n{old[:300]}\n---")
        text = text.replace(old, new)
        changed = True
    if changed:
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"[OK] patched {path}")


# ------------------------------------------------------------- endpoint.py
EP_PATCHES = [
    # 1) getUserImageFolders: output 追加二级子目录
    (
        '''def ACTIONS_getUserImageFolders():
    input_dir = Path(folder_paths.get_input_directory())
    output_dir = Path(folder_paths.get_output_directory())

    input_subdirs = [x.name for x in input_dir.iterdir() if x.is_dir()]
    output_subdirs = [x.name for x in output_dir.iterdir() if x.is_dir()]

    return {"input": input_subdirs, "output": output_subdirs}''',
        '''def ACTIONS_getUserImageFolders():
    input_dir = Path(folder_paths.get_input_directory())
    output_dir = Path(folder_paths.get_output_directory())

    input_subdirs = [x.name for x in input_dir.iterdir() if x.is_dir()]
    output_subdirs = [x.name for x in output_dir.iterdir() if x.is_dir()]

    # yunjii patch: output 追加二级子目录（"一级/二级"）——生成引擎（如 yunjii_v2v）
    # 把成片放在 output/<engine>/<run_id>/ 二级目录，只列一级时侧栏进不去。
    # 二级按 mtime 降序限量（每父目录最多 8 个、总量 40），避免 run_id 累积撑爆下拉。
    output_l2 = []
    try:
        for d in sorted(
            (x for x in output_dir.iterdir() if x.is_dir()),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                subs = [s for s in d.iterdir() if s.is_dir()]
            except OSError:
                continue
            subs.sort(key=lambda s: s.stat().st_mtime, reverse=True)
            for s in subs[:8]:
                output_l2.append(f"{d.name}/{s.name}")
            if len(output_l2) >= 40:
                break
    except OSError:
        pass
    output_subdirs.extend(output_l2)

    return {"input": input_subdirs, "output": output_subdirs}''',
    ),
    # 2) getUserImages: 白名单加视频 + 视频走官方 /view
    (
        '''    supported = ["png", "jpg", "jpeg", "webp", "gif"]

    entries = {}
    patterns = build_glob_patterns(supported, recursive=include_subfolders)
    entries = glob_multiple(entry_dir, patterns)''',
        '''    supported = ["png", "jpg", "jpeg", "webp", "gif"]
    # yunjii patch: 输出目录同时列出视频文件（成片 mp4 等）
    video_exts = {"mp4", "webm", "mov", "mkv"}
    supported = supported + [e for e in video_exts if e not in supported]

    entries = {}
    patterns = build_glob_patterns(supported, recursive=include_subfolders)
    entries = glob_multiple(entry_dir, patterns)''',
    ),
    (
        '''    imgs = {
        img.name: (
            f"/mtb/view?filename={img.name}{f'&width={target_width}' if target_width and target_width > 0 else ''}&type={mode}&subfolder={subfolder or ''}"
            f"{img.parent.relative_to(entry_dir) if include_subfolders else ''}"
            f"&preview={f'&rand={secrets.randbelow(424242)}' if salt_urls else ''}"
        )
        for i, img in enumerate(entries)
        if offset <= i < offset + count
    }
    return imgs''',
        '''    # yunjii patch: 视频条目 URL 走官方 /view——aiohttp FileResponse 支持 Range，
    # 浏览器 <video> 才能流式播放；/mtb/view 用 PIL 解码，视频必失败。
    imgs = {}
    for i, img in enumerate(entries):
        if not (offset <= i < offset + count):
            continue
        if img.suffix.lower().lstrip(".") in video_exts:
            imgs[img.name] = (
                f"/view?filename={urllib.parse.quote_plus(img.name)}"
                f"&type={mode}&subfolder={subfolder or ''}"
            )
        else:
            imgs[img.name] = (
                f"/mtb/view?filename={img.name}{f'&width={target_width}' if target_width and target_width > 0 else ''}&type={mode}&subfolder={subfolder or ''}"
                f"{img.parent.relative_to(entry_dir) if include_subfolders else ''}"
                f"&preview={f'&rand={secrets.randbelow(424242)}' if salt_urls else ''}"
            )
    return imgs''',
    ),
]

# ------------------------------------------------------- sidebar js
JS_PATCHES = [
    (
        '''  const elem = currentMode === 'video' ? 'video' : 'img'

  for (const [key, url] of Object.entries(urls)) {
    const a = makeElement(elem)
    a.src = url
    a.width = currentWidth
    if (currentMode === 'input') {''',
        '''  // yunjii patch: 输出目录里的视频文件（mp4/webm/...）也用 <video> 渲染；
  // 其 URL 由后端 getUserImages 指到官方 /view（支持 Range，可流式播放）。
  const isVideoUrl = (url) =>
    /\\.(mp4|webm|mov|mkv)(\\?|$)/i.test(url)

  for (const [key, url] of Object.entries(urls)) {
    const elem =
      currentMode === 'video' || isVideoUrl(url) ? 'video' : 'img'
    const a = makeElement(elem)
    a.src = url
    a.width = currentWidth
    if (elem === 'video') {
      a.autoplay = true
      a.muted = true
      a.loop = true
    }
    if (currentMode === 'input') {''',
    ),
    (
        '''    } else {
      a.autoplay = true

      a.muted = true
      a.loop = true
      a.onclick = (_e) => {
        const selected = app.canvas.selected_nodes
        if (selected && Object.keys(selected).length === 0) {
          app.extensionManager.toast.add({
            severity: 'warn',
            summary: 'No node selected!',
            detail:
              "For now the only action when clicking videos in the sidebar is to set the video on all selected 'Load Video (Upload)' nodes.",
            life: 5000,
          })
          return
        }

        for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
          updateImage(node, key)
        }
      }
    }''',
        '''    } else {
      // video 属性已统一在循环头部按元素类型设置（含 output 模式的视频）
      a.onclick = (_e) => {
        const selected = app.canvas.selected_nodes
        if (selected && Object.keys(selected).length === 0) {
          app.extensionManager.toast.add({
            severity: 'warn',
            summary: 'No node selected!',
            detail:
              "For now the only action when clicking videos in the sidebar is to set the video on all selected 'Load Video (Upload)' nodes.",
            life: 5000,
          })
          return
        }

        for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
          updateImage(node, key)
        }
      }
    }''',
    ),
]

if __name__ == "__main__":
    patch(ENDPOINT, EP_PATCHES)
    patch(SIDEBAR_JS, JS_PATCHES)
    print("ALL DONE")
