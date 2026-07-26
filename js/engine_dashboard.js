import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "yunjii.v2v.engine",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "YunjiiSegmentPlanner") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && message["计划摘要"] && message["计划摘要"].length > 0) {
                    _showPlanSummary(this, message["计划摘要"]);
                }
            };
        }

        if (nodeData.name === "YunjiiSegmentRunner") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && message["执行日志"] && message["执行日志"].length > 0) {
                    _showExecutionLog(this, message["执行日志"]);
                }
            };

            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;
                const btn = this.addWidget("button", "📋 从文件加载工作流", null, () => {
                    _loadWorkflowFile(this);
                });
                btn.serialize = false;
                return r;
            };
        }

        if (nodeData.name === "YunjiiSegmentStitcher") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && message["拼接报告"] && message["拼接报告"].length > 0) {
                    _showStitchReport(this, message["拼接报告"]);
                }
            };
        }
    },
});


function _showPlanSummary(node, summaryLines) {
    if (node._planWidget) {
        const idx = node.widgets.indexOf(node._planWidget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

    if (typeof node.addDOMWidget !== "function") return;

    const el = document.createElement("div");
    el.style.cssText = "background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;min-width:260px;max-height:220px;overflow-y:auto;font-family:monospace;font-size:11px;line-height:1.6;";

    const header = document.createElement("div");
    header.style.cssText = "color:#58a6ff;font-weight:600;margin-bottom:6px;font-size:12px;border-bottom:1px solid #30363d;padding-bottom:4px;";
    header.textContent = "📋 分段规划结果";
    el.appendChild(header);

    const lines = Array.isArray(summaryLines) ? summaryLines : [String(summaryLines)];
    for (const line of lines) {
        const row = document.createElement("div");
        row.style.cssText = "color:#c9d1d9;padding:1px 4px;";
        if (line.includes("段") && line.includes("帧")) {
            row.style.color = "#7ee787";
        } else if (line.startsWith("📋") || line.startsWith("📊") || line.startsWith("📐") || line.startsWith("🎞")) {
            row.style.color = "#58a6ff";
            row.style.fontWeight = "600";
        }
        row.textContent = line;
        el.appendChild(row);
    }

    const widget = node.addDOMWidget("_plan_summary", "custom", el, {
        serialize: false, getValue: () => "", setValue: () => {},
    });
    node._planWidget = widget;
    node.setDirtyCanvas(true, true);
}


function _showExecutionLog(node, logLines) {
    if (node._logWidget) {
        const idx = node.widgets.indexOf(node._logWidget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

    if (typeof node.addDOMWidget !== "function") return;

    const el = document.createElement("div");
    el.style.cssText = "background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;min-width:280px;max-height:280px;overflow-y:auto;font-family:monospace;font-size:10px;line-height:1.5;";

    const header = document.createElement("div");
    header.style.cssText = "color:#f0883e;font-weight:600;margin-bottom:6px;font-size:12px;border-bottom:1px solid #30363d;padding-bottom:4px;";
    header.textContent = "⛓ 链式执行日志";
    el.appendChild(header);

    const lines = Array.isArray(logLines) ? logLines : [String(logLines)];
    for (const line of lines) {
        const row = document.createElement("div");
        row.style.cssText = "color:#c9d1d9;padding:1px 4px;";
        if (line.includes("✅") || line.includes("完成")) {
            row.style.color = "#7ee787";
        } else if (line.includes("❌") || line.includes("失败") || line.includes("🛑")) {
            row.style.color = "#f85149";
        } else if (line.includes("🚀") || line.includes("▶")) {
            row.style.color = "#58a6ff";
            row.style.fontWeight = "600";
        } else if (line.includes("📤") || line.includes("🔗") || line.includes("👤")) {
            row.style.color = "#d2a8ff";
        } else if (line.includes("⚠️")) {
            row.style.color = "#f0883e";
        }
        row.textContent = line;
        el.appendChild(row);
    }

    const widget = node.addDOMWidget("_exec_log", "custom", el, {
        serialize: false, getValue: () => "", setValue: () => {},
    });
    node._logWidget = widget;
    node.setDirtyCanvas(true, true);
}


function _showStitchReport(node, reportLines) {
    if (node._stitchWidget) {
        const idx = node.widgets.indexOf(node._stitchWidget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

    if (typeof node.addDOMWidget !== "function") return;

    const el = document.createElement("div");
    el.style.cssText = "background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;min-width:260px;max-height:200px;overflow-y:auto;font-family:monospace;font-size:11px;line-height:1.6;";

    const header = document.createElement("div");
    header.style.cssText = "color:#3fb950;font-weight:600;margin-bottom:6px;font-size:12px;border-bottom:1px solid #30363d;padding-bottom:4px;";
    header.textContent = "🎞 无缝拼接报告";
    el.appendChild(header);

    const lines = Array.isArray(reportLines) ? reportLines : [String(reportLines)];
    for (const line of lines) {
        const row = document.createElement("div");
        row.style.cssText = "color:#c9d1d9;padding:1px 4px;";
        if (line.includes("✅") || line.includes("输出")) {
            row.style.color = "#7ee787";
        } else if (line.includes("🎬") || line.includes("📋")) {
            row.style.color = "#58a6ff";
            row.style.fontWeight = "600";
        } else if (line.includes("📊")) {
            row.style.color = "#d2a8ff";
        } else if (line.includes("⚠")) {
            row.style.color = "#f0883e";
        } else if (line.includes("🎵")) {
            row.style.color = "#79c0ff";
        }
        row.textContent = line;
        el.appendChild(row);
    }

    const widget = node.addDOMWidget("_stitch_report", "custom", el, {
        serialize: false, getValue: () => "", setValue: () => {},
    });
    node._stitchWidget = widget;
    node.setDirtyCanvas(true, true);
}


function _loadWorkflowFile(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                const wfData = JSON.parse(ev.target.result);
                const apiWf = _convertToApiFormat(wfData);
                const wfWidget = node.widgets.find(w => w.name === "工作流模板");
                if (wfWidget) {
                    wfWidget.value = JSON.stringify(apiWf);
                    node.setDirtyCanvas(true, true);
                }
            } catch (err) {
                alert("工作流文件解析失败: " + err.message);
            }
        };
        reader.readAsText(file);
    };
    input.click();
}


function _convertToApiFormat(uiWf) {
    const apiWf = {};
    const nodes = uiWf.nodes || [];
    const links = uiWf.links || [];

    const linkMap = {};
    for (const link of links) {
        const [, srcId, srcSlot, dstId, dstSlot, type] = link;
        linkMap[`${dstId}-${dstSlot}`] = [String(srcId), srcSlot];
    }

    for (const node of nodes) {
        if (node.mode === 4) continue;

        const classType = node.type;
        if (!classType) continue;

        const yunjiiTypes = ["MotionAnalysisNode", "VideoPoseExtractor", "MimicPromptGenerator",
            "PromptControlNode", "KeyframePreviewNode",
            "YunjiiSegmentPlanner", "YunjiiSegmentRunner", "YunjiiSegmentStitcher"];
        if (yunjiiTypes.includes(classType)) continue;

        const inputs = {};
        if (node.inputs) {
            for (const inp of node.inputs) {
                if (inp.link != null) {
                    const key = `${node.id}-${inp.slot_index != null ? inp.slot_index : inp.type === "VAE" ? 0 : 0}`;
                    const found = linkMap[key];
                    if (found) {
                        inputs[inp.name] = found;
                    } else {
                        for (const link of links) {
                            const [, srcId, srcSlot, dstId] = link;
                            if (dstId === node.id) {
                                const dstInput = node.inputs.find((_, i) => {
                                    return node.inputs[i].link === link[0];
                                });
                                if (dstInput && dstInput.name === inp.name) {
                                    inputs[inp.name] = [String(srcId), srcSlot];
                                }
                            }
                        }
                    }
                }
            }
        }

        if (node.widgets_values) {
            const widgetNames = _getWidgetNames(classType);
            let wIdx = 0;
            for (const val of node.widgets_values) {
                if (val != null && typeof val !== "object") {
                    const name = widgetNames[wIdx] || `param_${wIdx}`;
                    if (!(name in inputs)) {
                        inputs[name] = val;
                    }
                }
                wIdx++;
            }
        }

        apiWf[String(node.id)] = {
            classType: classType,
            inputs: inputs,
        };
    }

    return apiWf;
}


function _getWidgetNames(classType) {
    const map = {
        "LoadWanVideoT5TextEncoder": ["model_name", "dtype", "offload_device", "quant"],
        "LoadWanVideoClipTextEncoder": ["model_name", "dtype", "offload_device"],
        "WanVideoVAELoader": ["model_name", "dtype"],
        "WanVideoTextEncode": ["prompt", "negative_prompt", "force_offload", "prompt2", "prompt2_neg", "device"],
        "WanVideoClipVisionEncode": ["strength", "strength2", "start", "end", "force_offload"],
        "LoadImage": ["image", "upload"],
        "WanVideoAnimateEmbeds": ["width", "height", "frames", "force_offload", "context_frames", "context_overlap", "ref_strength", "pose_strength", "enable_preview"],
        "WanVideoModelLoader": ["model_name", "dtype", "quant", "offload_device", "attention"],
        "WanVideoSampler": ["steps", "cfg", "shift", "seed", "force_offload", "sampler", "scheduler", "contrast", "enable_preview", "preview_method"],
        "WanVideoDecode": ["enable_vae_tiling", "tile_x", "tile_y", "tile_stride_x", "tile_stride_y"],
        "VHS_VideoCombine": ["frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_output"],
    };
    return map[classType] || [];
}
