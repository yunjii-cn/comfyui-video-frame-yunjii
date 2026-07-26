import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "yunjii.video.preprocess",

    async setup() {
        _addDevToolbar();
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MotionAnalysisNode") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;

                const uploadBtn = this.addWidget("button", "📤 上传视频", null, () => {
                    _uploadVideo(this);
                });
                uploadBtn.serialize = false;

                const browseBtn = this.addWidget("button", "🎬 手动选帧", null, () => {
                    const vidWidget = this.widgets.find(w => w.name === "视频文件");
                    const videoName = vidWidget ? vidWidget.value : "";
                    if (!videoName || videoName === "(无视频文件)") { alert("请先选择或上传视频文件"); return; }
                    _openFrameBrowser(videoName, this);
                });
                browseBtn.serialize = false;

                const origOnExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function(message) {
                    if (origOnExecuted) origOnExecuted.apply(this, arguments);
                    if (message && message.scenes) {
                        try {
                            const scenes = JSON.parse(message.scenes);
                            this._yunjiiScenes = scenes;
                            const browser = window._yunjiiFrameBrowser;
                            if (browser && browser._setSceneData) browser._setSceneData(scenes);
                        } catch(e) {}
                    }
                };

                return r;
            };
        }

        if (nodeData.name === "VideoPoseExtractor") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;

                const hintWidget = this.addWidget("text", "ℹ️ 视频由运动分析节点传入", "", () => {});
                hintWidget.serialize = false;

                return r;
            };
        }

        if (nodeData.name === "KeyframePreviewNode") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && message.frame_info && message.frame_info.length > 0) {
                    _showFrameInfoCard(this, message.frame_info);
                }
            };
        }
    },
});


function _showFrameInfo(node, scenes) {
    if (node._yunjiiInfoWidget) {
        const idx = node.widgets.indexOf(node._yunjiiInfoWidget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

    const manualWidget = node.widgets.find(w => w.name === "\u624B\u52A8\u9009\u5E27");
    const manualFrames = manualWidget && manualWidget.value
        ? manualWidget.value.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n))
        : [];

    if (typeof node.addDOMWidget !== "function") return;

    const el = document.createElement("div");
    el.style.cssText = "display:flex;flex-direction:column;gap:2px;padding:4px;min-width:230px;max-height:180px;overflow-y:auto;";

    for (let i = 0; i < scenes.length; i++) {
        const scene = scenes[i];
        const midFrame = scene.start + Math.round((scene.end - scene.start) / 2);
        const isManual = manualFrames.includes(midFrame);
        const tag = scene.person ? "\u{1F464}" : "\u{1F305}";
        const src = isManual ? "\u{1F4CC}\u624B\u52A8+\u{1F3AC}\u81EA\u52A8" : "\u{1F3AC}\u81EA\u52A8";

        const badge = document.createElement("div");
        badge.style.cssText = "background:#2d5a3d;color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;font-family:monospace;";
        badge.textContent = `\u{1F3AC}\u955C\u5934${i+1} ${tag} \u2705\u5E27#${midFrame} ${src}`;
        el.appendChild(badge);
    }

    for (const mf of manualFrames) {
        const inScene = scenes.some(s => {
            const mid = s.start + Math.round((s.end - s.start) / 2);
            return mid === mf;
        });
        if (!inScene) {
            const badge = document.createElement("div");
            badge.style.cssText = "background:#5a2d5a;color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;font-family:monospace;";
            badge.textContent = `\u{1F4CC}\u624B\u52A8 \u2705\u5E27#${mf}`;
            el.appendChild(badge);
        }
    }

    const widget = node.addDOMWidget("_yunjii_info", "custom", el, {
        serialize: false, getValue: () => "", setValue: () => {},
    });
    node._yunjiiInfoWidget = widget;
    node.setDirtyCanvas(true, true);
}


function _showFrameInfoCard(node, frameInfo) {
    if (node._previewInfoWidget) {
        const idx = node.widgets.indexOf(node._previewInfoWidget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

    if (typeof node.addDOMWidget !== "function") return;

    const el = document.createElement("div");
    el.style.cssText = "background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:8px;min-width:220px;max-height:200px;overflow-y:auto;font-family:monospace;font-size:11px;";

    const header = document.createElement("div");
    header.style.cssText = "color:#4cf;font-weight:600;margin-bottom:6px;font-size:12px;";
    header.textContent = `\u{1F5BC} \u5E27\u4FE1\u606F (${frameInfo.length}\u5E27)`;
    el.appendChild(header);

    for (const line of frameInfo) {
        const row = document.createElement("div");
        row.style.cssText = "color:#ccc;padding:2px 4px;border-bottom:1px solid #222;";
        row.textContent = line;
        el.appendChild(row);
    }

    const widget = node.addDOMWidget("_preview_info", "custom", el, {
        serialize: false, getValue: () => "", setValue: () => {},
    });
    node._previewInfoWidget = widget;
    node.setDirtyCanvas(true, true);
}


function _openFrameBrowser(videoName, node) {
    const videoUrl = `/view?filename=${encodeURIComponent(videoName)}&type=input`;

    const manualWidget = node.widgets.find(w => w.name === "\u624B\u52A8\u9009\u5E27");
    let selectedFrames = new Set();
    if (manualWidget && manualWidget.value) {
        manualWidget.value.split(",").forEach(s => {
            const n = parseInt(s.trim());
            if (!isNaN(n)) selectedFrames.add(n);
        });
    }

    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position:"fixed",inset:"0",zIndex:"15000",
        background:"rgba(0,0,0,0.85)",display:"flex",
        flexDirection:"column",alignItems:"center",justifyContent:"center",
    });

    const container = document.createElement("div");
    Object.assign(container.style, {
        width:"90vw",maxWidth:"1100px",height:"82vh",
        background:"#1e1e1e",borderRadius:"12px",
        display:"flex",flexDirection:"column",overflow:"hidden",
        boxShadow:"0 10px 50px rgba(0,0,0,0.8)",
    });

    const header = document.createElement("div");
    Object.assign(header.style, {
        display:"flex",justifyContent:"space-between",alignItems:"center",
        padding:"10px 20px",borderBottom:"1px solid #333",color:"#fff",fontSize:"14px",fontWeight:"600",
    });
    header.innerHTML = `<span>\u{1F3AC} \u624B\u52A8\u9009\u5E27 - ${videoName}</span>`;

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u2715 \u5173\u95ED";
    Object.assign(closeBtn.style, {background:"#444",color:"#fff",border:"none",padding:"5px 14px",borderRadius:"6px",cursor:"pointer",fontSize:"12px"});
    closeBtn.onclick = () => { video.remove(); thumbVideo.remove(); overlay.remove(); };
    header.appendChild(closeBtn);

    const previewArea = document.createElement("div");
    Object.assign(previewArea.style, {
        flex:"1",background:"#000",display:"flex",justifyContent:"center",alignItems:"center",position:"relative",overflow:"hidden",
    });

    const mainCanvas = document.createElement("canvas");
    Object.assign(mainCanvas.style, {maxWidth:"100%",maxHeight:"100%",objectFit:"contain",cursor:"crosshair"});
    previewArea.appendChild(mainCanvas);

    const statusBadge = document.createElement("div");
    Object.assign(statusBadge.style, {
        position:"absolute",top:"10px",right:"10px",
        background:"rgba(39,174,96,0.9)",color:"#fff",
        padding:"5px 10px",borderRadius:"5px",fontSize:"12px",fontWeight:"600",
        display:"none",zIndex:"5",pointerEvents:"none",
    });
    previewArea.appendChild(statusBadge);

    const infoBar = document.createElement("div");
    Object.assign(infoBar.style, {
        position:"absolute",bottom:"0",left:"0",right:"0",
        background:"rgba(0,0,0,0.7)",color:"#fff",padding:"6px 14px",fontSize:"12px",fontFamily:"monospace",
        display:"flex",justifyContent:"space-between",
    });
    infoBar.innerHTML = `<span id="yunjii-frame-info">\u52A0\u8F7D\u4E2D...</span><span id="yunjii-scene-info"></span>`;
    previewArea.appendChild(infoBar);

    const hintBar = document.createElement("div");
    Object.assign(hintBar.style, {
        position:"absolute",top:"10px",left:"10px",
        background:"rgba(0,0,0,0.6)",color:"#aaa",
        padding:"4px 8px",borderRadius:"4px",fontSize:"10px",
    });
    hintBar.textContent = "\u5DE6\u952E=\u6807\u6CE8  \u53F3\u952E=\u53D6\u6D88";
    previewArea.appendChild(hintBar);

    const timelineArea = document.createElement("div");
    Object.assign(timelineArea.style, {
        height:"100px",background:"#252525",borderTop:"1px solid #333",padding:"0 16px",
        display:"flex",flexDirection:"column",justifyContent:"center",position:"relative",
    });

    const timeLabels = document.createElement("div");
    Object.assign(timeLabels.style, {
        display:"flex",justifyContent:"space-between",fontSize:"10px",fontFamily:"monospace",color:"#666",marginBottom:"3px",
    });
    timeLabels.innerHTML = `<span id="yunjii-time-current">00:00.00</span><span id="yunjii-time-total">00:00.00</span>`;
    timelineArea.appendChild(timeLabels);

    const trackContainer = document.createElement("div");
    Object.assign(trackContainer.style, {position:"relative",width:"100%",height:"50px"});

    const filmstrip = document.createElement("div");
    Object.assign(filmstrip.style, {
        position:"absolute",top:"0",left:"0",width:"100%",height:"100%",
        display:"flex",background:"#000",borderRadius:"4px",overflow:"hidden",opacity:"0.4",
    });
    trackContainer.appendChild(filmstrip);

    const playhead = document.createElement("div");
    Object.assign(playhead.style, {
        position:"absolute",top:"0",width:"2px",height:"100%",
        background:"#e74c3c",pointerEvents:"none",zIndex:"5",display:"none",
        boxShadow:"0 0 4px rgba(231,76,60,0.8)",
    });
    trackContainer.appendChild(playhead);

    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.step = "0.05"; slider.value = "0";
    Object.assign(slider.style, {
        position:"absolute",top:"-8px",left:"0",width:"100%",height:"calc(100% + 8px)",
        margin:"0",WebkitAppearance:"none",background:"transparent",zIndex:"10",cursor:"pointer",
    });
    trackContainer.appendChild(slider);
    timelineArea.appendChild(trackContainer);

    const markersRow = document.createElement("div");
    Object.assign(markersRow.style, {position:"relative",width:"100%",height:"18px",marginTop:"3px"});
    timelineArea.appendChild(markersRow);

    const bottomBar = document.createElement("div");
    Object.assign(bottomBar.style, {
        display:"flex",justifyContent:"center",alignItems:"center",padding:"8px 16px",borderTop:"1px solid #333",gap:"8px",flexWrap:"wrap",
    });

    const markBtn = document.createElement("button");
    markBtn.textContent = "\u{1F4CC} \u6807\u6CE8";
    Object.assign(markBtn.style, {background:"linear-gradient(135deg,#27ae60,#219a52)",color:"#fff",border:"none",padding:"6px 14px",borderRadius:"5px",cursor:"pointer",fontWeight:"600",fontSize:"12px"});

    const unmarkBtn = document.createElement("button");
    unmarkBtn.textContent = "\u274C \u53D6\u6D88";
    Object.assign(unmarkBtn.style, {background:"linear-gradient(135deg,#e67e22,#d35400)",color:"#fff",border:"none",padding:"6px 14px",borderRadius:"5px",cursor:"pointer",fontWeight:"600",fontSize:"12px"});

    const clearBtn = document.createElement("button");
    clearBtn.textContent = "\u{1F5D1} \u6E05\u7A7A";
    Object.assign(clearBtn.style, {background:"linear-gradient(135deg,#c0392b,#a93226)",color:"#fff",border:"none",padding:"6px 14px",borderRadius:"5px",cursor:"pointer",fontWeight:"600",fontSize:"12px"});

    const countDisplay = document.createElement("span");
    Object.assign(countDisplay.style, {color:"#aaa",fontSize:"12px",minWidth:"80px",textAlign:"center"});

    const applyBtn = document.createElement("button");
    applyBtn.textContent = "\u2705 \u5E94\u7528\u5230\u8282\u70B9";
    Object.assign(applyBtn.style, {background:"linear-gradient(135deg,#3498db,#2980b9)",color:"#fff",border:"none",padding:"6px 18px",borderRadius:"5px",cursor:"pointer",fontWeight:"600",fontSize:"12px"});

    bottomBar.appendChild(markBtn);
    bottomBar.appendChild(unmarkBtn);
    bottomBar.appendChild(clearBtn);
    bottomBar.appendChild(countDisplay);
    bottomBar.appendChild(applyBtn);

    container.appendChild(header);
    container.appendChild(previewArea);
    container.appendChild(timelineArea);
    container.appendChild(bottomBar);
    overlay.appendChild(container);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", (e) => { if (e.target === overlay) { video.remove(); thumbVideo.remove(); overlay.remove(); } });

    const video = document.createElement("video");
    video.muted = true; video.preload = "auto"; video.crossOrigin = "anonymous";
    video.src = videoUrl; video.style.display = "none";
    document.body.appendChild(video);

    const thumbVideo = document.createElement("video");
    thumbVideo.muted = true; thumbVideo.preload = "auto"; thumbVideo.crossOrigin = "anonymous";
    thumbVideo.src = videoUrl; thumbVideo.style.display = "none";
    document.body.appendChild(thumbVideo);

    const ctx = mainCanvas.getContext("2d");
    let sceneData = node._yunjiiScenes || [];

    function updateCountDisplay() {
        countDisplay.textContent = `\u5DF2\u9009: ${selectedFrames.size}\u5E27`;
    }
    updateCountDisplay();

    function getCurrentFrame() {
        return video.duration > 0 ? Math.round(video.currentTime * 30) : -1;
    }

    function markCurrentFrame() {
        const f = getCurrentFrame();
        if (f >= 0) {
            selectedFrames.add(f);
            updateCountDisplay();
            _updateMarkers(markersRow, sceneData, selectedFrames, video.duration);
            statusBadge.style.display = "block";
            statusBadge.textContent = `\u2705 \u5DF2\u6807\u6CE8 \u5E27#${f}`;
            statusBadge.style.background = "rgba(39,174,96,0.9)";
        }
    }

    function unmarkCurrentFrame() {
        const f = getCurrentFrame();
        if (f >= 0 && selectedFrames.has(f)) {
            selectedFrames.delete(f);
            updateCountDisplay();
            _updateMarkers(markersRow, sceneData, selectedFrames, video.duration);
            statusBadge.style.display = "block";
            statusBadge.textContent = `\u274C \u5DF2\u53D6\u6D88 \u5E27#${f}`;
            statusBadge.style.background = "rgba(231,76,60,0.9)";
        }
    }

    function clearAllFrames() {
        selectedFrames.clear();
        updateCountDisplay();
        _updateMarkers(markersRow, sceneData, selectedFrames, video.duration);
        statusBadge.style.display = "block";
        statusBadge.textContent = "\u{1F5D1} \u5DF2\u6E05\u7A7A\u6240\u6709\u6807\u6CE8";
        statusBadge.style.background = "rgba(192,57,43,0.9)";
    }

    mainCanvas.addEventListener("click", (e) => { e.preventDefault(); markCurrentFrame(); });
    mainCanvas.addEventListener("contextmenu", (e) => { e.preventDefault(); unmarkCurrentFrame(); });
    markBtn.onclick = markCurrentFrame;
    unmarkBtn.onclick = unmarkCurrentFrame;
    clearBtn.onclick = clearAllFrames;

    video.onloadedmetadata = () => {
        slider.max = video.duration;
        document.getElementById("yunjii-time-total").textContent = _fmtTime(video.duration);
        mainCanvas.width = video.videoWidth;
        mainCanvas.height = video.videoHeight;
        playhead.style.display = "block";
        video.currentTime = 0;
        _updateMarkers(markersRow, sceneData, selectedFrames, video.duration);
    };

    thumbVideo.onloadedmetadata = () => { _genFilmstrip(thumbVideo, filmstrip); };

    video.addEventListener("seeked", () => {
        ctx.drawImage(video, 0, 0, mainCanvas.width, mainCanvas.height);
        const fi = document.getElementById("yunjii-frame-info");
        if (fi && video.duration > 0) {
            const frame = getCurrentFrame();
            const marked = selectedFrames.has(frame) ? " \u2705" : "";
            fi.textContent = `\u5E27: ${frame} | \u65F6\u95F4: ${_fmtTime(video.currentTime)}${marked}`;
        }
    });

    slider.addEventListener("input", () => {
        const t = parseFloat(slider.value);
        video.currentTime = t;
        document.getElementById("yunjii-time-current").textContent = _fmtTime(t);
        if (video.duration > 0) playhead.style.left = (t / video.duration * 100) + "%";
        const curFrame = getCurrentFrame();
        const si = document.getElementById("yunjii-scene-info");
        if (si && sceneData.length) {
            for (let i = 0; i < sceneData.length; i++) {
                if (curFrame >= sceneData[i].start && curFrame < sceneData[i].end) {
                    si.textContent = `\u{1F3AC}\u955C\u5934 ${i+1}/${sceneData.length}`;
                    break;
                }
            }
        }
    });

    applyBtn.onclick = () => {
        if (manualWidget) {
            manualWidget.value = Array.from(selectedFrames).sort((a,b) => a - b).join(",");
            node.setDirtyCanvas(true, true);
        }
        if (sceneData.length) {
            _showFrameInfo(node, sceneData);
        }
        video.remove(); thumbVideo.remove(); overlay.remove();
    };

    overlay._setSceneData = (scenes) => {
        sceneData = scenes;
        _updateMarkers(markersRow, scenes, selectedFrames, video.duration);
    };
    window._yunjiiFrameBrowser = overlay;
}


function _genFilmstrip(thumbVideo, filmstrip) {
    filmstrip.innerHTML = "";
    const count = 12;
    const step = thumbVideo.duration / count;
    let i = 0;
    function next() {
        if (i >= count) return;
        thumbVideo.currentTime = i * step;
        const handler = () => {
            const c = document.createElement("canvas");
            const ratio = thumbVideo.videoWidth / thumbVideo.videoHeight;
            c.height = 50; c.width = 50 * ratio;
            c.style.cssText = "height:100%;flex-grow:1;object-fit:cover;border-right:1px solid rgba(255,255,255,0.1);";
            c.getContext("2d").drawImage(thumbVideo, 0, 0, c.width, c.height);
            filmstrip.appendChild(c);
            thumbVideo.removeEventListener("seeked", handler);
            i++; next();
        };
        thumbVideo.addEventListener("seeked", handler);
    }
    next();
}


function _updateMarkers(container, scenes, selectedFrames, duration) {
    container.innerHTML = "";
    if (!duration) return;
    const totalFrames = duration * 30;

    if (scenes && scenes.length) {
        scenes.forEach((scene, i) => {
            const midFrame = scene.start + Math.round((scene.end - scene.start) / 2);
            const isMarked = selectedFrames.has(midFrame);
            const marker = document.createElement("div");
            Object.assign(marker.style, {
                position:"absolute",
                left:(scene.start/totalFrames*100)+"%",
                width:((scene.end-scene.start)/totalFrames*100)+"%",
                height:"14px",top:"0",
                background:isMarked?"#27ae60":"#3498db",
                borderRadius:"2px",opacity:"0.7",
                fontSize:"9px",color:"#fff",display:"flex",alignItems:"center",
                justifyContent:"center",fontWeight:"600",
            });
            marker.textContent = `${i+1}${isMarked?"\u2705":""}`;
            marker.title = `\u955C\u5934${i+1}: \u5E27${scene.start}-${scene.end}`;
            container.appendChild(marker);
        });
    }

    if (selectedFrames && selectedFrames.size) {
        const sceneMids = new Set();
        if (scenes) scenes.forEach(s => sceneMids.add(s.start + Math.round((s.end - s.start) / 2)));

        selectedFrames.forEach(f => {
            if (sceneMids.has(f)) return;
            const dot = document.createElement("div");
            Object.assign(dot.style, {
                position:"absolute",
                left:(f/totalFrames*100)+"%",
                width:"4px",height:"14px",top:"0",
                background:"#e74c3c",borderRadius:"2px",
            });
            dot.title = `\u{1F4CC}\u624B\u52A8 \u5E27#${f}`;
            container.appendChild(dot);
        });
    }
}


function _fmtTime(s) {
    const m = Math.floor(s/60), sec = Math.floor(s%60), ms = Math.floor((s%1)*100);
    return `${m<10?'0'+m:m}:${sec<10?'0'+sec:sec}.${ms<10?'0'+ms:ms}`;
}


function _uploadVideo(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*,.mp4,.avi,.mov,.mkv,.webm";
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const btn = node.widgets.find(w => w.name === "📤 上传视频");
        if (btn) btn.name = "⏳ 上传中...";

        try {
            const formData = new FormData();
            formData.append("file", file);

            const resp = await fetch("/yunjii/upload_video", {
                method: "POST",
                body: formData,
            });

            const data = await resp.json();
            if (data.saved) {
                const vidWidget = node.widgets.find(w => w.name === "视频文件");
                if (vidWidget) {
                    vidWidget.value = data.saved;
                    node.setDirtyCanvas(true, true);
                }
                if (btn) btn.name = "✅ 上传成功";
                setTimeout(() => { if (btn) btn.name = "📤 上传视频"; }, 2000);
            } else {
                alert("上传失败: " + (data.error || "未知错误"));
                if (btn) btn.name = "📤 上传视频";
            }
        } catch (err) {
            alert("上传失败: " + err.message);
            if (btn) btn.name = "📤 上传视频";
        }
    };
    input.click();
}


function _addDevToolbar() {
    const existing = document.getElementById("yunjii-dev-toolbar");
    if (existing) return;

    const toolbar = document.createElement("div");
    toolbar.id = "yunjii-dev-toolbar";
    Object.assign(toolbar.style, {
        position: "fixed", bottom: "10px", right: "10px", zIndex: "10000",
        display: "flex", gap: "6px", alignItems: "center",
        background: "rgba(20,20,30,0.9)", padding: "6px 10px",
        borderRadius: "8px", boxShadow: "0 2px 12px rgba(0,0,0,0.5)",
        fontFamily: "monospace", fontSize: "11px",
    });

    const label = document.createElement("span");
    label.style.cssText = "color:#4cf;font-weight:600;font-size:11px;";
    label.textContent = "Yunjii";
    toolbar.appendChild(label);

    const reloadBtn = document.createElement("button");
    reloadBtn.textContent = "🔄 热重载";
    Object.assign(reloadBtn.style, {
        background: "linear-gradient(135deg,#3498db,#2980b9)", color: "#fff",
        border: "none", padding: "4px 10px", borderRadius: "4px",
        cursor: "pointer", fontSize: "11px", fontWeight: "600",
    });
    reloadBtn.onclick = async () => {
        reloadBtn.textContent = "⏳ 重载中...";
        reloadBtn.style.opacity = "0.6";
        try {
            const resp = await fetch("/yunjii/reload", { method: "POST" });
            const data = await resp.json();
            if (data.status === "ok" || data.status === "partial") {
                const info = `${data.reloaded_modules}模块, ${data.node_count}节点`;
                reloadBtn.textContent = `✅ ${info}`;
                if (data.errors && data.errors.length > 0) {
                    console.warn("[Yunjii] Reload errors:", data.errors);
                }
                setTimeout(() => { reloadBtn.textContent = "🔄 热重载"; reloadBtn.style.opacity = "1"; }, 3000);
            } else {
                reloadBtn.textContent = "❌ 失败";
                setTimeout(() => { reloadBtn.textContent = "🔄 热重载"; reloadBtn.style.opacity = "1"; }, 3000);
            }
        } catch (err) {
            reloadBtn.textContent = "❌ 连接失败";
            setTimeout(() => { reloadBtn.textContent = "🔄 热重载"; reloadBtn.style.opacity = "1"; }, 3000);
        }
    };
    toolbar.appendChild(reloadBtn);

    const logBtn = document.createElement("button");
    logBtn.textContent = "📋 日志";
    Object.assign(logBtn.style, {
        background: "linear-gradient(135deg,#27ae60,#219a52)", color: "#fff",
        border: "none", padding: "4px 10px", borderRadius: "4px",
        cursor: "pointer", fontSize: "11px", fontWeight: "600",
    });
    logBtn.onclick = () => _openLogPanel();
    toolbar.appendChild(logBtn);

    document.body.appendChild(toolbar);
}


function _openLogPanel() {
    const existing = document.getElementById("yunjii-log-overlay");
    if (existing) { existing.remove(); return; }

    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "15000",
        background: "rgba(0,0,0,0.85)", display: "flex",
        flexDirection: "column", alignItems: "center", justifyContent: "center",
    });

    const container = document.createElement("div");
    Object.assign(container.style, {
        width: "80vw", maxWidth: "900px", height: "75vh",
        background: "#1e1e1e", borderRadius: "12px",
        display: "flex", flexDirection: "column", overflow: "hidden",
        boxShadow: "0 10px 50px rgba(0,0,0,0.8)",
    });

    const header = document.createElement("div");
    Object.assign(header.style, {
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "10px 20px", borderBottom: "1px solid #333",
        color: "#fff", fontSize: "14px", fontWeight: "600",
    });
    header.innerHTML = "<span>📋 云集智能调试日志</span>";

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕ 关闭";
    Object.assign(closeBtn.style, { background: "#444", color: "#fff", border: "none", padding: "5px 14px", borderRadius: "6px", cursor: "pointer", fontSize: "12px" });
    closeBtn.onclick = () => overlay.remove();
    header.appendChild(closeBtn);

    const controls = document.createElement("div");
    Object.assign(controls.style, {
        display: "flex", gap: "8px", padding: "8px 16px", borderBottom: "1px solid #333", alignItems: "center",
    });

    const fileSelect = document.createElement("select");
    Object.assign(fileSelect.style, {
        background: "#2a2a2a", color: "#ccc", border: "1px solid #444",
        padding: "4px 8px", borderRadius: "4px", fontSize: "11px", fontFamily: "monospace",
    });
    controls.appendChild(fileSelect);

    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新";
    Object.assign(refreshBtn.style, { background: "#3498db", color: "#fff", border: "none", padding: "4px 10px", borderRadius: "4px", cursor: "pointer", fontSize: "11px" });
    controls.appendChild(refreshBtn);

    const tailInput = document.createElement("input");
    tailInput.type = "number"; tailInput.value = "100"; tailInput.min = "10"; tailInput.max = "5000";
    Object.assign(tailInput.style, { background: "#2a2a2a", color: "#ccc", border: "1px solid #444", padding: "4px 6px", borderRadius: "4px", fontSize: "11px", width: "60px" });
    controls.appendChild(tailInput);

    const tailLabel = document.createElement("span");
    tailLabel.style.cssText = "color:#888;font-size:11px;";
    tailLabel.textContent = "行";
    controls.appendChild(tailLabel);

    const logContent = document.createElement("pre");
    Object.assign(logContent.style, {
        flex: "1", overflow: "auto", padding: "12px 16px",
        color: "#c9d1d9", fontFamily: "monospace", fontSize: "11px",
        lineHeight: "1.5", margin: "0", whiteSpace: "pre-wrap", wordBreak: "break-all",
    });

    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(logContent);
    overlay.appendChild(container);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

    async function loadLogs() {
        try {
            const resp = await fetch("/yunjii/logs");
            const data = await resp.json();
            fileSelect.innerHTML = "";
            if (data.recent_logs) {
                for (const f of data.recent_logs) {
                    const opt = document.createElement("option");
                    opt.value = f; opt.textContent = f;
                    if (f === data.current) opt.selected = true;
                    fileSelect.appendChild(opt);
                }
            }
        } catch (err) {
            fileSelect.innerHTML = "<option>加载失败</option>";
        }
    }

    async function loadLogContent() {
        const file = fileSelect.value;
        const tail = tailInput.value || "100";
        if (!file) return;
        logContent.textContent = "加载中...";
        try {
            const resp = await fetch(`/yunjii/logs?file=${encodeURIComponent(file)}&tail=${tail}`);
            const data = await resp.json();
            let text = data.content || "(空)";
            text = text.replace(/\[DEBUG\]/g, "%c[DEBUG]").replace(/\[INFO\]/g, "%c[INFO]").replace(/\[WARNING\]/g, "%c[WARNING]").replace(/\[ERROR\]/g, "%c[ERROR]");

            const lines = (data.content || "").split("\n");
            const colored = lines.map(line => {
                if (line.includes("[ERROR]")) return `<span style="color:#f85149">${line}</span>`;
                if (line.includes("[WARNING]")) return `<span style="color:#f0883e">${line}</span>`;
                if (line.includes("[INFO]")) return `<span style="color:#58a6ff">${line}</span>`;
                if (line.includes("[DEBUG]")) return `<span style="color:#8b949e">${line}</span>`;
                if (line.includes("节点开始执行")) return `<span style="color:#7ee787;font-weight:600">${line}</span>`;
                if (line.includes("节点执行完成")) return `<span style="color:#3fb950;font-weight:600">${line}</span>`;
                if (line.includes("节点执行失败")) return `<span style="color:#f85149;font-weight:600">${line}</span>`;
                return line;
            }).join("\n");
            logContent.innerHTML = colored;
            logContent.scrollTop = logContent.scrollHeight;
        } catch (err) {
            logContent.textContent = "加载失败: " + err.message;
        }
    }

    refreshBtn.onclick = () => { loadLogs(); loadLogContent(); };
    fileSelect.onchange = loadLogContent;

    loadLogs().then(() => loadLogContent());
}
