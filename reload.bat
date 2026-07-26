@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set COMFYUI_HOST=127.0.0.1:8188
if "%1" neq "" set COMFYUI_HOST=%1

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   🔄 云集智能插件 - 热重载工具       ║
echo  ╚══════════════════════════════════════╝
echo.

echo [1/3] 热重载插件模块...
curl -s -X POST http://%COMFYUI_HOST%/yunjii/reload > "%TEMP%\yunjii_reload.json" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   ❌ 无法连接到 ComfyUI (%COMFYUI_HOST%)
    echo   请确认 ComfyUI 正在运行
    goto :end
)

:: Read and display result
for /f "usebackq delims=" %%a in ("%TEMP%\yunjii_reload.json") do set RESULT=%%a
echo   !RESULT!
echo.

echo [2/3] 查看插件状态...
curl -s http://%COMFYUI_HOST%/yunjii/status > "%TEMP%\yunjii_status.json" 2>nul
for /f "usebackq delims=" %%a in ("%TEMP%\yunjii_status.json") do set STATUS=%%a
echo   !STATUS!
echo.

echo [3/3] 最新日志 (最后20行)...
for /f "usebackq delims=" %%a in ("%TEMP%\yunjii_status.json") do (
    echo %%a | findstr /C:"current_log" >nul && (
        for /f "tokens=2 delims=:," %%b in ("%%a") do (
            set LOGFILE=%%~b
        )
    )
)
curl -s "http://%COMFYUI_HOST%/yunjii/logs?tail=20" 2>nul
echo.

:end
echo.
echo  ┌──────────────────────────────────────┐
echo  │  ✅ 热重载完成                        │
echo  │                                      │
echo  │  适用: 修改节点内部逻辑后快速验证      │
echo  │  不适用: 新增/删除节点、修改输入输出    │
echo  │  不适用时需重启 ComfyUI               │
echo  │                                      │
echo  │  日志目录: output/yunjii_logs/        │
echo  └──────────────────────────────────────┘
echo.

del "%TEMP%\yunjii_reload.json" 2>nul
del "%TEMP%\yunjii_status.json" 2>nul
pause
