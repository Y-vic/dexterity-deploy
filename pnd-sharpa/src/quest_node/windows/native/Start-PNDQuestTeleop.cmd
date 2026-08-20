@echo off
setlocal
set "DISABLE_XR_APILAYER_MANUS_handtracking=1"
if not defined PND_QUEST_ROS_HOST set "PND_QUEST_ROS_HOST=10.10.20.127"
if not defined PND_QUEST_ZED_HOST set "PND_QUEST_ZED_HOST=10.10.20.126"
if not defined PND_QUEST_ZED_PORT set "PND_QUEST_ZED_PORT=5602"
if not defined PND_QUEST_DISABLE_HW_DECODE set "PND_QUEST_DISABLE_HW_DECODE=1"
if not defined PND_QUEST_NETWORK_CACHE_MS set "PND_QUEST_NETWORK_CACHE_MS=100"

taskkill /im PNDQuestTeleop.exe /f >nul 2>&1
timeout /t 1 /nobreak >nul

if exist "%~dp0update\PNDQuestTeleop.exe" (
  if not exist "%~dp0publish" mkdir "%~dp0publish"
  xcopy "%~dp0update\*" "%~dp0publish\" /e /i /y >nul
  if errorlevel 1 (
    echo Failed to update the PND Quest Teleoperation publish directory.
    pause
    exit /b 1
  )
  rmdir /s /q "%~dp0update"
)

set "APP=%~dp0publish\PNDQuestTeleop.exe"
if not exist "%APP%" set "APP=%~dp0PNDQuestTeleop.exe"

for /f "tokens=2,*" %%A in ('reg query "HKLM\SOFTWARE\Khronos\OpenXR\1" /v ActiveRuntime 2^>nul ^| findstr /i "ActiveRuntime"') do set "XR_RUNTIME_JSON=%%B"

if not defined XR_RUNTIME_JSON (
  echo No active OpenXR runtime was found.
  echo Start Meta Quest Link and set Meta Quest Link as the active OpenXR runtime.
  pause
  exit /b 1
)

echo %XR_RUNTIME_JSON% | findstr /i "oculus_openxr_64.json" >nul
if errorlevel 1 (
  echo The active OpenXR runtime is not Meta Quest Link:
  echo %XR_RUNTIME_JSON%
  echo Set Meta Quest Link as the active OpenXR runtime and try again.
  pause
  exit /b 1
)

if not exist "%APP%" (
  echo PNDQuestTeleop.exe was not found.
  pause
  exit /b 1
)

if /i "%~1"=="--check" (
  echo Meta OpenXR runtime: %XR_RUNTIME_JSON%
  echo Application: %APP%
  echo ROS host: %PND_QUEST_ROS_HOST%
  echo ZED stream: %PND_QUEST_ZED_HOST%:%PND_QUEST_ZED_PORT%
  exit /b 0
)

start "PND Quest Teleoperation" "%APP%"
