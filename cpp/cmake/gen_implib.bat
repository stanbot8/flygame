@echo off
REM Generate MuJoCo import library from DLL.
REM Usage: gen_implib.bat <mujoco.dll> <output_dir>

set DLL=%1
set OUTDIR=%2

if "%DLL%"=="" (
    echo Usage: gen_implib.bat mujoco.dll output_dir
    exit /b 1
)
if "%OUTDIR%"=="" set OUTDIR=.

dumpbin /exports "%DLL%" > "%OUTDIR%\mujoco_raw.txt"

echo LIBRARY mujoco > "%OUTDIR%\mujoco.def"
echo EXPORTS >> "%OUTDIR%\mujoco.def"
for /f "tokens=4" %%a in ('findstr /r "^ *[0-9]" "%OUTDIR%\mujoco_raw.txt"') do (
    echo     %%a >> "%OUTDIR%\mujoco.def"
)

lib /def:"%OUTDIR%\mujoco.def" /out:"%OUTDIR%\mujoco.lib" /machine:x64
del "%OUTDIR%\mujoco_raw.txt"
echo Generated: %OUTDIR%\mujoco.lib
