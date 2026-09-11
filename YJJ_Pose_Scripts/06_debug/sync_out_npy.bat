@echo off
setlocal

REM ============================================================
REM 按 out_image 的完整时间戳，从 npy 同步出 out_npy
REM
REM 用法:
REM   sync_out_npy.bat ^<session_dir^> [python_exe]
REM
REM 例:
REM   sync_out_npy.bat H:\YJJ\Yolo_RGBD\Resource\session3_200348
REM   sync_out_npy.bat H:\YJJ\Yolo_RGBD\Resource\session4_200449 H:\YJJ\Conda\python.exe
REM
REM 说明:
REM   session_dir 需同时包含 out_image 与 npy 两个子目录。
REM   输出写入 session_dir\out_npy，写入前会清空该目录下的 *.npy。
REM ============================================================

if "%~1"=="" (
    echo [ERROR] 缺少 session 目录参数
    echo 用法: %~nx0 ^<session_dir^> [python_exe]
    echo 例:   %~nx0 H:\YJJ\Yolo_RGBD\Resource\session3_200348
    exit /b 1
)

set "ROOT=%~1"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PYTHON=%~2"
if "%PYTHON%"=="" set "PYTHON=H:\YJJ\Conda\python.exe"

if not exist "%ROOT%\npy" (
    echo [ERROR] 找不到 %ROOT%\npy
    exit /b 1
)

if not exist "%ROOT%\out_image" (
    echo [ERROR] 找不到 %ROOT%\out_image
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [ERROR] 找不到 python: %PYTHON%
    exit /b 1
)

echo Session : %ROOT%
echo Python  : %PYTHON%
echo.

set "SYNC_ROOT=%ROOT%"
"%PYTHON%" -c "from pathlib import Path; import os,re,sys,shutil; root=Path(os.environ['SYNC_ROOT']); src=root/'npy'; refs=root/'out_image'; dst=root/'out_npy'; pat=re.compile(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d+'); ts={m.group(0) for p in refs.iterdir() if p.is_file() for m in [pat.search(p.stem)] if m}; sys.exit('[ERROR] out_image 为空，已中止，未修改 out_npy') if not ts else None; dst.mkdir(exist_ok=True); [p.unlink() for p in dst.glob('*.npy')]; matches=[p for p in src.glob('*.npy') if (m:=pat.search(p.stem)) and m.group(0) in ts]; [shutil.copy2(p,dst/p.name) for p in matches]; print('out_image=',len(ts)); print('matched npy=',len(matches)); print('out_npy=',len(list(dst.glob('*.npy')))); print('copied to=',dst)"

if errorlevel 1 (
    echo.
    echo [ERROR] 同步失败
    exit /b 1
)

echo.
echo Done.
pause
