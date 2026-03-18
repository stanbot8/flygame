@echo off
REM flygame launcher
REM Usage:
REM   launch.bat              - default CPG controller
REM   launch.bat --hybrid     - hybrid controller (cubic spline + corrections)
REM   launch.bat --stepper    - IK stepper (experimental)
REM   launch.bat --headless   - no viewer window
REM   launch.bat --hybrid --headless --duration 5

cd /d "%~dp0cpp\build\Release"
nmfly-sim.exe --hybrid %*
