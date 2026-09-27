@echo off
rem DepthWizard master CLI wrapper (Windows).
rem Finds any usable Python 3 and runs scripts\dw.py - stdlib only.
setlocal
set "DIR=%~dp0"

set "PY="
for %%P in (py python python3) do (
  if not defined PY (
    where %%P >nul 2>nul && set "PY=%%P"
  )
)

if not defined PY (
  echo DepthWizard CLI needs Python 3.8+ on PATH ^(the project itself needs 3.12+^).
  echo Install it first:  https://www.python.org/downloads/
  exit /b 3
)

"%PY%" "%DIR%scripts\dw.py" %*
exit /b %errorlevel%
