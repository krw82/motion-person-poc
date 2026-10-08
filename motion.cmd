@echo off
if exist "%~dp0.venv\Scripts\python.exe" goto venv
python "%~dp0cli.py" %*
exit /b %errorlevel%
:venv
"%~dp0.venv\Scripts\python.exe" "%~dp0cli.py" %*
exit /b %errorlevel%
