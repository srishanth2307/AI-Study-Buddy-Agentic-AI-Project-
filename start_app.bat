@echo off
echo Starting AI Study Buddy App...
echo.

REM Find Python executable
set PYTHON_PATH=C:\Users\srish\AppData\Local\Programs\Python\Python313\python.exe

REM Navigate to app directory
cd /d "%~dp0"

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Please update PYTHON_PATH in this file.
    pause
    exit /b 1
)

REM Kill any existing Streamlit processes on port 8501
netstat -ano | findstr ":8501" >nul 2>&1
if %errorlevel% == 0 (
    echo Closing existing Streamlit instances...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

REM Start Streamlit
echo Starting Streamlit app...
echo.
"%PYTHON_PATH%" -m streamlit run app.py --server.port 8501

pause

