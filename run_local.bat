@echo off
echo ========================================================
echo   SatQuery AI - Local Development Runner
echo   SIH 2026 - Problem Statement 26167 (ISRO/SAC)
echo ========================================================
echo.

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=..\.venv\Scripts\python.exe

echo Starting FastAPI Backend on port 8000 (using %PYTHON_CMD%)...
start "SatQuery Backend" cmd /k "cd backend && %PYTHON_CMD% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 >nul

echo Starting React Frontend on port 5173...
start "SatQuery Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo Both services are launching:
echo   - Backend API: http://localhost:8000/health
echo   - Frontend UI: http://localhost:5173
echo ========================================================
