# AI Study Buddy - Startup Script
# Double-click this file to start the app

Write-Host "🚀 Starting AI Study Buddy App..." -ForegroundColor Cyan
Write-Host ""

$pythonPath = "C:\Users\srish\AppData\Local\Programs\Python\Python313\python.exe"
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if Python exists
if (-not (Test-Path $pythonPath)) {
    Write-Host "❌ ERROR: Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Please update the pythonPath variable in this script." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Navigate to app directory
Set-Location $appDir

# Kill any existing Streamlit on port 8501
Write-Host "Checking for existing instances..." -ForegroundColor Yellow
$existing = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Closing existing Streamlit instances..." -ForegroundColor Yellow
    $existing | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# Start Streamlit
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "Your app will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

& $pythonPath -m streamlit run app.py --server.port 8501

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ An error occurred. Check the messages above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}

