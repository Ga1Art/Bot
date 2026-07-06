param(
    [ValidateSet("setup", "db", "migrate", "api", "bot", "collect", "rescore", "previewscore", "analyzeai", "syncsheets", "synchot", "setupsheets", "ordermode", "expire", "doctor", "all")]
    [string]$Action = "api",
    [string]$ShareWith = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\pip.exe"

function Test-Venv {
    if (-not (Test-Path $PythonExe)) {
        throw "Virtual environment not found. Run: .\run.ps1 -Action setup"
    }
}

function Invoke-Setup {
    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating virtual environment..."
        python -m venv $VenvPath
    }

    Write-Host "Installing Python dependencies..."
    & $PipExe install -r (Join-Path $ProjectRoot "requirements.txt")
}

function Invoke-Db {
    Write-Host "Starting PostgreSQL via docker compose..."
    docker compose up -d
}

function Invoke-Migrate {
    Test-Venv
    Write-Host "Applying Alembic migrations..."
    & $PythonExe -m alembic upgrade head
}

function Invoke-Api {
    Test-Venv
    Write-Host "Starting FastAPI app from .env settings..."
    & $PythonExe -m app.scripts.run_api
}

function Invoke-Bot {
    Test-Venv
    Write-Host "Starting Telegram bot..."
    & $PythonExe -m app.scripts.run_bot
}

function Invoke-Collect {
    Test-Venv
    Write-Host "Running all collectors..."
    & $PythonExe -m app.scripts.run_all_collectors
}

function Invoke-SyncSheets {
    Test-Venv
    Write-Host "Syncing queue to Google Sheets..."
    & $PythonExe -m app.scripts.sync_google_sheets
}

function Invoke-SyncHot {
    Test-Venv
    Write-Host "Syncing hot prospects to Google Sheets..."
    & $PythonExe -m app.scripts.sync_google_sheets --hot
}

function Invoke-SetupSheets {
    Test-Venv
    Write-Host "Creating or preparing Google Sheets..."
    if ($ShareWith) {
        & $PythonExe -m app.scripts.setup_google_sheets --share-with $ShareWith
    }
    else {
        & $PythonExe -m app.scripts.setup_google_sheets
    }
}

function Invoke-OrderMode {
    Test-Venv
    Write-Host "Switching existing data to order-focused mode..."
    & $PythonExe -m app.scripts.switch_to_order_mode
}

function Invoke-Expire {
    Test-Venv
    Write-Host "Moving expired open leads to context..."
    & $PythonExe -m app.scripts.expire_stale_leads
}

function Invoke-Rescore {
    Test-Venv
    Write-Host "Recalculating lead scores..."
    & $PythonExe -m app.scripts.rescore_leads
}

function Invoke-PreviewScore {
    Test-Venv
    Write-Host "Previewing score breakdown for recent leads..."
    & $PythonExe -m app.scripts.preview_scores --limit 12 --sort recent
}

function Invoke-AnalyzeAi {
    Test-Venv
    Write-Host "Running AI analysis for active leads..."
    & $PythonExe -m app.scripts.analyze_ai_leads
}

function Invoke-Doctor {
    Test-Venv
    Write-Host "Checking first-run readiness..."
    & $PythonExe -m app.scripts.doctor
}

function Invoke-All {
    Invoke-Db
    Invoke-Migrate
    Invoke-Doctor
    Invoke-Collect
}

Push-Location $ProjectRoot
try {
    switch ($Action) {
        "setup" { Invoke-Setup }
        "db" { Invoke-Db }
        "migrate" { Invoke-Migrate }
        "api" { Invoke-Api }
        "bot" { Invoke-Bot }
        "collect" { Invoke-Collect }
        "rescore" { Invoke-Rescore }
        "previewscore" { Invoke-PreviewScore }
        "analyzeai" { Invoke-AnalyzeAi }
        "syncsheets" { Invoke-SyncSheets }
        "synchot" { Invoke-SyncHot }
        "setupsheets" { Invoke-SetupSheets }
        "ordermode" { Invoke-OrderMode }
        "expire" { Invoke-Expire }
        "doctor" { Invoke-Doctor }
        "all" { Invoke-All }
    }
}
finally {
    Pop-Location
}
