param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$backend = Join-Path $PSScriptRoot 'backend'
$python = Join-Path $backend '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Backend Python is missing: $python. Create backend/.venv and install backend/requirements.txt."
}
Push-Location -LiteralPath $backend
try {
    & $python -B -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    $backendExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $backendExit
