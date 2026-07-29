[CmdletBinding()]
param(
    [string]$Config = "config\default.toml",
    [string]$Output = "",
    [ValidateSet("full", "fast", "smoke")]
    [string]$Profile = "fast",
    [switch]$Refresh,
    [switch]$TestsOnly,
    [switch]$SkipTests,
    [switch]$Install
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($TestsOnly -and $SkipTests) {
    throw "Use apenas um entre -TestsOnly e -SkipTests."
}

$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# Cada processo cria temporarios proprios. Isso evita ACLs herdadas de
# execucoes anteriores, OneDrive, pytest ou ambientes de sandbox.
$SystemTempRoot = [System.IO.Path]::GetTempPath()
$RunIdentifier = [System.Guid]::NewGuid().ToString("N")
$RuntimeRoot = Join-Path $SystemTempRoot "itau_quant_$RunIdentifier"
$TempRoot = Join-Path $RuntimeRoot "temp"
$MatplotlibRoot = Join-Path $RuntimeRoot "matplotlib"
New-Item -ItemType Directory -Force -Path $TempRoot, $MatplotlibRoot | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:MPLCONFIGDIR = $MatplotlibRoot
$env:PYTHONUTF8 = "1"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "[setup] Criando .venv..." -ForegroundColor Cyan
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $SystemPython) {
        throw "Python nao encontrado no PATH. Instale Python 3.11 ou superior."
    }
    & $SystemPython.Source -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar a .venv."
    }
    $Install = $true
}

if (-not $Install) {
    & $VenvPython -c "import matplotlib, numpy, pandas, pytest, scipy, yfinance, network_momentum" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $Install = $true
    }
}

if ($Install) {
    Write-Host "[setup] Instalando dependencias..." -ForegroundColor Cyan
    & $VenvPython -m pip install --disable-pip-version-check -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar as dependencias."
    }
}

if (-not $SkipTests) {
    Write-Host "[test] Executando a suite..." -ForegroundColor Cyan
    & $VenvPython -m pytest -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {
        throw "Os testes falharam; o pipeline nao foi iniciado."
    }
    Write-Host "[test] Todos os testes passaram." -ForegroundColor Green
}

if ($TestsOnly) {
    Write-Host "[fim] Validacao concluida." -ForegroundColor Green
    exit 0
}

$ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $ProjectRoot $Config
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Configuracao nao encontrada: $ConfigPath"
}

$PipelineArguments = @(
    "-m",
    "network_momentum.cli",
    "--config",
    $ConfigPath,
    "--profile",
    $Profile
)
if ($Refresh) {
    $PipelineArguments += "--refresh"
}
if ($Output) {
    $OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
        $Output
    } else {
        Join-Path $ProjectRoot $Output
    }
    $PipelineArguments += @("--output", $OutputPath)
}

Write-Host "[pipeline] Iniciando download, features, grafos e backtest..." -ForegroundColor Cyan
& $VenvPython @PipelineArguments
if ($LASTEXITCODE -ne 0) {
    throw "O pipeline terminou com erro."
}

Write-Host "[fim] Pipeline concluido. Consulte a pasta outputs." -ForegroundColor Green
