<#
.SYNOPSIS
    Windows task runner. Mirrors the Makefile target for target.

.DESCRIPTION
    The Makefile is the entry point on Linux and in CI. It is not usable here:
    Windows has no make, and its clean/help targets are written against GNU
    coreutils. Rather than ask everyone on Windows to retype the underlying
    python -m invocations, this runs the same commands through the project
    virtualenv, so the documented workflow is the workflow on both platforms.

    Two things are set for every target:

      HF_HOME     - points the HuggingFace cache outside the repository. The
                    checkpoints total well over a gigabyte and the repository
                    lives in a synced folder; there is no reason for a cache to
                    be uploaded anywhere.
      PYTHONUTF8  - Windows still defaults stdio to the ANSI codepage, which
                    turns any em-dash in a generated table into a crash rather
                    than a character.

.EXAMPLE
    .\make.ps1 help
    .\make.ps1 test
    .\make.ps1 sweep
    .\make.ps1 all
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    Write-Warning "No virtualenv at .venv - falling back to the python on PATH. Run '.\make.ps1 venv' to create one."
    $Python = 'python'
}

if (-not $env:HF_HOME) { $env:HF_HOME = 'C:\hf-cache' }
$env:PYTHONUTF8 = '1'

$ModelDir = Join-Path $RepoRoot 'artifacts\run\model'
$OnnxDir  = Join-Path $RepoRoot 'artifacts\onnx'
$Image    = if ($env:MEDEXTRACT_IMAGE) { $env:MEDEXTRACT_IMAGE } else { 'medextract:local' }
$ApiUrl   = if ($env:MEDEXTRACT_API_URL) { $env:MEDEXTRACT_API_URL } else { 'http://localhost:8000' }

# Kept in declaration order so `help` reads as a workflow rather than an
# alphabetical list.
$Targets = [ordered]@{
    'venv'        = 'Create .venv and install everything needed to train and serve'
    'install'     = 'Runtime + dev dependencies'
    'lint'        = 'ruff check + format check'
    'format'      = 'Apply ruff formatting and autofixes'
    'test'        = 'Run the test suite'
    'cov'         = 'Test suite with coverage'
    'audit'       = 'Known vulnerabilities in what the serving image installs'
    'train'       = 'One fine-tuning run'
    'sweep'       = 'The full nine-arm comparison sweep'
    'report'      = 'Regenerate the MLflow comparison table'
    'export'      = 'Export to ONNX, verify parity, export the query encoder'
    'benchmark'   = 'PyTorch vs ONNX latency table'
    'loadtest'    = 'Latency under concurrency against a running API (up first)'
    'calibrate'   = 'Reliability of the entity score on the validation split'
    'fetch'       = 'Download PubMed abstracts'
    'index'       = 'Embed and index the abstracts into Qdrant'
    'serve'       = 'Run the API locally with reload'
    'docker'      = 'Build the serving image'
    'docker-size' = 'Print the built image size in MB'
    'up'          = 'Bring up the API and Qdrant'
    'down'        = 'Tear it down'
    'clean'       = 'Remove caches and build output'
    'all'         = 'The whole pipeline, unattended'
}

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkCyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe exited with $LASTEXITCODE"
    }
}

function Invoke-Py { param([string[]]$Arguments) Invoke-Step -Exe $Python -Arguments $Arguments }

switch ($Target) {
    'help' {
        Write-Host ''
        Write-Host '  MedExtract targets' -ForegroundColor White
        Write-Host ''
        foreach ($name in $Targets.Keys) {
            Write-Host ('  {0,-14}' -f $name) -ForegroundColor Cyan -NoNewline
            Write-Host $Targets[$name]
        }
        Write-Host ''
        Write-Host '  Anything after the target is passed through, e.g.' -ForegroundColor DarkGray
        Write-Host '    .\make.ps1 train --lr 5e-5 --epochs 3' -ForegroundColor DarkGray
        Write-Host ''
    }

    'venv' {
        # --system-site-packages so an existing CUDA torch build is reused
        # rather than re-downloading roughly 2.5GB of wheels. Packages
        # installed into the venv still take precedence over the system ones,
        # so the project's own pins win without touching anything outside it.
        Invoke-Step -Exe 'python' -Arguments @('-m', 'venv', '.venv', '--system-site-packages')
        $venvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
        Invoke-Step -Exe $venvPy -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Step -Exe $venvPy -Arguments @(
            '-m', 'pip', 'install',
            'transformers>=4.57,<6', 'datasets>=2.18', 'mlflow', 'accelerate',
            'onnx', 'onnxruntime', 'qdrant-client', 'pytest', 'pytest-cov', 'ruff', 'pip-audit'
        )
        Invoke-Step -Exe $venvPy -Arguments @('-m', 'pip', 'install', '-e', '.', '--no-deps')
    }

    'install'     { Invoke-Py @('-m', 'pip', 'install', '-e', '.[dev,serve]') }
    'lint'        { Invoke-Py @('-m', 'ruff', 'check', '.'); Invoke-Py @('-m', 'ruff', 'format', '--check', '.') }
    'format'      { Invoke-Py @('-m', 'ruff', 'check', '--fix', '.'); Invoke-Py @('-m', 'ruff', 'format', '.') }
    'test'        { Invoke-Py (@('-m', 'pytest', '-q') + $Rest) }
    'cov'         { Invoke-Py (@('-m', 'pytest', '-q', '--cov=medextract', '--cov-report=term-missing') + $Rest) }
    'audit'       { Invoke-Py (@('-m', 'pip_audit', '-r', 'requirements-serve.txt', '--strict') + $Rest) }

    'train'       { Invoke-Py (@('-m', 'medextract.train') + $Rest) }
    'sweep'       { Invoke-Py (@('-m', 'medextract.sweep') + $Rest) }

    'report' {
        $out = Join-Path $RepoRoot 'docs\sweep_table.md'
        New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
        # Set-Content defaults to the ANSI codepage on Windows PowerShell, which
        # would mangle the table; utf8 is explicit for that reason.
        & $Python -m medextract.report --experiment medextract-ner | Out-File -FilePath $out -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "report failed with $LASTEXITCODE" }
        Write-Host "wrote $out" -ForegroundColor Green
    }

    'export' {
        Invoke-Py @('-m', 'medextract.export_onnx', '--model-dir', $ModelDir, '--output-dir', $OnnxDir)
        Invoke-Py @('-m', 'medextract.retrieval.export_embedder', '--output-dir', (Join-Path $RepoRoot 'artifacts\embedder'))
    }

    'benchmark'   { Invoke-Py (@('-m', 'medextract.benchmark', '--torch-dir', $ModelDir, '--onnx-dir', $OnnxDir) + $Rest) }
    'loadtest'    { Invoke-Py (@('-m', 'medextract.loadtest', '--url', $ApiUrl, '--corpus', 'data\pubmed.jsonl') + $Rest) }
    'calibrate'   { Invoke-Py (@('-m', 'medextract.calibration', '--model-dir', $OnnxDir) + $Rest) }
    'fetch'       { Invoke-Py (@('-m', 'medextract.retrieval.pubmed', '--max', '3000') + $Rest) }
    'index'       { Invoke-Py (@('-m', 'medextract.retrieval.build_index') + $Rest) }
    'serve'       { Invoke-Py @('-m', 'uvicorn', 'medextract.api.main:app', '--reload', '--port', '8000') }

    'docker'      { Invoke-Step -Exe 'docker' -Arguments @('build', '-t', $Image, '.') }
    'docker-size' {
        $bytes = (& docker image inspect $Image --format '{{.Size}}')
        if ($LASTEXITCODE -ne 0) { throw "no such image: $Image" }
        Write-Host ('{0} MB' -f [math]::Round([int64]$bytes / 1MB))
    }
    'up'          { Invoke-Step -Exe 'docker' -Arguments @('compose', 'up', '--build', '-d') }
    'down'        { Invoke-Step -Exe 'docker' -Arguments @('compose', 'down') }

    'clean' {
        foreach ($path in @('.pytest_cache', '.ruff_cache', '.coverage', 'htmlcov', 'build', 'dist')) {
            $full = Join-Path $RepoRoot $path
            if (Test-Path $full) { Remove-Item -Recurse -Force $full }
        }
        Get-ChildItem -Path $RepoRoot -Include '__pycache__' -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notlike '*\.venv\*' } |
            Remove-Item -Recurse -Force
        Write-Host 'cleaned' -ForegroundColor Green
    }

    'all'         { Invoke-Py (@('-m', 'medextract.pipeline') + $Rest) }

    default {
        Write-Host "Unknown target '$Target'." -ForegroundColor Red
        Write-Host "Run '.\make.ps1 help' for the list." -ForegroundColor DarkGray
        exit 2
    }
}
