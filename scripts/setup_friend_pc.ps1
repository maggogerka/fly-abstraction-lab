[CmdletBinding()]
param(
    [ValidateSet('Guided', 'Menu', 'Check', 'Build', 'DoctorGpu', 'SmokeGpu')]
    [string]$Action = 'Menu'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:Diagnostics = Join-Path $script:RepoRoot 'results\diagnostics'
$script:MinimumDiskGiB = 25
New-Item -ItemType Directory -Force -Path $script:Diagnostics | Out-Null
$script:TranscriptStarted = $false
try {
    Start-Transcript -Path (Join-Path $script:Diagnostics 'friend-setup.log') -Append | Out-Null
    $script:TranscriptStarted = $true
} catch {
    Write-Warning "Не удалось открыть общий лог setup: $($_.Exception.Message)"
}

function Write-Step([string]$Message) {
    Write-Host "`n== $Message ==" -ForegroundColor Cyan
}

function Read-Exact([string]$Prompt, [string]$Expected) {
    $answer = Read-Host "$Prompt (введите точно: $Expected)"
    return $answer -ceq $Expected
}

function Find-NvidiaSmi {
    $command = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:WINDIR 'System32\nvidia-smi.exe'),
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

function Install-WithWinget([string]$Id, [string]$Name, [string]$Confirmation) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "winget не найден. Установите $Name вручную и повторите запуск."
    }
    if (-not (Read-Exact "Разрешить установку $Name через winget?" $Confirmation)) {
        throw "Установка $Name не подтверждена. Ничего не изменено."
    }
    & winget.exe install --id $Id --exact --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget не смог установить $Name (код $LASTEXITCODE)."
    }
    throw "$Name установлен. Закройте терминал, при необходимости перезагрузите Windows и снова запустите START_HERE.cmd."
}

function Invoke-Logged([string]$Name, [string[]]$Arguments, [string]$LogName) {
    New-Item -ItemType Directory -Force -Path $script:Diagnostics | Out-Null
    $logPath = Join-Path $script:Diagnostics $LogName
    Write-Host "Команда: $Name $($Arguments -join ' ')"
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps native stderr as PowerShell errors. Docker
        # Compose writes normal progress (for example, "Image ... Building") to
        # stderr, so global Stop would abort a healthy build before its exit code.
        $ErrorActionPreference = 'Continue'
        & $Name @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Команда завершилась с кодом $exitCode. Полный лог: $logPath"
    }
    Write-Host "Лог: $logPath" -ForegroundColor Green
}

function Test-FriendPc([switch]$OfferInstall) {
    Write-Step 'Проверка Windows-ПК'
    New-Item -ItemType Directory -Force -Path $script:Diagnostics | Out-Null
    $report = [ordered]@{
        checked_at = (Get-Date).ToString('o')
        repository = $script:RepoRoot
        windows = $false
        x64 = [Environment]::Is64BitOperatingSystem
        disk_free_gib = 0
        nvidia_smi = $false
        gpu = $null
        driver = $null
        vram = $null
        wsl2 = $false
        git = $false
        docker_cli = $false
        docker_compose = $false
        docker_engine = $false
        ready = $false
    }
    $report.windows = $env:OS -eq 'Windows_NT'
    if (-not $report.windows -or -not $report.x64) {
        throw 'Нужна 64-битная Windows 10/11. Текущая система не поддерживается.'
    }

    $driveName = ([IO.Path]::GetPathRoot($script:RepoRoot)).TrimEnd('\').TrimEnd(':')
    $drive = Get-PSDrive -Name $driveName
    $report.disk_free_gib = [math]::Round($drive.Free / 1GB, 2)
    if ($report.disk_free_gib -lt $script:MinimumDiskGiB) {
        Write-Warning "Свободно $($report.disk_free_gib) GB. Рекомендуется не менее $script:MinimumDiskGiB GB для Docker-образа и результатов."
    } else {
        Write-Host "Свободное место: $($report.disk_free_gib) GB" -ForegroundColor Green
    }

    $nvidiaSmi = Find-NvidiaSmi
    if ($nvidiaSmi) {
        # Save the native exit code before invoking another PowerShell command.
        # Windows PowerShell 5.1 can otherwise expose a stale $LASTEXITCODE.
        $gpuOutput = @(& $nvidiaSmi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)
        $gpuExitCode = $LASTEXITCODE
        $gpuLine = $gpuOutput | Select-Object -First 1
        if ($gpuExitCode -ne 0) {
            throw "nvidia-smi найден, но не работает (код $gpuExitCode): $($gpuOutput -join ' ')"
        }
        if (-not $gpuLine) {
            throw "nvidia-smi не вернул сведения о GPU."
        }
        $parts = @($gpuLine -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -lt 3) {
            throw "Неожиданный формат ответа nvidia-smi: $gpuLine"
        }
        $report.nvidia_smi = $true
        $report.gpu = $parts[0]
        $report.driver = if ($parts.Count -gt 1) { $parts[1] } else { 'unknown' }
        $report.vram = if ($parts.Count -gt 2) { $parts[2] } else { 'unknown' }
        Write-Host "GPU: $($report.gpu); драйвер: $($report.driver); VRAM: $($report.vram)" -ForegroundColor Green
        if ($report.gpu -match 'RTX 50\d{2}') {
            Write-Host 'Обнаружена совместимая GeForce RTX 50-series. Точная проверка sm_120 выполняется внутри Docker.' -ForegroundColor Green
        } else {
            Write-Warning 'Название GPU не похоже на GeForce RTX 50-series. Продолжение возможно, но итоговое решение принимает doctor-gpu по CUDA и compute capability.'
        }
        $driverMajor = 0
        [void][int]::TryParse(($report.driver -split '\.')[0], [ref]$driverMajor)
        if ($driverMajor -lt 570) {
            throw "Драйвер NVIDIA $($report.driver) слишком стар для целевого CUDA 12.8 runtime. Установите актуальный Game Ready/Studio Driver вручную с https://www.nvidia.com/Download/index.aspx, перезагрузите Windows и повторите запуск. Драйвер автоматически не устанавливается."
        }
    } else {
        throw 'nvidia-smi не найден. Установите актуальный драйвер NVIDIA для GeForce RTX 5070/50-series вручную с https://www.nvidia.com/Download/index.aspx, перезагрузите Windows и повторите запуск. Скрипт не устанавливает драйвер автоматически.'
    }

    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
        $wslStatus = (& wsl.exe --status 2>&1 | Out-String)
        $wslCode = $LASTEXITCODE
        $wslList = (& wsl.exe --list --verbose 2>&1 | Out-String)
        $report.wsl2 = $wslCode -eq 0 -and (($wslStatus + $wslList) -match '(?m)\b2\b')
    }
    if (-not $report.wsl2) {
        throw 'WSL2 не готов. Откройте PowerShell от администратора, выполните `wsl --install`, перезагрузите Windows и снова запустите START_HERE.cmd.'
    }
    Write-Host 'WSL2 доступен.' -ForegroundColor Green

    $report.git = [bool](Get-Command git.exe -ErrorAction SilentlyContinue)
    if (-not $report.git) {
        if ($OfferInstall) {
            Install-WithWinget 'Git.Git' 'Git' 'INSTALL GIT'
        }
        throw 'Git не найден. Установите Git или запустите проект из уже распакованного ZIP.'
    }
    Write-Host "Git: $(& git.exe --version)" -ForegroundColor Green

    $report.docker_cli = [bool](Get-Command docker.exe -ErrorAction SilentlyContinue)
    if (-not $report.docker_cli) {
        if ($OfferInstall) {
            Install-WithWinget 'Docker.DockerDesktop' 'Docker Desktop' 'INSTALL DOCKER'
        }
        throw 'Docker Desktop не найден. Установите его, включите WSL2 backend и повторите запуск.'
    }
    & docker.exe compose version *> $null
    $report.docker_compose = $LASTEXITCODE -eq 0
    if (-not $report.docker_compose) {
        throw 'Docker Compose v2 недоступен. Обновите Docker Desktop и повторите запуск.'
    }
    & docker.exe info *> $null
    $report.docker_engine = $LASTEXITCODE -eq 0
    if (-not $report.docker_engine) {
        throw 'Docker engine недоступен. Запустите Docker Desktop, дождитесь статуса Engine running; после установки/обновления может понадобиться перезагрузка.'
    }
    $report.ready = $true
    $reportPath = Join-Path $script:Diagnostics 'pc-check.json'
    $report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding utf8
    Write-Host "Docker Desktop, Compose и engine доступны. Отчёт: $reportPath" -ForegroundColor Green
    Write-Host 'Python, PyTorch, CUDA Toolkit и cuDNN на Windows устанавливать НЕ нужно: они находятся внутри закреплённого Docker-образа.' -ForegroundColor Yellow
    return $true
}

function Invoke-Build {
    Write-Step 'Сборка закреплённого GPU Docker-образа'
    Test-FriendPc | Out-Null
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'build', 'research-gpu') 'docker-build.log'
}

function Invoke-DoctorGpu {
    Write-Step 'GPU doctor'
    Test-FriendPc | Out-Null
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', 'doctor-gpu') 'doctor-gpu.log'
}

function Invoke-SmokeGpu {
    Write-Step 'GPU smoke: один forward/backward без optimizer step'
    Test-FriendPc | Out-Null
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', 'smoke-gpu') 'smoke-gpu.log'
}

function Invoke-UciInfo {
    Write-Step 'Сведения об UCI Energy Efficiency (без загрузки)'
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', 'data', 'download', 'uci_energy_efficiency') 'uci-info.log'
}

function Invoke-UciDownload {
    Write-Step 'Подтверждаемая загрузка UCI'
    if (-not (Read-Exact 'Проверены URL, лицензия CC BY 4.0 и свободное место?' 'DOWNLOAD UCI')) {
        throw 'Загрузка UCI не подтверждена. Ничего не скачано.'
    }
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', 'data', 'download', 'uci_energy_efficiency', '--confirm-download') 'uci-download.log'
}

function Invoke-UciPrepare {
    Write-Step 'Подготовка локально загруженного UCI'
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', 'data', 'prepare-uci-energy') 'uci-prepare.log'
}

function Invoke-PilotDryRun {
    Write-Step 'Pilot dry-run: обучения и каталога run не будет'
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', '--profile', 'pilot_gpu', 'train') 'pilot-dry-run.log'
}

function Invoke-PilotTraining {
    Write-Step 'Ручной запуск pilot training'
    if (-not (Read-Exact 'Это запустит реальное обучение. Продолжить?' 'TRAIN PILOT')) {
        throw 'Pilot training не подтверждён. Обучение не запущено.'
    }
    $runId = 'pilot_gpu_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
    Invoke-Logged 'docker.exe' @('compose', '--profile', 'research', 'run', '--rm', 'research-gpu', '--profile', 'pilot_gpu', 'train', '--confirm-train', '--run-id', $runId) "$runId.log"
    Write-Host "Результаты сохранены в results\$runId" -ForegroundColor Green
}

function Invoke-GuidedSetup {
    Test-FriendPc -OfferInstall | Out-Null
    Invoke-Build
    Invoke-DoctorGpu
    Invoke-SmokeGpu
    Write-Host "`nБазовая GPU-проверка завершена. Датасеты не скачивались, обучение не запускалось." -ForegroundColor Green
}

function Show-Menu {
    while ($true) {
        Write-Host @'

Действия выполняются строго отдельно:
  1 — проверить ПК
  2 — собрать Docker-образ
  3 — GPU doctor
  4 — GPU smoke
  5 — показать сведения об UCI (без загрузки)
  6 — скачать UCI (нужно DOWNLOAD UCI)
  7 — подготовить UCI
  8 — pilot dry-run (без обучения)
  9 — pilot training (нужно TRAIN PILOT)
  0 — выход
'@
        $choice = Read-Host 'Выберите действие'
        if ($choice -eq '0') { return }
        try {
            switch ($choice) {
                '1' { Test-FriendPc -OfferInstall | Out-Null }
                '2' { Invoke-Build }
                '3' { Invoke-DoctorGpu }
                '4' { Invoke-SmokeGpu }
                '5' { Invoke-UciInfo }
                '6' { Invoke-UciDownload }
                '7' { Invoke-UciPrepare }
                '8' { Invoke-PilotDryRun }
                '9' { Invoke-PilotTraining }
                default { Write-Warning 'Неизвестный пункт меню.' }
            }
        } catch {
            Write-Host "ОШИБКА: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host 'Текущее действие остановлено; существующие data/results не удалены.' -ForegroundColor Yellow
        }
    }
}

Push-Location $script:RepoRoot
try {
    switch ($Action) {
        'Guided' { Invoke-GuidedSetup; Show-Menu }
        'Menu' { Show-Menu }
        'Check' { Test-FriendPc -OfferInstall | Out-Null }
        'Build' { Invoke-Build }
        'DoctorGpu' { Invoke-DoctorGpu }
        'SmokeGpu' { Invoke-SmokeGpu }
    }
} catch {
    Write-Host "`nОШИБКА: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Выполнение остановлено. Датасеты и результаты не удалялись.' -ForegroundColor Yellow
    exit 1
} finally {
    Pop-Location
    if ($script:TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}
