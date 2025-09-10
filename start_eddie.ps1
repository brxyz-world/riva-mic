# start_eddie.ps1
# Launch Eddie mic + orchestrator with warmup and a live tail of the JSONL log.

$ErrorActionPreference = "Stop"

# ----- Paths -----
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

# ----- Env -----
# Defaults; will be overridden by reachable WSL IP detection below
$defaultServer = "localhost:50051"
$env:RIVA_SPEECH_API_URL = $defaultServer
# If using WSL Riva, resolve WSL IP each run (safe if Ubuntu exists; non-fatal if not)
try { $env:RIVA_SPEECH_API = "$((wsl -d Ubuntu hostname -I) -split '\s+' | Select-Object -First 1):50051" } catch { }
$env:OLLAMA_URL          = "http://127.0.0.1:11434"
$env:EDDIE_DEVICE        = "9"
$env:OLLAMA_MODEL        = "qwen2.5:7b-instruct-q4_K_M"
$env:VOICE_NAME          = "English-US.Male-1"
$env:EDDIE_LOG           = Join-Path $root "eddie_turns.log.jsonl"
$env:ACK_THRESHOLD_MS    = "950"
$env:PTT_LINGER_MS       = "220"
$env:EDDIE_OPEN_WAV      = "0"      # keep OS media player closed for low latency
$env:EDDIE_WAKE          = "eddie"  # default wake word

# Select server: prefer WSL IP if port 50051 is reachable; else localhost
$server = $defaultServer
try {
  $wslIp = (wsl -d Ubuntu hostname -I) -split '\s+' | Select-Object -First 1
  if ($wslIp -and (Test-NetConnection -ComputerName $wslIp -Port 50051 -InformationLevel Quiet)) {
    $server = "$wslIp:50051"
  }
} catch { }
# Hard fallback if anything went wrong
if ([string]::IsNullOrWhiteSpace($server)) { $server = $defaultServer }
# Keep orchestrator and mic consistent
$env:RIVA_SPEECH_API_URL = $server

Write-Host "E D D I E   T A L K S   B A C K"
Write-Host "Root:   $root"
Write-Host "Venv:   $py"
Write-Host "Riva:   $server   Ollama: $($env:OLLAMA_URL)   Model: $($env:OLLAMA_MODEL)"
Write-Host "Voice:  $($env:VOICE_NAME)"
Write-Host ""

# ----- WARMUP OLLAMA -----
try {
  # 0) Ensure model exists; pull if missing (non-fatal if fails)
  try {
    $tags = Invoke-RestMethod -Method Get -Uri "$($env:OLLAMA_URL)/api/tags" -TimeoutSec 5
    $haveModel = $false
    if ($tags -and $tags.models) {
      foreach ($m in $tags.models) { if ($m.name -eq $env:OLLAMA_MODEL) { $haveModel = $true; break } }
    }
    if (-not $haveModel) {
      Write-Host "[*] Pulling model: $($env:OLLAMA_MODEL)"
      $pPull = Start-Process -FilePath "ollama" -ArgumentList @("pull", $env:OLLAMA_MODEL) -NoNewWindow -PassThru
      if (-not $pPull.WaitForExit(300000)) { $pPull.Kill() }
    }
  } catch { Write-Host "[*] Model check skipped: $_" }

  # 1) Ensure server is up (non-fatal if already running)
  # & ollama serve | Out-Null  # (optional: usually you run it separately)

  # 2) Warm the model using a quick run
  $argsList = @("run", $env:OLLAMA_MODEL, "ready")
  $p = Start-Process -FilePath "ollama" -ArgumentList $argsList -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\ollama_warmup.out" -RedirectStandardError "$env:TEMP\ollama_warmup.err"
  if (-not $p.WaitForExit(20000)) { $p.Kill() }

  # 3) Tiny generate to lock caches
  $body = @{ model=$env:OLLAMA_MODEL; prompt="ok"; stream=$false } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri "$($env:OLLAMA_URL)/api/generate" -Body $body -ContentType "application/json" -TimeoutSec 12 | Out-Null
  Write-Host "[*] Ollama warmup complete."
} catch {
  Write-Host "[*] Warmup skipped: $_"
}

# ----- Tail the JSONL log (so you can watch 'spoke:' lines) -----
try {
  Start-Job -Name EddieTail -ScriptBlock {
    Get-Content $using:env:EDDIE_LOG -Wait
  } | Out-Null
} catch { }

Write-Host ""
Write-Host "[*] Starting mic:  hold F24 (or SPACE) to talk; release to hear Eddie."
Write-Host "    Press F23 to toggle ALWAYS-ON (wake word: $($env:EDDIE_WAKE))."
Write-Host "    Press Ctrl+C to exit."
Write-Host ""

# ----- Launch mic client -----
& $py .\riva_streaming_mic.py `
  --server $server `
  --device $env:EDDIE_DEVICE `
  --rate 16000 --lang en-US --punct `
  --always_on --wake $env:EDDIE_WAKE --type_to_cursor
