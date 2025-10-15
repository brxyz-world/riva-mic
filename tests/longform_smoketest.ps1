param(
  [string] $Prompt = "",
  # --- 1R-hotfix-5a-alt (longform-only env) ---
  [switch] $Internal = $false,         # do not use directly
  [switch] $NoEnv    = $false,         # skip env setup & run as-is
  [string] $OllamaHost   = "http://127.0.0.1:11434",
  [string] $KeepAlive    = "30m",
  [int]    $AckTimeoutSec = 30,
  [int]    $LlmTimeoutSec = 180,
  [int]    $SampleRate    = 22050,
  [string] $Sentinel      = "<END>",
  [int]    $TtsMaxChars   = 0          # 0 = disabled
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Driver branch: relaunch this script in a *child* PS with scoped env vars,
# so your current shell / short-form flow stays clean.
if (-not $Internal -and -not $NoEnv) {
  # find a PowerShell host (pwsh preferred, fall back to Windows PowerShell)
  $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
  $psCmd   = Get-Command powershell -ErrorAction SilentlyContinue

  $psExe = $null
  if ($pwshCmd) {
    $psExe = $pwshCmd.Path
  } elseif ($psCmd) {
    $psExe = $psCmd.Path
  } else {
    $candidates = @(
      (Join-Path $Env:ProgramFiles 'PowerShell\7\pwsh.exe'),
      (Join-Path $Env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'),
      (Join-Path $PSHOME 'pwsh.exe'),
      (Join-Path $PSHOME 'powershell.exe')
    ) | Select-Object -Unique
    foreach ($c in $candidates) { if (Test-Path $c) { $psExe = $c; break } }
  }

  if (-not $psExe) { throw 'Could not locate PowerShell executable (pwsh/powershell).' }

  $escapedPrompt = $Prompt.Replace('"','`"')
  $cmd = @"
`$ErrorActionPreference='Stop';
`$env:PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning:google.protobuf.runtime_version';
`$env:RIVA_TTS_SAMPLE_RATE='$SampleRate';
`$env:OLLAMA_HOST='$OllamaHost';
`$env:OLLAMA_KEEP_ALIVE='$KeepAlive';
`$env:EDDIE_ACK_TIMEOUT_S='$AckTimeoutSec';
`$env:EDDIE_OLLAMA_TIMEOUT_S='$LlmTimeoutSec';
`$env:EDDIE_LONGFORM_STOP_SENTINEL='$Sentinel';
"@
  if ($TtsMaxChars -gt 0) { $cmd += "`$env:EDDIE_TTS_MAX_CHARS='$TtsMaxChars';`n" }
  $cmd += "Write-Host '[@longform_env] host=$OllamaHost keepAlive=$KeepAlive ack=${AckTimeoutSec}s llm=${LlmTimeoutSec}s sr=$SampleRate sentinel=""$Sentinel"" ttsMax=$TtsMaxChars' -ForegroundColor Cyan;`n"
  $cmd += ". `"$PSCommandPath`" -Internal"
  if ($PSBoundParameters.ContainsKey('Prompt') -and $Prompt) { $cmd += " -Prompt `"$escapedPrompt`"" }

  & $psExe -NoProfile -ExecutionPolicy Bypass -Command $cmd
  exit $LASTEXITCODE
}

if (-not $Prompt) {
  $Prompt = "Tell me a detailed, multi-paragraph story about the first human expedition across Jupiter's moons."
}

Write-Host "Warmup..." -ForegroundColor Cyan

# Warmup: short, non-longform ping
$env:LLM_WARMUP                  = "1"
$env:LLM_WARMUP_TIMEOUT_MS       = "30000"
$env:OLLAMA_KEEP_ALIVE           = "30m"

$env:LONG_FORM                   = "0"
$env:NUM_PREDICT                 = "12"
$env:TEMPERATURE                 = "0.0"
$env:LONGFORM_AUTOCONFIRM        = "0"
Remove-Item Env:STOP_TOKENS -ErrorAction SilentlyContinue

python .\Eddie\30_SYSTEMS\orchestrator\eddie_orchestrator.py --text "ping"

Write-Host "Longform turn..." -ForegroundColor Cyan

# Longform request + autoconfirm with more time
$env:LONG_FORM                          = "1"
$env:NUM_PREDICT                        = "512"
$env:LONGFORM_MIN_CHARS                 = "900"
$env:LONGFORM_TEMPERATURE               = "0.6"
$env:LONGFORM_AUTOCONFIRM               = "1"
$env:LONGFORM_AUTOCONFIRM_NUM_PREDICT   = "256"
$env:LONGFORM_AUTOCONFIRM_TIMEOUT_MS    = "60000"
$env:LLM_TIMEOUT_MS                     = "65000"
Remove-Item Env:STOP_TOKENS -ErrorAction SilentlyContinue

python .\Eddie\30_SYSTEMS\orchestrator\eddie_orchestrator.py --text $Prompt

# Show the last two log lines (warmup + longform)
Get-ChildItem .\logs\*Eddie_Convo_*.jsonl |
  Sort-Object LastWriteTime |
  Select-Object -Last 1 |
  Get-Content |
  Select-Object -Last 2
