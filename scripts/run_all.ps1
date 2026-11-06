# Runs the whole experiment from one config: train both models, score all three on the test set, compare.
param([string]$Config = "configs/default.yaml", [string]$RunName = "default")

Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$env:TQDM_DISABLE = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

# Stops Windows sleeping from idle while this script runs, without changing any power settings.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
# 2147483649 is ES_CONTINUOUS | ES_SYSTEM_REQUIRED.
[void][Win32.Power]::SetThreadExecutionState([uint32]2147483649)

$python = ".\.venv\Scripts\python.exe"
$logs = "logs\$RunName"
New-Item -ItemType Directory -Force $logs | Out-Null
$status = "$logs\status.txt"
Remove-Item "$logs\done.txt" -ErrorAction SilentlyContinue
Set-Content $status "started $(Get-Date -Format s)"

function Step([string]$Name, [string[]]$Arguments) {
    Add-Content $status "$Name start $(Get-Date -Format s)"
    # cmd handles the redirect because PowerShell 5.1 treats a native program's stderr as an error.
    cmd /c "$python $($Arguments -join ' ') > $logs\$Name.log 2>&1"
    $code = $LASTEXITCODE
    Add-Content $status "$Name exit=$code $(Get-Date -Format s)"
    return $code -eq 0
}

$ok = @(
    Step "train_grpo" @("-m", "src.train_grpo", "--config", $Config)
    Step "train_sft" @("-m", "src.train_sft", "--config", $Config)
    Step "eval_base" @("-m", "src.eval", "--config", $Config, "--name", "base")
    Step "eval_sft" @("-m", "src.eval", "--config", $Config, "--name", "sft")
    Step "eval_grpo" @("-m", "src.eval", "--config", $Config, "--name", "grpo")
)
if ($ok -notcontains $false) {
    [void](Step "compare" @("-m", "src.eval", "--config", $Config, "--compare"))
    [void](Step "plots" @("-m", "src.plots", "--config", $Config))
}
Set-Content "$logs\done.txt" "finished $(Get-Date -Format s)"
