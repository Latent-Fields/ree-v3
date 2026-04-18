# Post-gaming recovery for EWIN-PC.
# Run from the ree-v3 directory after a gaming session ends.
#
# What this does:
#   1. Finds the experiment_runner.py process (if any).
#   2. Asks you to Ctrl-C its window for graceful drain, OR offers a force-kill.
#   3. Commits any uncommitted runner_status/EWIN-PC.json entries to REE_assembly.
#      (Recovers 406a/429a/430a outcomes if they were ever written locally.)
#   4. Pulls latest ree-v3 (picks up the git_push_status() fix, commit f37cfbd).
#   5. Pulls latest REE_assembly.
#   6. Relaunches the runner via start_runner_EWIN.bat.
#
# Usage (from PowerShell in ree-v3 directory):
#   .\post_gaming_recover_EWIN.ps1

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "[info] $msg" -ForegroundColor Gray }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-OK($msg)   { Write-Host "[ok]   $msg" -ForegroundColor Green }

# --- Sanity check: must be in ree-v3 ---
if (-not (Test-Path "experiment_runner.py")) {
    Write-Warn "experiment_runner.py not found. Run this script from the ree-v3 directory."
    exit 1
}

$ReeV3     = Get-Location
$Assembly  = Resolve-Path "..\REE_assembly" -ErrorAction SilentlyContinue
if (-not $Assembly) {
    Write-Warn "Could not find REE_assembly next to ree-v3. Check your folder layout."
    exit 1
}

# --- Step 1: find runner process ---
Write-Step "Step 1: Check for running experiment_runner"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
         Where-Object { $_.CommandLine -like "*experiment_runner.py*" }

if ($procs) {
    foreach ($p in $procs) {
        Write-Info "Found runner PID $($p.ProcessId): $($p.CommandLine)"
    }
    Write-Host ""
    Write-Host "RECOMMENDED: focus the runner's console window and press Ctrl-C ONCE."
    Write-Host "The runner will finish its current experiment, push results, and exit."
    Write-Host ""
    $answer = Read-Host "Options: [W]ait (I'll press Ctrl-C), [F]orce-kill, [S]kip"
    switch ($answer.ToUpper()) {
        "W" {
            Write-Info "Waiting for runner to exit (check every 15s)..."
            while ($true) {
                Start-Sleep -Seconds 15
                $still = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
                         Where-Object { $_.CommandLine -like "*experiment_runner.py*" }
                if (-not $still) {
                    Write-OK "Runner exited cleanly."
                    break
                }
                Write-Info "...still running"
            }
        }
        "F" {
            Write-Warn "Force-killing runner (current experiment will be lost)."
            foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force }
            Start-Sleep -Seconds 3
        }
        default {
            Write-Info "Skipping stop. Moving on (new runner will fail to start if old one holds the queue)."
        }
    }
} else {
    Write-OK "No runner process found."
}

# --- Step 2: recover local-uncommitted status entries ---
Write-Step "Step 2: Recover any local status entries in REE_assembly"
Push-Location $Assembly

$dirty = git status --porcelain runner_status/EWIN-PC.json evidence/experiments/runner_status/EWIN-PC.json 2>$null
if ($dirty) {
    Write-Info "Uncommitted EWIN-PC status detected:"
    Write-Host $dirty
    git add evidence/experiments/runner_status/EWIN-PC.json
    git commit -m "runner_status: EWIN-PC post-gaming recovery (includes missing 406a/429a/430a if present)"
    for ($i = 1; $i -le 3; $i++) {
        try {
            git pull --rebase origin master
            git push origin HEAD:master
            Write-OK "REE_assembly pushed (attempt $i)."
            break
        } catch {
            Write-Warn "Push attempt $i failed, retrying..."
            Start-Sleep -Seconds 5
        }
    }
} else {
    Write-Info "No uncommitted EWIN-PC status file. 406a/429a/430a outcomes are gone (runner bug)."
}

# Also flush any other uncommitted evidence (experiments that wrote flat JSON but
# didn't get committed before gaming started)
$otherDirty = git status --porcelain evidence/experiments 2>$null | Where-Object { $_ -notmatch "runner_status" }
if ($otherDirty) {
    Write-Info "Other uncommitted evidence files detected, flushing..."
    git add evidence/experiments
    git commit -m "evidence: flush pending EWIN-PC evidence post-gaming"
    git pull --rebase origin master
    git push origin HEAD:master
}

Pop-Location

# --- Step 3: pull latest ree-v3 (picks up the git_push_status fix) ---
Write-Step "Step 3: Pull latest ree-v3"
git pull --rebase origin main
$head = git rev-parse HEAD
Write-OK "ree-v3 at $head"

# --- Step 4: pull latest REE_assembly ---
Write-Step "Step 4: Pull latest REE_assembly"
Push-Location $Assembly
git pull --rebase origin master
Pop-Location

# --- Step 5: validate queue before restart ---
Write-Step "Step 5: Validate queue"
python validate_queue.py
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Queue validation failed. Fix experiment_queue.json before restarting runner."
    exit 1
}
Write-OK "Queue valid."

# --- Step 6: relaunch runner in a new window ---
Write-Step "Step 6: Launch runner"
Write-Info "Starting runner in a new console window via start_runner_EWIN.bat..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "start_runner_EWIN.bat"
Write-OK "Runner launched. Check the new console window for progress."
Write-Host ""
Write-Host "Recovery complete. Leave the runner console open while experiments run." -ForegroundColor Green
