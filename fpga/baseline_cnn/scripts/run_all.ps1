<#
.SYNOPSIS
  Unified acceptance entry for the BaselineCNN arithmetic smoke stage.

.DESCRIPTION
  Runs, in order:
    1. check_toolchain        (Quartus / Questa toolchain probe)
    2. check_device           (EP4CE10F17C8 recognised by Quartus)
    3. Questa simulation      (original requant 20384 + pipelined requant
                              fixed-latency/gap/reset coverage + GAP vectors
                              + stem_conv_serial 12544 golden + padding专项
                              + maxpool2x2_stream 3136 x3 golden + gap/X-Z
                              + stem_pool1 integration golden, two passes
                              + conv2+pool2 golden, two passes
                              + conv3 golden, two passes + conv3/GAP co-done
                              + streaming GAP 32 x3 passes
                              + FC/argmax 10 x3 passes incl. tie
                              + complete core: digit8 full trace + 10 smoke)
    4. Quartus smoke compile  (arithmetic baseline_cnn_smoke project)
    5. Quartus stem compile   (full serial stem engine, stem_conv_smoke project)
    6. Python contract test   (arithmetic golden vectors)
    7. Python stem contract   (stem golden vectors / addresses / padding)
    8. Quartus maxpool compile(streaming 2x2 max-pool, maxpool_smoke project)
    9. Python maxpool contract(pool golden / CHW mapping / 3136 recompute)
   10. Quartus integration    (stem+pool1, stem_pool1_smoke project)
   11. Quartus conv23 compile (shared conv2/conv3 engine, conv23_smoke project)
   12. Python conv23 contract (conv2/conv3 golden / OIHW+CHW / pool2 recompute)
   13. Quartus full-core comp (complete compute core, baseline_cnn_core_smoke)
   14. Python full-core contr (stage order / GAP+FC recompute / argmax tie /
                               storage limits / ROM gating / node counts)

  Any failing step stops the run with a non-zero exit code.

.PARAMETER ONN_PYTHON
  Optional; the Python interpreter for the contract test (defaults to the
  frozen `onn` conda env, falling back to `python` on PATH).
#>
$ErrorActionPreference = "Continue"

$root     = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$scripts  = Join-Path $root "fpga\baseline_cnn\scripts"

$failures = @()
$totalSteps = 14
$runStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

function Invoke-Step {
    param([int]$Index, [string]$Name, [scriptblock]$Body)
    Write-Host "`n===== [$Index/$totalSteps] $Name =====" -ForegroundColor Cyan
    $stepStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    & $Body
    $stepExit = $LASTEXITCODE
    $stepStopwatch.Stop()
    $stepElapsed = $stepStopwatch.Elapsed.ToString("hh\:mm\:ss\.fff")
    if ($stepExit -ne 0) {
        Write-Host "[run_all] [$Index/$totalSteps] $Name FAILED (exit=$stepExit, elapsed=$stepElapsed)" -ForegroundColor Red
        $script:failures += $Name
        Write-Host ("[run_all] TOTAL ELAPSED={0}" -f $script:runStopwatch.Elapsed.ToString("hh\:mm\:ss\.fff")) -ForegroundColor Red
        exit $stepExit
    } else {
        Write-Host "[run_all] [$Index/$totalSteps] $Name OK (elapsed=$stepElapsed)" -ForegroundColor Green
    }
}

# ---- 1. toolchain probe ----
Invoke-Step 1 "check_toolchain" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "check_toolchain.ps1")
}

# ---- 2. device check ----
Invoke-Step 2 "check_device" {
    $quartus_sh = "D:\tools\altera_lite\25.1std\quartus\bin64\quartus_sh.exe"
    if (Test-Path env:QUARTUS_BIN) {
        $p = (Get-Item env:QUARTUS_BIN).Value
        $quartus_sh = if (Test-Path $p -PathType Leaf) { $p } else { Join-Path $p "quartus_sh.exe" }
    }
    & $quartus_sh -t (Join-Path $scripts "check_device.tcl")
}

# ---- 3. Questa simulation ----
Invoke-Step 3 "questa_sim" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_questa.ps1")
}

# ---- 4. Quartus arithmetic smoke compile ----
Invoke-Step 4 "quartus_smoke" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_quartus_smoke.ps1")
}

# ---- 5. Quartus stem engine smoke compile ----
Invoke-Step 5 "quartus_stem" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_quartus_smoke.ps1") `
        -ProjectName stem_conv_smoke -CreateScript create_stem_conv_project.tcl
}

# ---- 6. Python arithmetic contract test ----
Invoke-Step 6 "python_contract" {
    $py = if (Test-Path env:ONN_PYTHON) { (Get-Item env:ONN_PYTHON).Value }
          elseif (Test-Path "D:\tools\anaconda3\envs\onn\python.exe") { "D:\tools\anaconda3\envs\onn\python.exe" }
          else { "python" }
    & $py -m pytest model\tests\test_rtl_vector_contract.py -q
}

# ---- 7. Python stem contract test ----
Invoke-Step 7 "python_stem_contract" {
    $py = if (Test-Path env:ONN_PYTHON) { (Get-Item env:ONN_PYTHON).Value }
          elseif (Test-Path "D:\tools\anaconda3\envs\onn\python.exe") { "D:\tools\anaconda3\envs\onn\python.exe" }
          else { "python" }
    & $py -m pytest model\tests\test_stem_rtl_contract.py -q
}

# ---- 8. Quartus maxpool smoke compile ----
Invoke-Step 8 "quartus_maxpool" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_quartus_smoke.ps1") `
        -ProjectName maxpool_smoke -CreateScript create_maxpool_project.tcl
}

# ---- 9. Python maxpool contract test ----
Invoke-Step 9 "python_maxpool_contract" {
    $py = if (Test-Path env:ONN_PYTHON) { (Get-Item env:ONN_PYTHON).Value }
          elseif (Test-Path "D:\tools\anaconda3\envs\onn\python.exe") { "D:\tools\anaconda3\envs\onn\python.exe" }
          else { "python" }
    & $py -m pytest model\tests\test_maxpool_rtl_contract.py -q
}

# ---- 10. Quartus integration compile (stem + pool1) ----
Invoke-Step 10 "quartus_stem_pool1" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_quartus_smoke.ps1") `
        -ProjectName stem_pool1_smoke -CreateScript create_stem_pool1_project.tcl
}

# ---- 11. Quartus shared conv2/conv3 engine compile ----
Invoke-Step 11 "quartus_conv23" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_quartus_smoke.ps1") `
        -ProjectName conv23_smoke -CreateScript create_conv23_project.tcl
}

# ---- 12. Python conv23 contract test ----
Invoke-Step 12 "python_conv23_contract" {
    $py = if (Test-Path env:ONN_PYTHON) { (Get-Item env:ONN_PYTHON).Value }
          elseif (Test-Path "D:\tools\anaconda3\envs\onn\python.exe") { "D:\tools\anaconda3\envs\onn\python.exe" }
          else { "python" }
    & $py -m pytest model\tests\test_conv23_rtl_contract.py -q
}

# ---- 13. Quartus complete-core compile ----
Invoke-Step 13 "quartus_full_core" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "run_quartus_smoke.ps1") `
        -ProjectName baseline_cnn_core_smoke -CreateScript create_full_core_project.tcl
}

# ---- 14. Python full-core contract test ----
Invoke-Step 14 "python_full_core_contract" {
    $py = if (Test-Path env:ONN_PYTHON) { (Get-Item env:ONN_PYTHON).Value }
          elseif (Test-Path "D:\tools\anaconda3\envs\onn\python.exe") { "D:\tools\anaconda3\envs\onn\python.exe" }
          else { "python" }
    & $py -m pytest model\tests\test_full_core_rtl_contract.py -q
}

Write-Host "`n=============================================" -ForegroundColor Cyan
$runStopwatch.Stop()
Write-Host ("[run_all] TOTAL ELAPSED={0}" -f $runStopwatch.Elapsed.ToString("hh\:mm\:ss\.fff")) -ForegroundColor Cyan
if ($failures.Count -eq 0) {
    Write-Host "[run_all] ALL $totalSteps STEPS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host ("[run_all] FAILED STEPS: {0}" -f ($failures -join ", ")) -ForegroundColor Red
    exit 1
}
