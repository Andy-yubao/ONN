<#
.SYNOPSIS
  Unified acceptance entry for the BaselineCNN arithmetic smoke stage.

.DESCRIPTION
  Runs, in order:
    1. check_toolchain        (Quartus / Questa toolchain probe)
    2. check_device           (EP4CE10F17C8 recognised by Quartus)
    3. Questa simulation      (requant 20384 + GAP exhaustive/golden vectors
                              + stem_conv_serial 12544 golden + padding专项
                              + maxpool2x2_stream 3136 x2 golden + gap/X-Z)
    4. Quartus smoke compile  (arithmetic baseline_cnn_smoke project)
    5. Quartus stem compile   (full serial stem engine, stem_conv_smoke project)
    6. Python contract test   (arithmetic golden vectors)
    7. Python stem contract   (stem golden vectors / addresses / padding)
    8. Quartus maxpool compile(streaming 2x2 max-pool, maxpool_smoke project)
    9. Python maxpool contract(pool golden / CHW mapping / 3136 recompute)

  Any failing step stops the run with a non-zero exit code.

.PARAMETER ONN_PYTHON
  Optional; the Python interpreter for the contract test (defaults to the
  frozen `onn` conda env, falling back to `python` on PATH).
#>
$ErrorActionPreference = "Continue"

$root     = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$scripts  = Join-Path $root "fpga\baseline_cnn\scripts"

$failures = @()
$totalSteps = 9

function Invoke-Step {
    param([int]$Index, [string]$Name, [scriptblock]$Body)
    Write-Host "`n===== [$Index/$totalSteps] $Name =====" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[run_all] [$Index/$totalSteps] $Name FAILED (exit=$LASTEXITCODE)" -ForegroundColor Red
        $script:failures += $Name
    } else {
        Write-Host "[run_all] [$Index/$totalSteps] $Name OK" -ForegroundColor Green
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

Write-Host "`n=============================================" -ForegroundColor Cyan
if ($failures.Count -eq 0) {
    Write-Host "[run_all] ALL $totalSteps STEPS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host ("[run_all] FAILED STEPS: {0}" -f ($failures -join ", ")) -ForegroundColor Red
    exit 1
}
