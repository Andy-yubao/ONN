$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "D:\tools\anaconda3\envs\onn\python.exe"
$simDir = Join-Path $root "fpga\.sim"
$exportDir = Join-Path $root "model\snn\export"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Required onn environment was not found at $python"
}

& $python -m fpga.scripts.generate_phase1_vectors
if ($LASTEXITCODE -ne 0) { throw "Golden-vector generation failed" }

$xvlog = (Get-Command xvlog -ErrorAction Stop).Source
$xelab = (Get-Command xelab -ErrorAction Stop).Source
$xsim = (Get-Command xsim -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path $simDir | Out-Null

$sources = @(
    (Join-Path $root "fpga\rtl\onn_snn_params_pkg.sv"),
    (Join-Path $root "fpga\rtl\onn_signed_saturate.sv"),
    (Join-Path $root "fpga\rtl\onn_if_unit.sv"),
    (Join-Path $root "fpga\rtl\onn_conv1_if1.sv"),
    (Join-Path $root "fpga\tb\tb_if_unit.sv"),
    (Join-Path $root "fpga\tb\tb_conv1_if1.sv")
)

Push-Location $simDir
try {
    & $xvlog --sv --include $exportDir @sources --nolog
    if ($LASTEXITCODE -ne 0) { throw "SystemVerilog compilation failed" }

    & $xelab tb_if_unit -s phase1_if --nolog
    if ($LASTEXITCODE -ne 0) { throw "IF testbench elaboration failed" }
    & $xsim phase1_if --runall --nolog
    if ($LASTEXITCODE -ne 0) { throw "IF testbench failed" }

    & $xelab tb_conv1_if1 -s phase1_conv1_if1 --nolog
    if ($LASTEXITCODE -ne 0) { throw "Conv1+IF1 testbench elaboration failed" }
    & $xsim phase1_conv1_if1 --runall --nolog
    if ($LASTEXITCODE -ne 0) { throw "Conv1+IF1 testbench failed" }
} finally {
    Pop-Location
}

Write-Host "PASS: Phase 1 golden generation and RTL comparisons completed"
