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
    (Join-Path $root "fpga\rtl\onn_conv2_if2.sv"),
    (Join-Path $root "fpga\rtl\onn_readout.sv"),
    (Join-Path $root "fpga\rtl\onn_snn_core.sv"),
    (Join-Path $root "fpga\rtl\onn_sparse_conv1_if1.sv"),
    (Join-Path $root "fpga\rtl\onn_sparse_conv2_if2.sv"),
    (Join-Path $root "fpga\rtl\onn_sparse_readout.sv"),
    (Join-Path $root "fpga\rtl\onn_snn_core_sparse.sv"),
    (Join-Path $root "fpga\tb\tb_if_unit.sv"),
    (Join-Path $root "fpga\tb\tb_conv1_if1.sv"),
    (Join-Path $root "fpga\tb\tb_conv2_if2.sv"),
    (Join-Path $root "fpga\tb\tb_readout.sv"),
    (Join-Path $root "fpga\tb\tb_snn_core.sv")
)
$tests = @(
    @{ Top = "tb_if_unit"; Snapshot = "rtl_if" },
    @{ Top = "tb_conv1_if1"; Snapshot = "rtl_conv1_if1" },
    @{ Top = "tb_conv2_if2"; Snapshot = "rtl_conv2_if2" },
    @{ Top = "tb_readout"; Snapshot = "rtl_readout" },
    @{ Top = "tb_snn_core"; Snapshot = "rtl_snn_core" }
)

Push-Location $simDir
try {
    & $xvlog --sv --include $exportDir @sources --nolog
    if ($LASTEXITCODE -ne 0) { throw "SystemVerilog compilation failed" }
    foreach ($test in $tests) {
        & $xelab $test.Top -s $test.Snapshot --nolog
        if ($LASTEXITCODE -ne 0) { throw "$($test.Top) elaboration failed" }
        & $xsim $test.Snapshot --runall --nolog
        if ($LASTEXITCODE -ne 0) { throw "$($test.Top) simulation failed" }
    }
} finally {
    Pop-Location
}

Write-Host "PASS: complete RTL golden-vector regression finished"
