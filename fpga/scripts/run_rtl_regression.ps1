$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "D:\tools\anaconda3\envs\onn\python.exe"
$simDir = Join-Path $root "fpga\.sim"
$exportDir = Join-Path $root "model\snn\export"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Required onn environment was not found at $python"
}

& $python -m pytest -q -p no:cacheprovider model/snn/test_integer_reference.py model/snn/test_export_params.py fpga/basys3/test_protocol.py experiments/common/test_device_latency_encoder.py
if ($LASTEXITCODE -ne 0) { throw "Python unit tests failed" }

& $python -m fpga.scripts.generate_phase1_vectors
if ($LASTEXITCODE -ne 0) { throw "Golden-vector generation failed" }
& $python -m fpga.scripts.generate_regression_vectors
if ($LASTEXITCODE -ne 0) { throw "Compact regression-vector generation failed" }
& $python -m fpga.scripts.generate_uart_vectors
if ($LASTEXITCODE -ne 0) { throw "UART-vector generation failed" }

$vivadoBin = "D:\tools\AMDDesignTools\2026.1\Vivado\bin"
$xvlog = Join-Path $vivadoBin "xvlog.bat"
$xelab = Join-Path $vivadoBin "xelab.bat"
$xsim = Join-Path $vivadoBin "xsim.bat"
foreach ($tool in @($xvlog, $xelab, $xsim)) {
    if (-not (Test-Path -LiteralPath $tool)) { throw "Vivado Simulator tool not found: $tool" }
}
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
    (Join-Path $root "fpga\tb\tb_snn_core.sv"),
    (Join-Path $root "fpga\tb\tb_snn_core_regression.sv"),
    (Join-Path $root "fpga\basys3\rtl\onn_uart_rx.sv"),
    (Join-Path $root "fpga\basys3\rtl\onn_uart_tx.sv"),
    (Join-Path $root "fpga\basys3\rtl\onn_basys3_top.sv"),
    (Join-Path $root "fpga\tb\tb_basys3_uart.sv"),
    (Join-Path $root "fpga\tb\tb_basys3_debug.sv")
)
$tests = @(
    @{ Top = "tb_if_unit"; Snapshot = "rtl_if" },
    @{ Top = "tb_conv1_if1"; Snapshot = "rtl_conv1_if1" },
    @{ Top = "tb_conv2_if2"; Snapshot = "rtl_conv2_if2" },
    @{ Top = "tb_readout"; Snapshot = "rtl_readout" },
    @{ Top = "tb_snn_core"; Snapshot = "rtl_snn_core" },
    @{ Top = "tb_snn_core_regression"; Snapshot = "rtl_snn_core_regression" },
    @{ Top = "tb_basys3_uart"; Snapshot = "rtl_basys3_uart" },
    @{ Top = "tb_basys3_debug"; Snapshot = "rtl_basys3_debug" }
)

Push-Location $simDir
try {
    & $xvlog --sv --include $exportDir @sources --nolog
    if ($LASTEXITCODE -ne 0) { throw "SystemVerilog compilation failed" }
    foreach ($test in $tests) {
        & $xelab $test.Top -s $test.Snapshot --nolog
        if ($LASTEXITCODE -ne 0) { throw "$($test.Top) elaboration failed" }
        $simLines = & $xsim $test.Snapshot --runall --nolog 2>&1
        $simLines | Write-Host
        if ($LASTEXITCODE -ne 0 -or ($simLines -join "`n") -match '(?m)^(Fatal:|ERROR:|FATAL_ERROR:)') {
            throw "$($test.Top) simulation failed"
        }
    }
} finally {
    Pop-Location
}

Write-Host "PASS: complete RTL golden-vector regression finished"
