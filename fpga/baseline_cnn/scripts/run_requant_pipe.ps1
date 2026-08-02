<# Independent local Questa entry for tb_requantize_u8_pipe only. #>
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$questa = if (Test-Path env:QUESTA_BIN) { (Get-Item env:QUESTA_BIN).Value } `
          else { "D:\tools\altera_lite\25.1std\questa_fse\win64" }
if (Test-Path $questa -PathType Leaf) { $bin = Split-Path $questa } else { $bin = $questa }
$vlib = Join-Path $bin "vlib.exe"
$vlog = Join-Path $bin "vlog.exe"
$vsim = Join-Path $bin "vsim.exe"
$workRel = "fpga/baseline_cnn/sim/work_requant_pipe"
$workDir = Join-Path $root "fpga\baseline_cnn\sim\work_requant_pipe"
if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir }
& $vlib $workDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $vlog -work $workRel `
    (Join-Path $root "fpga\baseline_cnn\rtl\requantize_u8.v") `
    (Join-Path $root "fpga\baseline_cnn\rtl\requantize_u8_pipe.v") `
    (Join-Path $root "fpga\baseline_cnn\tb\tb_requantize_u8_pipe.v")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Push-Location $root
try {
    & $vsim -c -work $workRel -do "run -all; quit -f" -l `
        "fpga/baseline_cnn/sim/tb_requantize_u8_pipe.log" tb_requantize_u8_pipe
    exit $LASTEXITCODE
} finally { Pop-Location }
