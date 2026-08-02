<# Independent local Questa entry for the three conv retiming regressions. #>
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$questa = if (Test-Path env:QUESTA_BIN) { (Get-Item env:QUESTA_BIN).Value } `
          else { "D:\tools\altera_lite\25.1std\questa_fse\win64" }
if (Test-Path $questa -PathType Leaf) { $bin = Split-Path $questa } else { $bin = $questa }
$vlib = Join-Path $bin "vlib.exe"; $vlog = Join-Path $bin "vlog.exe"; $vsim = Join-Path $bin "vsim.exe"
$workRel = "fpga/baseline_cnn/sim/work_conv_retiming"
$workDir = Join-Path $root "fpga\baseline_cnn\sim\work_conv_retiming"
if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir }
& $vlib $workDir; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$rtl = Join-Path $root "fpga\baseline_cnn\rtl"; $tb = Join-Path $root "fpga\baseline_cnn\tb"
$srcs = @(
    (Join-Path $rtl "requantize_u8_pipe.v"), (Join-Path $rtl "sync_ram_u8.v"),
    (Join-Path $rtl "sync_rom_s8.v"), (Join-Path $rtl "sync_rom_s32.v"),
    (Join-Path $rtl "conv_u8_serial.v"), (Join-Path $rtl "maxpool2x2_stream.v"),
    (Join-Path $rtl "gap_div49.v"), (Join-Path $rtl "gap_stream_u8.v"),
    (Join-Path $tb "tb_conv2_pool2.v"), (Join-Path $tb "tb_conv3_serial.v"),
    (Join-Path $tb "tb_conv3_gap.v")
)
& $vlog -work $workRel $srcs; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Push-Location $root
try {
    foreach ($name in @("tb_conv2_pool2", "tb_conv3_serial", "tb_conv3_gap")) {
        & $vsim -c -work $workRel -do "run -all; quit -f" -l "fpga/baseline_cnn/sim/$name.log" $name
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} finally { Pop-Location }
