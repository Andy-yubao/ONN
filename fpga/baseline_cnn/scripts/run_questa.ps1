<#
.SYNOPSIS
  Questa golden-vector verification for the arithmetic blocks and the serial
  stem convolution engine.

.DESCRIPTION
  1. vlib  -> fpga/baseline_cnn/sim/work (gitignored)
  2. vlog  -> rtl/{requantize_u8,gap_div49,arithmetic_smoke_top,
            sync_ram_u8,sync_rom_s8,sync_rom_s32,stem_conv_serial}.v + tb/*.v
  3. vsim  -> tb_requantize_u8     (20384 golden comparisons)
  4. vsim  -> tb_gap_div49         (12496 exhaustive + 32-channel golden)
  5. vsim  -> tb_stem_conv_serial  (digit8 golden: conv1_acc/stem_q stream and
                                    output-RAM readback, 12544 x3 comparisons)
  6. vsim  -> tb_stem_conv_padding (padding专项: tap counts / illegal addrs)

  Any compile error, simulation fatal or mismatch sets a non-zero exit code.
  $readmemh paths inside the testbenches are relative to the repo root, so
  the vsim working directory is the repo root.

.PARAMETER QUESTA_BIN
  Optional; overrides the default Questa bin directory (or points straight
  at the .exe). Falls back to the frozen install path if unset.
#>
$ErrorActionPreference = "Continue"

$DEFAULT_QUESTA_BIN = "D:\tools\altera_lite\25.1std\questa_fse\win64"

function Resolve-Tool {
    param([string]$EnvVar, [string]$DefaultDir, [string]$Exe)
    if (Test-Path env:$EnvVar) {
        $p = (Get-Item env:$EnvVar).Value
        $candidate = if (Test-Path $p -PathType Leaf) { $p } else { Join-Path $p $Exe }
        if (Test-Path $candidate) { return $candidate }
        Write-Host "[run_questa] ERROR: $Exe not found via $EnvVar ($p)" -ForegroundColor Red
        return $null
    }
    $candidate = Join-Path $DefaultDir $Exe
    if (Test-Path $candidate) { return $candidate }
    Write-Host "[run_questa] ERROR: $Exe not found at $candidate" -ForegroundColor Red
    return $null
}

$vlib = Resolve-Tool "QUESTA_BIN" $DEFAULT_QUESTA_BIN "vlib.exe"
$vlog = Resolve-Tool "QUESTA_BIN" $DEFAULT_QUESTA_BIN "vlog.exe"
$vsim = Resolve-Tool "QUESTA_BIN" $DEFAULT_QUESTA_BIN "vsim.exe"
if (-not $vlib -or -not $vlog -or -not $vsim) { exit 1 }

$root     = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$workRel  = "fpga/baseline_cnn/sim/work"        # forward slashes: Questa vsim/vlog
                                                # cannot take an absolute backslash
                                                # path for -work (vopt merges it
                                                # with the design unit).
$workDir  = Join-Path $root "fpga\baseline_cnn\sim\work"   # absolute, for vlib only
$rtlDir   = Join-Path $root "fpga\baseline_cnn\rtl"
$tbDir    = Join-Path $root "fpga\baseline_cnn\tb"
$vecDir   = "sim/vectors/golden_trace"          # relative to repo root (vsim cwd)
$logDir   = Join-Path $root "fpga\baseline_cnn\sim"

# ---- recreate the work library ----
if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir }
Write-Host "[run_questa] vlib $workDir" -ForegroundColor Cyan
& $vlib $workDir
if ($LASTEXITCODE -ne 0) { Write-Host "[run_questa] vlib failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1 }

# ---- compile ----
$srcs = @(
    (Join-Path $rtlDir "requantize_u8.v"),
    (Join-Path $rtlDir "gap_div49.v"),
    (Join-Path $rtlDir "arithmetic_smoke_top.v"),
    (Join-Path $rtlDir "sync_ram_u8.v"),
    (Join-Path $rtlDir "sync_rom_s8.v"),
    (Join-Path $rtlDir "sync_rom_s32.v"),
    (Join-Path $rtlDir "stem_conv_serial.v"),
    (Join-Path $tbDir  "tb_requantize_u8.v"),
    (Join-Path $tbDir  "tb_gap_div49.v"),
    (Join-Path $tbDir  "tb_stem_conv_serial.v"),
    (Join-Path $tbDir  "tb_stem_conv_padding.v")
)
Write-Host "[run_questa] vlog -work $workRel" -ForegroundColor Cyan
& $vlog -work $workRel $srcs 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "[run_questa] vlog failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1 }

# ---- simulate ----
Push-Location $root
$allOk = $true
try {
    foreach ($tb in @("tb_requantize_u8", "tb_gap_div49", "tb_stem_conv_serial", "tb_stem_conv_padding")) {
        $logFile = Join-Path $logDir "$tb.log"
        Write-Host "[run_questa] vsim -c -work $workRel $tb (log: $logFile)" -ForegroundColor Cyan
        & $vsim -c -work $workRel -do "run -all; quit -f" -l $logFile "$tb" 2>&1
        $code = $LASTEXITCODE
        # Also scan the transcript for our pass/fail markers (belt and braces).
        $tail = Get-Content $logFile -ErrorAction SilentlyContinue
        $passed  = ($tail | Select-String -Pattern "ALL_PASS" -Quiet)
        if ($code -ne 0 -or -not $passed) {
            Write-Host "[run_questa] $tb FAILED (exit=$code)" -ForegroundColor Red
            $allOk = $false
        } else {
            Write-Host "[run_questa] $tb OK" -ForegroundColor Green
        }
    }
}
finally {
    Pop-Location
}

if (-not $allOk) { Write-Host "[run_questa] FAILURE" -ForegroundColor Red; exit 1 }
Write-Host "[run_questa] ALL_PASS" -ForegroundColor Green
exit 0
