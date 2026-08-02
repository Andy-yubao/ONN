<#
.SYNOPSIS
  Independent build+verify entry for the AC620 fixed-digit BaselineCNN board
  self-test (Questa end-to-end self-test + Quartus full board compile).

.DESCRIPTION
  Part 1 - Questa end-to-end self-test of tb_ac620_cnn_selftest:
      real ac620_cnn_selftest_top -> baseline_cnn_core -> every sub-block,
      PASS instance (EXPECTED_PRED=8) + FAIL instance (EXPECTED_PRED=7),
      full 1,590,315-cycle digit-8 inference, all board-loader assertions,
      prints AC620_CNN_SELFTEST_PASS.

  Part 2 - Quartus full board compile of ac620_cnn_selftest:
      1. check_device.tcl                       -> verify EP4CE10F17C8
      2. create_ac620_cnn_selftest_project.tcl  -> write .qpf/.qsf/.sdc under
                                                   quartus/ (REAL pins:
                                                   clk_50m=E1, led[0..3]=
                                                   A2/B3/A4/A3, 3.3-V LVTTL,
                                                   SDC_FILE referenced, NO
                                                   virtual pins)
      3. quartus_sh --flow compile ac620_cnn_selftest
      4. Assertions:
           Flow Successful | device == EP4CE10F17C8 | physical pins == 5 |
           virtual pins == 0 | SDC read | no "Timing requirements not
           specified" | no latch | no multi-driver | no illegal pin | no
           missing ROM init | no RAM/ROM depth overflow | CNN hierarchy
           preserved | SOF generated
      5. Report resources (LE / registers / M9K / block memory bits /
         9-bit multipliers / PLLs), every timing corner's setup/hold slack,
         Fmax, worst warning, the Fitter RAM Summary, and the .sof path.

  This is a REAL physical board project: it is NOT part of run_all.ps1's
  normal RTL regression (which keeps 14 steps and never runs this).  Run it
  standalone when you want to rebuild/verify the AC620 self-test bitstream.

.PARAMETER QUARTUS_BIN
  Optional; overrides the default Quartus bin directory (or points straight
  at the .exe). Falls back to the frozen install path if unset.

.PARAMETER QUESTA_BIN
  Optional; overrides the default Questa bin directory (or points straight
  at the .exe). Falls back to the frozen install path if unset.

.PARAMETER SkipQuesta
  Optional; set to $true to run only the Quartus part.

.PARAMETER ValidateExistingReportsOnly
  Read and validate the existing Quartus reports/SOF without invoking Questa,
  project creation, Quartus compilation, or FPGA programming.  This mode is
  intended for deterministic guard regression tests and engineering review.

.PARAMETER ExistingReportDir
  Optional report directory override for ValidateExistingReportsOnly.  It is
  rejected in normal build mode and exists solely for read-only validation of
  temporary report copies.
#>
param(
    [string]$ProjectName = "ac620_cnn_selftest",
    [switch]$SkipQuesta,    # pass -SkipQuesta to run only the Quartus part
    [switch]$SkipQuartus,   # pass -SkipQuartus to run only the Questa part
    [switch]$ValidateExistingReportsOnly,
    [string]$ExistingReportDir = ""
)
$ErrorActionPreference = "Continue"

$DEFAULT_QUARTUS_BIN = "D:\tools\altera_lite\25.1std\quartus\bin64"
$DEFAULT_QUESTA_BIN  = "D:\tools\altera_lite\25.1std\questa_fse\win64"

function Resolve-Tool {
    param([string]$EnvVar, [string]$DefaultDir, [string]$Exe)
    if (Test-Path env:$EnvVar) {
        $p = (Get-Item env:$EnvVar).Value
        $candidate = if (Test-Path $p -PathType Leaf) { $p } else { Join-Path $p $Exe }
        if (Test-Path $candidate) { return $candidate }
        Write-Host "[ac620_selftest] ERROR: $Exe not found via $EnvVar ($p)" -ForegroundColor Red
        return $null
    }
    $candidate = Join-Path $DefaultDir $Exe
    if (Test-Path $candidate) { return $candidate }
    Write-Host "[ac620_selftest] ERROR: $Exe not found at $candidate" -ForegroundColor Red
    return $null
}

function Test-Regex {
    param([string]$Text, [string]$Pattern)
    $m = [regex]::Match($Text, $Pattern)
    if ($m.Success) { return $m.Groups[1].Value }
    return ""
}

$root       = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$quartusDir = Join-Path $root "fpga\baseline_cnn\quartus"
$scriptsDir = Join-Path $root "fpga\baseline_cnn\scripts"
$rtlDir     = Join-Path $root "fpga\baseline_cnn\rtl"
$tbDir      = Join-Path $root "fpga\baseline_cnn\tb"
$outDir     = Join-Path $quartusDir "output_files"
$reportDir  = $quartusDir
if ($ExistingReportDir -ne "") {
    if (-not $ValidateExistingReportsOnly) {
        Write-Host "[ac620_selftest] ERROR: ExistingReportDir requires ValidateExistingReportsOnly" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path -LiteralPath $ExistingReportDir -PathType Container)) {
        Write-Host "[ac620_selftest] ERROR: report directory not found: $ExistingReportDir" -ForegroundColor Red
        exit 1
    }
    $reportDir = (Resolve-Path -LiteralPath $ExistingReportDir).Path
}
$compileLog = if ($reportDir -eq $quartusDir) {
    Join-Path $outDir "compile_$ProjectName.log"
} else {
    Join-Path $reportDir "compile_$ProjectName.log"
}
$sofPath    = Join-Path $reportDir "$ProjectName.sof"

$fail = @()

# ---- tool resolution (report-only mode must not touch either toolchain) ----
if (-not $ValidateExistingReportsOnly) {
    $quartus_sh = Resolve-Tool "QUARTUS_BIN" $DEFAULT_QUARTUS_BIN "quartus_sh.exe"
    if (-not $quartus_sh) { exit 1 }
    if (-not $SkipQuesta) {
        $vlib = Resolve-Tool "QUESTA_BIN" $DEFAULT_QUESTA_BIN "vlib.exe"
        $vlog = Resolve-Tool "QUESTA_BIN" $DEFAULT_QUESTA_BIN "vlog.exe"
        $vsim = Resolve-Tool "QUESTA_BIN" $DEFAULT_QUESTA_BIN "vsim.exe"
        if (-not $vlib -or -not $vlog -or -not $vsim) { exit 1 }
    }
} else {
    Write-Host "[ac620_selftest] REPORT-ONLY: no Questa, project creation, Quartus compile, or FPGA programming" -ForegroundColor Cyan
}

# ============================================================ #
# Part 1 - Questa end-to-end self-test
# ============================================================ #
if (-not $ValidateExistingReportsOnly -and -not $SkipQuesta) {
    Write-Host "`n===== [1/2] Questa end-to-end board self-test =====" -ForegroundColor Cyan
    $workRel  = "fpga/baseline_cnn/sim/work"       # forward slashes for Questa
    $workDir  = Join-Path $root "fpga\baseline_cnn\sim\work"
    if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir }
    & $vlib $workDir
    if ($LASTEXITCODE -ne 0) { Write-Host "[ac620_selftest] vlib failed" -ForegroundColor Red; exit 1 }

    $srcs = @(
        (Join-Path $rtlDir "requantize_u8.v"),
        (Join-Path $rtlDir "requantize_u8_pipe.v"),
        (Join-Path $rtlDir "gap_div49.v"),
        (Join-Path $rtlDir "sync_ram_u8.v"),
        (Join-Path $rtlDir "sync_rom_s8.v"),
        (Join-Path $rtlDir "sync_rom_s32.v"),
        (Join-Path $rtlDir "stem_conv_serial.v"),
        (Join-Path $rtlDir "maxpool2x2_stream.v"),
        (Join-Path $rtlDir "stem_pool1_pipeline.v"),
        (Join-Path $rtlDir "conv_u8_serial.v"),
        (Join-Path $rtlDir "gap_stream_u8.v"),
        (Join-Path $rtlDir "fc_argmax_serial.v"),
        (Join-Path $rtlDir "baseline_cnn_core.v"),
        (Join-Path $rtlDir "ac620_cnn_selftest_top.v"),
        (Join-Path $tbDir  "tb_ac620_cnn_selftest.v")
    )
    & $vlog -work $workRel $srcs 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "[ac620_selftest] vlog failed" -ForegroundColor Red; exit 1 }

    $logFile = Join-Path $root "fpga\baseline_cnn\sim\tb_ac620_cnn_selftest.log"
    Push-Location $root
    try {
        & $vsim -c -work $workRel -do "run -all; quit -f" -l $logFile "tb_ac620_cnn_selftest" 2>&1
        $qCode = $LASTEXITCODE
    }
    finally { Pop-Location }
    $passed = (Get-Content $logFile -ErrorAction SilentlyContinue | Select-String -Pattern "ALL_PASS" -Quiet)
    if ($qCode -ne 0 -or -not $passed) {
        Write-Host "[ac620_selftest] Questa self-test FAILED (exit=$qCode)" -ForegroundColor Red
        $fail += "Questa self-test failed (exit=$qCode)"
    } else {
        Write-Host "[ac620_selftest] Questa self-test OK (log: $logFile)" -ForegroundColor Green
    }
} elseif (-not $ValidateExistingReportsOnly) {
    Write-Host "[ac620_selftest] Skipping Questa part (SkipQuesta)" -ForegroundColor Yellow
}

# ---- early exit if only the Questa part was requested ----
if (-not $ValidateExistingReportsOnly -and $SkipQuartus) {
    Write-Host "`n===== AC620 CNN SELFTEST QUESTA SUMMARY =====" -ForegroundColor Cyan
    if ($fail.Count -gt 0) {
        Write-Host "[ac620_selftest] Questa self-test FAILURE" -ForegroundColor Red
        foreach ($f in $fail) { Write-Host "  - $f" -ForegroundColor Red }
        exit 1
    }
    Write-Host "[ac620_selftest] AC620_CNN_SELFTEST_QUESTA_OK" -ForegroundColor Green
    exit 0
}

# ============================================================ #
# Part 2 - Quartus full board compile + assertions
# ============================================================ #
if (-not $ValidateExistingReportsOnly) {
    Write-Host "`n===== [2/2] Quartus full board compile =====" -ForegroundColor Cyan
    if (-not (Test-Path $quartusDir)) { New-Item -ItemType Directory -Path $quartusDir | Out-Null }
    Push-Location $quartusDir
    try {
        Write-Host "[ac620_selftest] 1/3 check_device.tcl" -ForegroundColor Cyan
        & $quartus_sh -t (Join-Path $scriptsDir "check_device.tcl") 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ac620_selftest] check_device failed" -ForegroundColor Red; exit 1
        }

        Write-Host "[ac620_selftest] 2/3 create_ac620_cnn_selftest_project.tcl" -ForegroundColor Cyan
        & $quartus_sh -t (Join-Path $scriptsDir "create_ac620_cnn_selftest_project.tcl") 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ac620_selftest] project creation failed" -ForegroundColor Red; exit 1
        }

        Write-Host "[ac620_selftest] 3/3 quartus_sh --flow compile $ProjectName" -ForegroundColor Cyan
        if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
        & $quartus_sh --flow compile $ProjectName 2>&1 | Tee-Object -FilePath $compileLog | Out-Null
        $flowCode = $LASTEXITCODE
        Write-Host "[ac620_selftest] compile exit code = $flowCode" -ForegroundColor $(if ($flowCode -eq 0) { "Green" } else { "Red" })
    }
    finally { Pop-Location }
} else {
    Write-Host "`n===== Existing AC620 report guard validation =====" -ForegroundColor Cyan
    $flowCode = 0
}

# ---- report files ----
$flowRpt   = Join-Path $reportDir "$ProjectName.flow.rpt"
$fitRpt    = Join-Path $reportDir "$ProjectName.fit.rpt"
$mapRpt    = Join-Path $reportDir "$ProjectName.map.rpt"
$fitSum    = Join-Path $reportDir "$ProjectName.fit.summary"
$mapSum    = Join-Path $reportDir "$ProjectName.map.summary"
$staSum    = Join-Path $reportDir "$ProjectName.sta.summary"
$staRpt    = Join-Path $reportDir "$ProjectName.sta.rpt"
$qsf       = Join-Path $reportDir "$ProjectName.qsf"

$flowTxt = if (Test-Path $flowRpt) { Get-Content $flowRpt -Raw } else { "" }
$fitTxt  = if (Test-Path $fitRpt)  { Get-Content $fitRpt  -Raw } else { "" }
$mapTxt  = if (Test-Path $mapRpt)  { Get-Content $mapRpt  -Raw } else { "" }
$fitSumT = if (Test-Path $fitSum)  { Get-Content $fitSum  -Raw } else { "" }
$mapSumT = if (Test-Path $mapSum)  { Get-Content $mapSum  -Raw } else { "" }
$staSumT = if (Test-Path $staSum)  { Get-Content $staSum  -Raw } else { "" }
$staRptT = if (Test-Path $staRpt)  { Get-Content $staRpt  -Raw } else { "" }
$qsfTxt  = if (Test-Path $qsf)     { Get-Content $qsf     -Raw } else { "" }
$logTxt  = if (Test-Path $compileLog) { Get-Content $compileLog -Raw } else { "" }
$allTxt  = "$logTxt`n$fitTxt`n$flowTxt`n$staRptT"

# ---- 4a. Flow Successful ----
$flowStatus = Test-Regex $flowTxt 'Flow Status\s*[=:;]\s*(\S+)'
if ($flowCode -ne 0 -or $flowStatus -ne "Successful") {
    $fail += "Flow not Successful (exit=$flowCode status='$flowStatus')"
    Write-Host "[ac620_selftest] FAIL Flow status = '$flowStatus'" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  Flow = Successful" -ForegroundColor Green }

# ---- 4a2. Fitter Successful ----
$fitStatus = Test-Regex $fitSumT 'Fitter Status\s*[=:;]\s*(\S+)'
if ($fitStatus -ne "Successful") {
    $fail += "Fitter not Successful (status='$fitStatus')"
    Write-Host "[ac620_selftest] FAIL Fitter status = '$fitStatus'" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  Fitter = Successful" -ForegroundColor Green }

# ---- 4b. device strictly EP4CE10F17C8 ----
$device = Test-Regex $fitSumT 'Device\s*[=:;]\s*(\S+)'
if ($device -ne "EP4CE10F17C8") {
    $fail += "Device mismatch: '$device'"
    Write-Host "[ac620_selftest] FAIL Device = '$device'" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  Device = EP4CE10F17C8" -ForegroundColor Green }

# ---- 4c. physical pins == 5, virtual == 0 ----
$pins = Test-Regex $mapSumT 'Total pins\s*[=:;]\s*([\d,]+)'
$pins = $pins -replace ",", ""
$virtPins = Test-Regex $mapSumT 'Total virtual pins\s*[=:;]\s*([\d,]+)'
$virtPins = $virtPins -replace ",", ""
if ($virtPins -eq "") { $virtPins = "0" }
if ([int]$pins -ne 5) {
    $fail += "Physical pin count $pins != 5"
    Write-Host "[ac620_selftest] FAIL Total pins = $pins (expect 5)" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  Physical pins = 5" -ForegroundColor Green }
$virtInQsf = [regex]::Matches($qsfTxt, "(?i)VIRTUAL_PIN").Count
if ($virtInQsf -gt 0 -or $virtPins -ne "0") {
    $fail += "Virtual pins present (qsf=$virtInQsf, map=$virtPins)"
    Write-Host "[ac620_selftest] FAIL Virtual pins: qsf=$virtInQsf map=$virtPins" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  No VIRTUAL_PIN (qsf=0, map=0)" -ForegroundColor Green }

# ---- 4d. SDC read ----
$sdcRead = [regex]::Matches($allTxt, "(?i)SDC file").Count
if ($sdcRead -lt 1) {
    $fail += "SDC file not read"
    Write-Host "[ac620_selftest] FAIL SDC not read" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  SDC file read ($sdcRead mention(s))" -ForegroundColor Green }

# ---- 4e. no "Timing requirements not specified" ----
$timingReqs = ([regex]::Matches($allTxt, "Timing requirements not specified")).Count
if ($timingReqs -gt 0) {
    $fail += "'Timing requirements not specified' present x$timingReqs"
    Write-Host "[ac620_selftest] FAIL 'Timing requirements not specified' x$timingReqs" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  No 'Timing requirements not specified'" -ForegroundColor Green }

# ---- 4f. no latch / multi-driver / illegal pin / missing ROM / depth overflow ----
$latchCount = ([regex]::Matches($allTxt, "Inferred latch")).Count
if ($latchCount -gt 0) { $fail += "Latches inferred ($latchCount)"; Write-Host "[ac620_selftest] FAIL latches=$latchCount" -ForegroundColor Red }
else { Write-Host "[ac620_selftest] OK  No latches" -ForegroundColor Green }

$multiDrv = ([regex]::Matches($allTxt, "(?i)multiple driver|multi-driver")).Count
if ($multiDrv -gt 0) { $fail += "Multi-driver warnings ($multiDrv)"; Write-Host "[ac620_selftest] FAIL multi-driver=$multiDrv" -ForegroundColor Red }
else { Write-Host "[ac620_selftest] OK  No multi-driver" -ForegroundColor Green }

$illegalPin = ([regex]::Matches($allTxt, "(?i)illegal (pin|location)|not a legal location")).Count
if ($illegalPin -gt 0) { $fail += "Illegal pin/location ($illegalPin)"; Write-Host "[ac620_selftest] FAIL illegal-pin=$illegalPin" -ForegroundColor Red }
else { Write-Host "[ac620_selftest] OK  No illegal pins" -ForegroundColor Green }

$romMiss = ([regex]::Matches($allTxt, "(?i)cannot open memory initialization|couldn't open|unable to open|memory initialization file .* not found")).Count
if ($romMiss -gt 0) { $fail += "ROM init file missing ($romMiss)"; Write-Host "[ac620_selftest] FAIL missing-ROM=$romMiss" -ForegroundColor Red }
else { Write-Host "[ac620_selftest] OK  No missing ROM init" -ForegroundColor Green }

# A real depth overflow reads "exceeds the depth" / "depth exceeded".  The
# benign Warning 127005 ("Memory depth ... differs from memory depth ... setting
# initial value for remaining addresses to 0") is the sync_rom_s32 template
# padding MIFs up to a 16-entry ROM and is NOT a depth overflow.
$depthOv = ([regex]::Matches($allTxt, "(?i)exceeds the (memory|RAM|block|specified|depth)|depth (is )?(exceeded|too (large|small))|not enough depth|RAM.*depth.*not (fit|enough)")).Count
if ($depthOv -gt 0) { $fail += "RAM/ROM depth overflow ($depthOv)"; Write-Host "[ac620_selftest] FAIL depth-overflow=$depthOv" -ForegroundColor Red }
else { Write-Host "[ac620_selftest] OK  No RAM/ROM depth overflow (Warning 127005 ROM-padding excluded)" -ForegroundColor Green }

# ---- 4g. CNN hierarchy preserved (not optimised away) ----
$hierNeeded = @("baseline_cnn_core", "stem_conv_serial", "conv_u8_serial",
                "gap_stream_u8", "fc_argmax_serial", "requantize_u8_pipe")
$hierMissing = @()
foreach ($h in $hierNeeded) {
    if (-not $fitTxt.Contains($h)) { $hierMissing += $h }
}
if ($hierMissing.Count -gt 0) {
    $fail += "CNN hierarchy missing: $($hierMissing -join ',')"
    Write-Host "[ac620_selftest] FAIL hierarchy missing: $($hierMissing -join ',')" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  CNN hierarchy preserved" -ForegroundColor Green }

# ---- resources ----
$le   = Test-Regex $fitSumT 'Total logic elements\s*[=:;]\s*([\d,]+)'
$regs = Test-Regex $fitSumT 'Total registers\s*[=:;]\s*([\d,]+)'
$dsp  = Test-Regex $fitSumT 'Embedded Multiplier 9-bit elements\s*[=:;]\s*([\d,]+)'
$pll  = Test-Regex $fitSumT 'Total PLLs\s*[=:;]\s*([\d,]+)'
if ($pll -eq "") { $pll = "0" }
# block memory bits and the M9K count come from fit.rpt (not fit.summary):
#   Total block memory bits              ; 166,784 / 423,936 ( 39 % )
#   Total block memory implementation bits ; 276,480 / 423,936 ( 65 % )
# M9K blocks = implementation bits / 9216 (one M9K = 9216 bits on Cyclone IV E).
$memBits = Test-Regex $fitTxt 'Total block memory bits\s*[=:;]\s*([\d,]+)'
$implBits = Test-Regex $fitTxt 'Total block memory implementation bits\s*[=:;]\s*([\d,]+)'
if ($implBits -ne "") { $m9k = [math]::Floor(([double]($implBits -replace ",","")) / 9216.0) } else { $m9k = "?" }

# Multi-evidence resource guard.  LE is deliberately not the preservation
# criterion: retiming and fitter packing can legitimately move logic between
# LEs, registers, DSP input registers and M9Ks.  The hierarchy plus the frozen
# storage/compute lower bounds provide the real "CNN not optimised away" proof.
$leN   = if ($le   -ne "") { [int]($le   -replace ",","") } else { -1 }
$regsN = if ($regs -ne "") { [int]($regs -replace ",","") } else { -1 }
$dspN  = if ($dsp  -ne "") { [int]($dsp  -replace ",","") } else { -1 }
if ($leN -le 0) {
    $fail += "Logic element report missing or invalid ('$le')"
    Write-Host "[ac620_selftest] FAIL LE report invalid ('$le')" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  LE reported ($leN; informational, no arbitrary 3000 threshold)" -ForegroundColor Green }
if ($regsN -lt 900) {
    $fail += "Register count below structural lower bound ($regsN < 900)"
    Write-Host "[ac620_selftest] FAIL registers=$regsN below 900" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  Register guard ($regsN >= 900)" -ForegroundColor Green }
if ($m9k -eq "?" -or [int]$m9k -lt 29) {
    $fail += "M9K count below structural lower bound ($m9k < 29)"
    Write-Host "[ac620_selftest] FAIL M9K=$m9k below 29" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  M9K guard ($m9k >= 29)" -ForegroundColor Green }
if ($dspN -lt 19) {
    $fail += "9-bit multiplier elements below structural lower bound ($dspN < 19)"
    Write-Host "[ac620_selftest] FAIL 9-bit multipliers=$dspN below 19" -ForegroundColor Red
} else { Write-Host "[ac620_selftest] OK  Multiplier guard ($dspN >= 19)" -ForegroundColor Green }

# ---- 4h. every timing corner setup/hold >= 0 ----
$worstSetup = $null; $worstHold = $null
$timingPattern = "(?m)^\s*Type\s*:\s*([^\r\n]+?)\s+(Setup|Hold)\s+'clk_50m'\s*\r?\n\s*Slack\s*:\s*(-?[\d.]+)\s*\r?\n\s*TNS\s*:\s*(-?[\d.]+)"
$timingMatches = [regex]::Matches($staSumT, $timingPattern)
if ($timingMatches.Count -ne 6) {
    $fail += "Expected exactly six setup/hold corner records, parsed $($timingMatches.Count)"
    Write-Host "[ac620_selftest] FAIL timing corner records=$($timingMatches.Count), expect 6" -ForegroundColor Red
}
foreach ($tm in $timingMatches) {
    $corner = $tm.Groups[1].Value.Trim()
    $kind   = $tm.Groups[2].Value
    $sl     = [double]$tm.Groups[3].Value
    $tns    = [double]$tm.Groups[4].Value
    Write-Host "[ac620_selftest] timing $kind | $corner | slack=$sl ns TNS=$tns ns" -ForegroundColor Gray
    if ($kind -eq "Setup") { if ($worstSetup -eq $null -or $sl -lt $worstSetup) { $worstSetup = $sl } }
    else { if ($worstHold -eq $null -or $sl -lt $worstHold) { $worstHold = $sl } }
    if ($sl -lt 0) {
        $fail += "$kind slack negative ($sl ns) at corner $corner"
        Write-Host "[ac620_selftest] FAIL $kind slack=$sl ns at $corner" -ForegroundColor Red
    }
    if ([math]::Abs($tns) -gt 0.0000001) {
        $fail += "$kind TNS non-zero ($tns ns) at corner $corner"
        Write-Host "[ac620_selftest] FAIL $kind TNS=$tns ns at $corner" -ForegroundColor Red
    }
}
if ($worstSetup -ne $null -and $worstHold -ne $null) {
    Write-Host "[ac620_selftest] OK  Timing guard (all parsed setup/hold slack >= 0 and TNS = 0)" -ForegroundColor Green
    Write-Host "[ac620_selftest] worst Setup slack=$worstSetup ns, worst Hold slack=$worstHold ns" -ForegroundColor Cyan
}

# ---- Fmax (from the worst setup slack against 20 ns) ----
if ($worstSetup -ne $null) {
    $fmax = if ((20.0 - $worstSetup) -gt 0) { [math]::Round(1000.0 / (20.0 - $worstSetup), 1) } else { 0 }
    Write-Host "[ac620_selftest] Fmax ~ $fmax MHz (per 20 ns clock, worst setup)" -ForegroundColor Cyan
}

# ---- worst warning ----
$sev = "none"
foreach ($pattern in @("(?m)^\s*Error:", "(?m)^\s*Critical Warning:", "(?m)^\s*Warning \(10240\)")) {
    if ([regex]::IsMatch($allTxt, $pattern)) { $sev = $pattern.Replace("(?m)^\s*", "").Replace(":",""); break }
}
Write-Host "[ac620_selftest] Highest severity in log: $sev" -ForegroundColor Cyan

# ---- Fitter RAM Summary (the M9K breakdown) ----
Write-Host "`n----- Fitter RAM Summary -----" -ForegroundColor Cyan
$fitLines = if (Test-Path $fitRpt) { Get-Content $fitRpt } else { @() }
$ramStart = -1
for ($i = 0; $i -lt $fitLines.Count; $i++) {
    if ($fitLines[$i] -match "(?i)Fitter RAM Summary") { $ramStart = $i; break }
}
if ($ramStart -ge 0) {
    for ($j = $ramStart; $j -lt $fitLines.Count; $j++) {
        Write-Host $fitLines[$j]
        if ($j -gt $ramStart + 55) { Write-Host "(... truncated ...)"; break }
    }
} else {
    Write-Host "(no Fitter RAM Summary block found; see $fitRpt)" -ForegroundColor Yellow
}

# ---- 4i. .sof generated ----
if (Test-Path $sofPath) {
    $sofSize = (Get-Item $sofPath).Length
    if ($sofSize -le 0) { $fail += "SOF is empty: $sofPath" }
    Write-Host "[ac620_selftest] SOF: $sofPath ($sofSize bytes)" -ForegroundColor Green
} else {
    $fail += "SOF not generated: $sofPath"
    Write-Host "[ac620_selftest] FAIL SOF missing: $sofPath" -ForegroundColor Red
}

# ---- summary ----
Write-Host "`n===== AC620 CNN SELFTEST SUMMARY =====" -ForegroundColor Cyan
Write-Host "Flow        : $flowStatus"
Write-Host "Fitter      : $fitStatus"
Write-Host "Device      : $device"
Write-Host "Pins        : physical=$pins virtual=$virtPins"
Write-Host "Resources   : LE=$le regs=$regs M9K=$m9k blockMemBits=$memBits 9bitMult=$dsp PLL=$pll"
Write-Host "Timing      : worstSetup=$worstSetup ns worstHold=$worstHold ns Fmax=$fmax MHz"
Write-Host "Worst warn  : $sev"
Write-Host "SOF         : $sofPath"
Write-Host "================================"

if ($fail.Count -gt 0) {
    Write-Host "[ac620_selftest] FAILURE:" -ForegroundColor Red
    foreach ($f in $fail) { Write-Host "  - $f" -ForegroundColor Red }
    exit 1
}
if ($ValidateExistingReportsOnly) {
    Write-Host "[ac620_selftest] AC620_CNN_SELFTEST_REPORT_GUARDS_OK" -ForegroundColor Green
} else {
    Write-Host "[ac620_selftest] AC620_CNN_SELFTEST_BUILD_OK" -ForegroundColor Green
}
exit 0
