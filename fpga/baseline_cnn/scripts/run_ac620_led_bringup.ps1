<#
.SYNOPSIS
  Independent build+check entry for the AC620 LED bring-up board project.

.DESCRIPTION
  1. check_device.tcl                       -> verify EP4CE10F17C8 is recognised
  2. create_ac620_led_bringup_project.tcl   -> write .qpf/.qsf under quartus/
                                               (REAL pins: clk_50m=E1,
                                               led[0..3]=A2/B3/A4/A3, 3.3-V
                                               LVTTL; SDC_FILE referenced;
                                               NO virtual pins)
  3. quartus_sh --flow compile ac620_led_bringup  (full compile flow)
  4. Assertions on the reports / log:
       Flow Successful | device == EP4CE10F17C8 | physical pins == 5 |
       virtual pins == 0 | no VIRTUAL_PIN in .qsf | SDC read |
       no "Timing requirements not specified"
  5. Report setup / hold slack (slow 1200mV 85C corner) and Fmax
     (computed from the worst-case setup slack against the 20 ns clock).
  6. Print the exact generated .sof path (for the Quartus Programmer).

  This is a REAL physical board project: it is NOT part of run_all.ps1's
  normal RTL regression.  Run it standalone when you want to rebuild/verify
  the AC620 download bitstream.

.PARAMETER QUARTUS_BIN
  Optional; overrides the default Quartus bin directory (or points straight
  at the .exe). Falls back to the frozen install path if unset.
#>
param(
    [string]$ProjectName = "ac620_led_bringup"
)
$ErrorActionPreference = "Continue"

$DEFAULT_QUARTUS_BIN = "D:\tools\altera_lite\25.1std\quartus\bin64"

function Resolve-Tool {
    param([string]$EnvVar, [string]$DefaultDir, [string]$Exe)
    if (Test-Path env:$EnvVar) {
        $p = (Get-Item env:$EnvVar).Value
        $candidate = if (Test-Path $p -PathType Leaf) { $p } else { Join-Path $p $Exe }
        if (Test-Path $candidate) { return $candidate }
        Write-Host "[ac620_led] ERROR: $Exe not found via $EnvVar ($p)" -ForegroundColor Red
        return $null
    }
    $candidate = Join-Path $DefaultDir $Exe
    if (Test-Path $candidate) { return $candidate }
    Write-Host "[ac620_led] ERROR: $Exe not found at $candidate" -ForegroundColor Red
    return $null
}

$quartus_sh = Resolve-Tool "QUARTUS_BIN" $DEFAULT_QUARTUS_BIN "quartus_sh.exe"
if (-not $quartus_sh) { exit 1 }

# Paths are derived from the script location (repo-relative, no hardcoded drive).
$root       = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$quartusDir = Join-Path $root "fpga\baseline_cnn\quartus"
$scriptsDir = Join-Path $root "fpga\baseline_cnn\scripts"
$outDir     = Join-Path $quartusDir "output_files"
$compileLog = Join-Path $outDir "compile_$ProjectName.log"
$sofPath    = Join-Path $quartusDir "$ProjectName.sof"

function Test-Regex {
    param([string]$Text, [string]$Pattern)
    $m = [regex]::Match($Text, $Pattern)
    if ($m.Success) { return $m.Groups[1].Value }
    return ""
}

$fail = @()

# ---- 1. device check ----
Write-Host "[ac620_led] 1/3 check_device.tcl" -ForegroundColor Cyan
if (-not (Test-Path $quartusDir)) { New-Item -ItemType Directory -Path $quartusDir | Out-Null }
Push-Location $quartusDir
try {
    & $quartus_sh -t (Join-Path $scriptsDir "check_device.tcl") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ac620_led] check_device failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1
    }

    # ---- 2. create / refresh project ----
    Write-Host "[ac620_led] 2/3 create_ac620_led_bringup_project.tcl" -ForegroundColor Cyan
    & $quartus_sh -t (Join-Path $scriptsDir "create_ac620_led_bringup_project.tcl") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ac620_led] project creation failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1
    }

    # ---- 3. full compile flow ----
    Write-Host "[ac620_led] 3/3 quartus_sh --flow compile $ProjectName" -ForegroundColor Cyan
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    & $quartus_sh --flow compile $ProjectName 2>&1 | Tee-Object -FilePath $compileLog | Out-Null
    $flowCode = $LASTEXITCODE
    Write-Host "[ac620_led] compile exit code = $flowCode" -ForegroundColor $(if ($flowCode -eq 0) { "Green" } else { "Red" })
}
finally {
    Pop-Location
}

# ---- 4. assertions ----
$flowRpt   = Join-Path $quartusDir "$ProjectName.flow.rpt"
$fitRpt    = Join-Path $quartusDir "$ProjectName.fit.rpt"
$mapRpt    = Join-Path $quartusDir "$ProjectName.map.rpt"
$fitSum    = Join-Path $quartusDir "$ProjectName.fit.summary"
$mapSum    = Join-Path $quartusDir "$ProjectName.map.summary"
$staSum    = Join-Path $quartusDir "$ProjectName.sta.summary"
$staRpt    = Join-Path $quartusDir "$ProjectName.sta.rpt"
$qsf       = Join-Path $quartusDir "$ProjectName.qsf"

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

# 4a. Flow Successful
$flowStatus = Test-Regex $flowTxt 'Flow Status\s*[=:;]\s*(\S+)'
if ($flowCode -ne 0 -or $flowStatus -ne "Successful") {
    $fail += "Flow not Successful (exit=$flowCode status='$flowStatus')"
    Write-Host "[ac620_led] FAIL Flow status = '$flowStatus'" -ForegroundColor Red
} else {
    Write-Host "[ac620_led] OK  Flow status = Successful" -ForegroundColor Green
}

# 4b. device strictly EP4CE10F17C8
$device = Test-Regex $fitSumT 'Device\s*[=:;]\s*(\S+)'
if ($device -ne "EP4CE10F17C8") {
    $fail += "Device mismatch: '$device'"
    Write-Host "[ac620_led] FAIL Device = '$device'" -ForegroundColor Red
} else {
    Write-Host "[ac620_led] OK  Device = EP4CE10F17C8" -ForegroundColor Green
}

# 4c. physical pins == 5 (and not 0)
$pins = Test-Regex $mapSumT 'Total pins\s*[=:;]\s*([\d,]+)'
$pins = $pins -replace ",", ""
$virtPins = Test-Regex $mapSumT 'Total virtual pins\s*[=:;]\s*([\d,]+)'
$virtPins = $virtPins -replace ",", ""
if ($virtPins -eq "") { $virtPins = "0" }   # no virtual-pin line == 0 virtual pins
if ([int]$pins -ne 5) {
    $fail += "Physical pin count $pins != 5"
    Write-Host "[ac620_led] FAIL Total pins = $pins (expect 5)" -ForegroundColor Red
} else {
    Write-Host "[ac620_led] OK  Physical pins = 5" -ForegroundColor Green
}

# 4d. no VIRTUAL_PIN (neither in .qsf nor in the reports)
$virtInQsf  = [regex]::Matches($qsfTxt, "(?i)VIRTUAL_PIN").Count
$virtFromMap = if ($virtPins -eq "") { "?" } else { $virtPins }
if ($virtInQsf -gt 0 -or $virtPins -ne "0") {
    $fail += "Virtual pins present (qsf=$virtInQsf, map=$virtFromMap)"
    Write-Host "[ac620_led] FAIL Virtual pins: qsf=$virtInQsf map=$virtFromMap" -ForegroundColor Red
} else {
    Write-Host "[ac620_led] OK  No VIRTUAL_PIN (qsf=$virtInQsf, map=0)" -ForegroundColor Green
}

# 4e. SDC was read by the flow
$sdcRead = [regex]::Matches($allTxt, "(?i)SDC file").Count
if ($sdcRead -lt 1) {
    $fail += "SDC file not read by the flow"
    Write-Host "[ac620_led] FAIL SDC not read" -ForegroundColor Red
} else {
    Write-Host "[ac620_led] OK  SDC file read ($sdcRead mention(s))" -ForegroundColor Green
}

# 4f. no "Timing requirements not specified"
$timingReqs = ([regex]::Matches($allTxt, "Timing requirements not specified")).Count
if ($timingReqs -gt 0) {
    $fail += "'Timing requirements not specified' present ($timingReqs)"
    Write-Host "[ac620_led] FAIL 'Timing requirements not specified' x$timingReqs" -ForegroundColor Red
} else {
    Write-Host "[ac620_led] OK  No 'Timing requirements not specified'" -ForegroundColor Green
}

# ---- 5. timing summary (slow 1200mV 85C corner) ----
$setupSlack = Test-Regex $staSumT 'Slow 1200mV 85C Model Setup[^\n]*\n.*?Slack\s*[=:;]\s*(-?[\d.]+)'
$holdSlack  = Test-Regex $staSumT 'Slow 1200mV 85C Model Hold[^\n]*\n.*?Slack\s*[=:;]\s*(-?[\d.]+)'
if ($setupSlack -eq "") {
    Write-Host "[ac620_led] WARN setup/hold slack not found in sta.summary" -ForegroundColor Yellow
} else {
    $per = 20.0
    $slackNs = [double]$setupSlack
    $fmax = if (($per - $slackNs) -gt 0) { [math]::Round(1000.0 / ($per - $slackNs), 1) } else { 0 }
    Write-Host "[ac620_led] Setup slack = ${setupSlack} ns, Hold slack = $holdSlack ns, Fmax = ${fmax} MHz (per 20 ns clock)" -ForegroundColor Cyan
}

# ---- 6. .sof path ----
if (Test-Path $sofPath) {
    Write-Host "[ac620_led] SOF: $sofPath" -ForegroundColor Green
} else {
    $fail += "SOF not generated: $sofPath"
    Write-Host "[ac620_led] FAIL SOF missing: $sofPath" -ForegroundColor Red
}

# ---- summary ----
Write-Host "`n===== AC620 LED BRING-UP SUMMARY =====" -ForegroundColor Cyan
Write-Host "Flow        : $flowStatus"
Write-Host "Device      : $device"
Write-Host "Pins        : physical=$pins virtual=$virtFromMap"
Write-Host "Setup slack : $setupSlack ns | Hold slack : $holdSlack ns"
Write-Host "SOF         : $sofPath"
Write-Host "================================"

if ($fail.Count -gt 0) {
    Write-Host "[ac620_led] FAILURE:" -ForegroundColor Red
    foreach ($f in $fail) { Write-Host "  - $f" -ForegroundColor Red }
    exit 1
}
Write-Host "[ac620_led] AC620_LED_BRINGUP_OK" -ForegroundColor Green
exit 0
