<#
.SYNOPSIS
  Run the Quartus smoke flow for baseline_cnn_smoke on the frozen target.

.DESCRIPTION
  1. check_device.tcl          -> verify EP4CE10F17C8 is recognised
  2. create_quartus_project.tcl-> write .qpf/.qsf under fpga/baseline_cnn/quartus
  3. quartus_sh --flow compile baseline_cnn_smoke
  4. Parse the compile log + map/fit reports and print a summary:

       Flow status | Logic elements | Registers | Embedded multiplier 9-bit
       elements | Memory bits | Highest severity message | latches |
       truncation / signed warnings

  This is a clockless combinational smoke project, so NO timing conclusion is
  drawn from this run.

.PARAMETER QUARTUS_BIN
  Optional; overrides the default Quartus bin directory (or points straight
  at the .exe). Falls back to the frozen install path if unset.
#>
$ErrorActionPreference = "Continue"

$DEFAULT_QUARTUS_BIN = "D:\tools\altera_lite\25.1std\quartus\bin64"

function Resolve-Tool {
    param([string]$EnvVar, [string]$DefaultDir, [string]$Exe)
    if (Test-Path env:$EnvVar) {
        $p = (Get-Item env:$EnvVar).Value
        $candidate = if (Test-Path $p -PathType Leaf) { $p } else { Join-Path $p $Exe }
        if (Test-Path $candidate) { return $candidate }
        Write-Host "[quartus_smoke] ERROR: $Exe not found via $EnvVar ($p)" -ForegroundColor Red
        return $null
    }
    $candidate = Join-Path $DefaultDir $Exe
    if (Test-Path $candidate) { return $candidate }
    Write-Host "[quartus_smoke] ERROR: $Exe not found at $candidate" -ForegroundColor Red
    return $null
}

$quartus_sh = Resolve-Tool "QUARTUS_BIN" $DEFAULT_QUARTUS_BIN "quartus_sh.exe"
if (-not $quartus_sh) { exit 1 }

$root       = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$quartusDir = Join-Path $root "fpga\baseline_cnn\quartus"
$scriptsDir = Join-Path $root "fpga\baseline_cnn\scripts"
$projName   = "baseline_cnn_smoke"
$outDir     = Join-Path $quartusDir "output_files"
$compileLog = Join-Path $outDir "compile.log"

function Test-Regex {
    param([string]$Text, [string]$Pattern)
    $m = [regex]::Match($Text, $Pattern)
    if ($m.Success) { return $m.Groups[1].Value }
    return ""
}

# ---- 1. device check ----
Write-Host "[quartus_smoke] 1/4 check_device.tcl" -ForegroundColor Cyan
if (-not (Test-Path $quartusDir)) { New-Item -ItemType Directory -Path $quartusDir | Out-Null }
Push-Location $quartusDir
try {
    & $quartus_sh -t (Join-Path $scriptsDir "check_device.tcl") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[quartus_smoke] check_device failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1
    }

    # ---- 2. create project ----
    Write-Host "[quartus_smoke] 2/4 create_quartus_project.tcl" -ForegroundColor Cyan
    & $quartus_sh -t (Join-Path $scriptsDir "create_quartus_project.tcl") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[quartus_smoke] create_quartus_project failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1
    }

    # ---- 3. compile ----
    Write-Host "[quartus_smoke] 3/4 quartus_sh --flow compile $projName" -ForegroundColor Cyan
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    & $quartus_sh --flow compile $projName 2>&1 | Tee-Object -FilePath $compileLog | Out-Null
    $flowCode = $LASTEXITCODE
    Write-Host "[quartus_smoke] compile exit code = $flowCode" -ForegroundColor $(if ($flowCode -eq 0) { "Green" } else { "Red" })

    # ---- 4. report summary ----
    Write-Host "`n===== QUARTUS SMOKE SUMMARY =====" -ForegroundColor Cyan
    # Reports are emitted next to the .qpf (project dir), not in output_files/.
    $mapRpt  = Join-Path $quartusDir "$projName.map.rpt"
    $fitRpt  = Join-Path $quartusDir "$projName.fit.rpt"
    $flowRpt = Join-Path $quartusDir "$projName.flow.rpt"
    $qsf     = Join-Path $quartusDir "$projName.qsf"

    if (Test-Path $qsf) {
        $qsfText = Get-Content $qsf -Raw
        Write-Host ("QSF  FAMILY : {0}" -f (Test-Regex $qsfText 'set_global_assignment -name FAMILY "([^"]+)"'))
        Write-Host ("QSF  DEVICE : {0}" -f (Test-Regex $qsfText 'set_global_assignment -name DEVICE ([A-Za-z0-9]+)'))
    }

    if (Test-Path $flowRpt) {
        $ftxt = Get-Content $flowRpt -Raw
        Write-Host ("FLOW status : {0}" -f (Test-Regex $ftxt 'Flow Status\s*[=:;]\s*(\S+)'))
    }

    if (Test-Path $mapRpt) {
        $txt = Get-Content $mapRpt -Raw
        Write-Host ("MAP  Logic  : {0}" -f (Test-Regex $txt 'Total logic elements\s*[=:;]\s*([\d,]+)'))
        Write-Host ("MAP  Regs   : {0}" -f (Test-Regex $txt 'Total registers\s*[=:;]\s*([\d,]+)'))
        Write-Host ("MAP  Pins   : {0}" -f (Test-Regex $txt 'Total pins\s*[=:;]\s*([\d,]+)'))
        Write-Host ("MAP  VirtPins: {0}" -f (Test-Regex $txt 'Total virtual pins\s*[=:;]\s*([\d,]+)'))
        Write-Host ("MAP  DSP    : {0}" -f (Test-Regex $txt 'Embedded Multiplier 9-bit elements\s*[=:;]\s*([\d,]+)'))
        Write-Host ("MAP  Memory : {0}" -f (Test-Regex $txt 'Total memory bits\s*[=:;]\s*([\d,]+)'))
    }

    # latches / truncation / signed warnings (across log + map report)
    $logTxt     = Get-Content $compileLog -Raw -ErrorAction SilentlyContinue
    $mapRptText = if (Test-Path $mapRpt) { Get-Content $mapRpt -Raw } else { "" }
    $allTxt = "$logTxt`n" + $mapRptText
    $latchCount   = ([regex]::Matches($allTxt, "Inferred latch")).Count
    $truncCount   = ([regex]::Matches($allTxt, "(?i)truncat")).Count
    $signedCount  = ([regex]::Matches($allTxt, "(?i)signed")).Count
    Write-Host "WARN latches (Inferred latch): $latchCount"
    Write-Host "WARN truncation messages     : $truncCount"
    Write-Host "WARN signed-related messages : $signedCount"

    # highest severity message
    $sev = "none"
    foreach ($pattern in @("(?m)^\s*Error:", "(?m)^\s*Critical Warning:", "(?m)^\s*Warning \(10240\)")) {
        if ([regex]::IsMatch($allTxt, $pattern)) { $sev = $pattern.Replace("(?m)^\s*", "").Replace(":",""); break }
    }
    Write-Host "HIGHEST severity in log     : $sev"

    Write-Host "compile.log -> $compileLog"
    Write-Host "=============================`n"
}
finally {
    Pop-Location
}

if ($flowCode -ne 0) {
    Write-Host "[quartus_smoke] FAILURE (compile exit=$flowCode)" -ForegroundColor Red; exit 1
}
Write-Host "[quartus_smoke] OK" -ForegroundColor Green
exit 0
