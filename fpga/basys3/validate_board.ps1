param(
    [Parameter(Mandatory=$true)][string]$Port,
    [ValidateRange(1,3)][int]$Level = 1,
    [ValidateRange(1,5000)][int]$Count = 100,
    [ValidateRange(55000,59999)][int]$StartIndex = 55000
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'D:\tools\anaconda3\envs\onn\python.exe'
$resultDir = Join-Path $PSScriptRoot '.vivado\board_results'
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
if ($StartIndex + $Count -gt 60000) { throw 'Batch exceeds validation split' }

Push-Location $root
try {
    $common = @('-m', 'fpga.basys3.host', '--backend', 'serial', '--port', $Port)
    & $python @common --level artificial --output (Join-Path $resultDir 'level1.json')
    if ($LASTEXITCODE -ne 0) { throw 'Level 1 failed' }
    if ($Level -ge 2) {
        & $python @common --level golden --split validation --output (Join-Path $resultDir 'level2.json')
        if ($LASTEXITCODE -ne 0) { throw 'Level 2 failed' }
    }
    if ($Level -ge 3) {
        $indices = @($StartIndex..($StartIndex + $Count - 1))
        & $python @common --level batch --split validation --indices @indices `
            --output (Join-Path $resultDir 'level3.json')
        if ($LASTEXITCODE -ne 0) { throw 'Level 3 failed' }
    }
} finally {
    Pop-Location
}
Write-Host "PASS: requested board validation levels completed; results in $resultDir"
