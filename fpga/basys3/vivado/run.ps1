param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('synth', 'impl', 'program')]
    [string]$Stage
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$build = Join-Path $root 'fpga\basys3\.vivado'
$homeDir = Join-Path $build 'home'
$tempDir = Join-Path $build 'tmp'
$reports = Join-Path $root 'fpga\basys3\reports'
New-Item -ItemType Directory -Force -Path $homeDir, $tempDir, $reports | Out-Null

# Vivado 2026.1 on this Windows host misparses the inherited C:\ user paths
# while loading Tcl apps. Keep its process-local profile/temp paths inside build.
$env:HOME = $homeDir.Replace('\', '/')
$env:USERPROFILE = $env:HOME
$env:APPDATA = $env:HOME
$env:LOCALAPPDATA = $env:HOME
$env:TEMP = $tempDir.Replace('\', '/')
$env:TMP = $env:TEMP
$env:HOMEDRIVE = (Split-Path -Qualifier $homeDir)
$env:HOMEPATH = $homeDir.Substring($env:HOMEDRIVE.Length).Replace('\', '/')

$vivado = 'D:\tools\AMDDesignTools\2026.1\Vivado\bin\vivado.bat'
if (-not (Test-Path -LiteralPath $vivado)) { throw "Vivado not found at $vivado" }
$source = if ($Stage -eq 'program') { 'program_board.tcl' } else { 'run_direct.tcl' }
Push-Location $build
try {
    & $vivado -mode batch -log (Join-Path $reports "vivado_$Stage.log") `
        -journal (Join-Path $reports "vivado_$Stage.jou") `
        -source (Join-Path $PSScriptRoot $source) -tclargs $Stage
    $vivadoExit = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($vivadoExit -ne 0) { throw "Vivado $Stage failed with exit code $vivadoExit" }
