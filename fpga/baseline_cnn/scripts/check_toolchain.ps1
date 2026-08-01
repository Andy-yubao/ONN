<#
.SYNOPSIS
    FPGA 工具链探测脚本（BaselineCNN）
.DESCRIPTION
    查找并验证 Quartus / ModelSim-Questa 命令行工具链：
      1) 优先读取环境变量 QUARTUS_BIN / QUESTA_BIN
      2) 未设置时使用默认安装路径
      3) 执行版本命令验证可用性并输出绝对路径
    不修改系统 PATH，不调用 GUI。
    任一工具缺失或不可用时以非零退出码结束。

    在 PowerShell 中运行：
      .\fpga\baseline_cnn\scripts\check_toolchain.ps1
.EXAMPLE
    .\fpga\baseline_cnn\scripts\check_toolchain.ps1
#>

$ErrorActionPreference = "Continue"

# ---------- 默认安装路径 ----------
$DEFAULT_QUARTUS_BIN = "D:\tools\altera_lite\25.1std\quartus\bin64"
$DEFAULT_QUESTA_BIN  = "D:\tools\altera_lite\25.1std\questa_fse\win64"

# ---------- 工具定义：名称 / 版本参数 / 环境变量 / 默认目录 ----------
$Tools = @(
    @{ Name = "quartus_sh.exe"; VersionArg = "--version"; EnvVar = "QUARTUS_BIN"; DefaultDir = $DEFAULT_QUARTUS_BIN },
    @{ Name = "vlog.exe";       VersionArg = "-version";  EnvVar = "QUESTA_BIN";  DefaultDir = $DEFAULT_QUESTA_BIN },
    @{ Name = "vsim.exe";       VersionArg = "-version";  EnvVar = "QUESTA_BIN";  DefaultDir = $DEFAULT_QUESTA_BIN }
)

function Resolve-ToolPath {
    param(
        [string]$Name,
        [string]$EnvVar,
        [string]$DefaultDir
    )

    $envValue = [Environment]::GetEnvironmentVariable($EnvVar, "Process")
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        # 环境变量优先：可指向 bin 目录，也可直接指向 .exe
        if ($envValue.Trim().EndsWith(".exe", [System.StringComparison]::OrdinalIgnoreCase)) {
            $candidate = $envValue.Trim()
        } else {
            $candidate = Join-Path $envValue.Trim() $Name
        }
        if (Test-Path -LiteralPath $candidate) {
            return @{ Found = $true; Path = $candidate; Source = "env:$EnvVar" }
        }
        return @{ Found = $false; Path = $candidate; Source = "env:$EnvVar" }
    }

    # 默认安装路径
    $candidate = Join-Path $DefaultDir $Name
    if (Test-Path -LiteralPath $candidate) {
        return @{ Found = $true; Path = $candidate; Source = "default" }
    }
    return @{ Found = $false; Path = $candidate; Source = "default" }
}

Write-Host ""
Write-Host "=== FPGA 工具链探测 ===" -ForegroundColor Cyan

$results = @()
$index = 1
$anyFailed = $false

foreach ($t in $Tools) {
    Write-Host "[$index/$($Tools.Count)] $($t.Name)" -ForegroundColor Yellow
    $res = Resolve-ToolPath -Name $t.Name -EnvVar $t.EnvVar -DefaultDir $t.DefaultDir

    if ($res.Found) {
        Write-Host "  路径: $($res.Path)  ($($res.Source))" -ForegroundColor Green
        $output = & $res.Path $t.VersionArg 2>&1
        if ($LASTEXITCODE -eq 0) {
            $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            Write-Host "  [OK] 版本命令执行成功" -ForegroundColor Green
            $results += @{ Name = $t.Name; Path = $res.Path; Ok = $true }
        } else {
            Write-Host "  [错误] 版本命令执行失败 (exit $LASTEXITCODE)" -ForegroundColor Red
            $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            $results += @{ Name = $t.Name; Path = $res.Path; Ok = $false }
            $anyFailed = $true
        }
    } else {
        Write-Host "  [缺失] 未找到 $($t.Name)" -ForegroundColor Red
        Write-Host "  已尝试: $($res.Path)" -ForegroundColor Red
        Write-Host "  提示: 可设置 $($t.EnvVar) 环境变量覆盖默认路径" -ForegroundColor Yellow
        $results += @{ Name = $t.Name; Path = $res.Path; Ok = $false }
        $anyFailed = $true
    }
    $index++
}

# ---------- 汇总 ----------
Write-Host ""
Write-Host "=== 汇总 ===" -ForegroundColor Cyan
foreach ($r in $results) {
    if ($r.Ok) {
        Write-Host ("{0,-16} {1}  {2}" -f $r.Name, "[OK]", $r.Path) -ForegroundColor Green
    } else {
        Write-Host ("{0,-16} {1}  {2}" -f $r.Name, "[缺失]", $r.Path) -ForegroundColor Red
    }
}

if ($anyFailed) {
    Write-Host ""
    Write-Host "工具链不完整，请检查安装路径或设置环境变量后重试。" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "工具链完整，全部可用。" -ForegroundColor Green
    exit 0
}
