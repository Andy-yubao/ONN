param(
    [string]$Python = "D:\tools\anaconda3\envs\onn\python.exe",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$resultRoot = Join-Path $PSScriptRoot "results\input_encoding_joint"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "onn Python not found: $Python"
}

$runs = @(
    [pscustomobject]@{ T = 4; Mapping = "quantile"; Ratio = "baseline"; Name = "T4_quantile_baseline" },
    [pscustomobject]@{ T = 4; Mapping = "linear";   Ratio = "0.30";    Name = "T4_linear_030" },
    [pscustomobject]@{ T = 4; Mapping = "quantile"; Ratio = "0.30";    Name = "T4_quantile_030" },
    [pscustomobject]@{ T = 5; Mapping = "quantile"; Ratio = "baseline"; Name = "T5_quantile_baseline" },
    [pscustomobject]@{ T = 5; Mapping = "linear";   Ratio = "0.30";    Name = "T5_linear_030" },
    [pscustomobject]@{ T = 5; Mapping = "quantile"; Ratio = "0.30";    Name = "T5_quantile_030" },
    [pscustomobject]@{ T = 8; Mapping = "quantile"; Ratio = "baseline"; Name = "T8_quantile_baseline" },
    [pscustomobject]@{ T = 8; Mapping = "linear";   Ratio = "0.30";    Name = "T8_linear_030" },
    [pscustomobject]@{ T = 8; Mapping = "quantile"; Ratio = "0.30";    Name = "T8_quantile_030" }
)

Push-Location $repoRoot
try {
    foreach ($run in $runs) {
        $resultDir = Join-Path $resultRoot $run.Name
        $metricsPath = Join-Path $resultDir "metrics.json"
        if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
            $metrics = Get-Content -Raw -LiteralPath $metricsPath | ConvertFrom-Json
            if ($metrics.epochs_run -ne 20 -or $metrics.test_evaluations -ne 1) {
                throw "Existing result is not a complete 20-epoch run: $resultDir"
            }
            Write-Host "SKIP complete $($run.Name)"
            continue
        }
        if ((Test-Path -LiteralPath $resultDir) -and
            (Get-ChildItem -LiteralPath $resultDir -Force | Select-Object -First 1)) {
            throw "Refusing to overwrite incomplete output: $resultDir"
        }

        Write-Host "START $($run.Name)"
        & $Python -m experiments.snn.conv_small.train `
            --time-steps $run.T `
            --time-mapping $run.Mapping `
            --target-firing-ratio $run.Ratio `
            --readout-mode accumulated_membrane `
            --seed 17 `
            --epochs 20 `
            --batch-size 256 `
            --learning-rate 0.001 `
            --device $Device `
            --results-dir $resultDir
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $($run.Name)"
        }
        Write-Host "DONE $($run.Name)"
    }
}
finally {
    Pop-Location
}

Write-Host "All nine joint input-encoding configurations are complete."
