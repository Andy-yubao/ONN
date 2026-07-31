<#
.SYNOPSIS
    M2-A Compact Model Experiments - Automation Script
.DESCRIPTION
    Runs profiling, single-seed training, multi-seed training, and INT8 PTQ
    analysis for the three new compact models.  Does NOT commit or push.
.PARAMETER profile
    Profile all five models with FPGA resource proxy.
.PARAMETER singleSeed
    Run single-seed training for the three new models.
.PARAMETER multiseed
    Run multi-seed training for Pareto-selected candidates.
.PARAMETER quantize
    Run BN fusion and INT8 PTQ analysis.
.PARAMETER all
    Run all phases sequentially.
.PARAMETER skipExisting
    Skip phases that already have result files.
#>

param(
    [switch]$profile,
    [switch]$singleSeed,
    [switch]$multiseed,
    [switch]$quantize,
    [switch]$all,
    [switch]$skipExisting
)

if ($all) {
    $profile = $true
    $singleSeed = $true
    $multiseed = $true
    $quantize = $true
}

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$EXPERIMENT_DIR = Join-Path $ROOT "experiments/model_compaction/m2_compact_models"
$PYTHON = "python"

# Create experiment directory
New-Item -ItemType Directory -Force -Path $EXPERIMENT_DIR | Out-Null

function Write-Phase {
    param([string]$Phase)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $Phase" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Check-Exists {
    param([string]$Path)
    if ($skipExisting -and (Test-Path $Path)) {
        Write-Host "  [SKIP] $Path already exists." -ForegroundColor Yellow
        return $true
    }
    return $false
}

# ======================================================================
#  1. Profile all models
# ======================================================================
if ($profile) {
    Write-Phase "Phase 1: Profiling All Models"

    $MODELS = @("BaselineCNN", "TinyResNet", "MicroCNNSmall", "MicroCNNExtraSmall", "DepthwiseMicroCNN")
    $PROFILE_DIR = Join-Path $ROOT "model/profiles"
    New-Item -ItemType Directory -Force -Path $PROFILE_DIR | Out-Null

    foreach ($m in $MODELS) {
        $outFile = Join-Path $PROFILE_DIR "$($m.ToLower())_profile.json"
        if (Check-Exists $outFile) { continue }

        Write-Host "  Profiling $m..." -ForegroundColor Green
        & $PYTHON -m "$ROOT.model.profile_model" --model $m --output $outFile 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [WARN] Profile for $m failed (exit $LASTEXITCODE). Continuing..." -ForegroundColor Red
        }
    }

    # Generate combined profile table
    Write-Host "  Generating combined profile table..." -ForegroundColor Green
    & $PYTHON -c @"
import json, os, sys, glob
profiles = {}
profile_dir = r'$PROFILE_DIR'
for f in glob.glob(os.path.join(profile_dir, '*_profile.json')):
    with open(f) as fp:
        p = json.load(fp)
        profiles[p['model_name']] = p
table = {}
for name, p in sorted(profiles.items()):
    fp = p.get('fpga_proxy', {})
    table[name] = {
        'params': p['total_params'],
        'macs': p['total_macs'],
        'w8_bytes': fp.get('total_weight_bytes_int8', 0),
        'a8_peak': fp.get('peak_activation_bytes_int8', 0),
        'ping_pong_a8': fp.get('estimated_ping_pong_activation_bytes_int8', 0),
    }
out_path = r'$EXPERIMENT_DIR/model_profiles.json'
with open(out_path, 'w') as f:
    json.dump({'profiles': profiles, 'summary_table': table}, f, indent=2)
print(f'Profiles saved to {out_path}')
"@ 2>&1
    Write-Host "  [DONE] Profiling complete." -ForegroundColor Green
}

# ======================================================================
#  2. Single-seed training
# ======================================================================
if ($singleSeed) {
    Write-Phase "Phase 2: Single-Seed Training"

    $CONFIGS = @(
        @{Name="MicroCNN-S"; Config="micro_cnn_s.json"},
        @{Name="MicroCNN-XS"; Config="micro_cnn_xs.json"},
        @{Name="DS-MicroCNN"; Config="ds_micro_cnn.json"}
    )

    $SINGLE_RESULTS = @()
    foreach ($cfg in $CONFIGS) {
        Write-Host "  Training $($cfg.Name)..." -ForegroundColor Green
        $cfgPath = Join-Path $ROOT "model/configs/$($cfg.Config)"
        & $PYTHON "$ROOT/model/train.py" --config $cfgPath 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [WARN] Training for $($cfg.Name) failed. Continuing..." -ForegroundColor Red
            continue
        }
    }

    # Collect results
    Write-Host "  Collecting single-seed results..." -ForegroundColor Green
    & $PYTHON -c @"
import json, os, glob, sys
from pathlib import Path

runs_dir = os.path.join(r'$ROOT', 'model', 'runs')
results = []
for model_name in ['MicroCNNSmall', 'MicroCNNExtraSmall', 'DepthwiseMicroCNN']:
    model_dir = os.path.join(runs_dir, model_name)
    if not os.path.isdir(model_dir):
        continue
    # Find latest run
    runs = sorted(os.listdir(model_dir))
    if not runs:
        continue
    latest = os.path.join(model_dir, runs[-1])
    metrics_path = os.path.join(latest, 'metrics.json')
    config_path = os.path.join(latest, 'config.json')
    if os.path.exists(metrics_path) and os.path.exists(config_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(config_path) as f:
            cfg = json.load(f)
        results.append({
            'model': model_name,
            'run_dir': latest,
            'best_val_acc': metrics['best_val_acc'],
            'best_val_loss': metrics['best_val_loss'],
            'best_epoch': metrics['best_epoch'],
            'total_params': metrics['total_params'],
            'config': cfg,
        })

out_path = os.path.join(r'$EXPERIMENT_DIR', 'single_seed_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved {len(results)} results to {out_path}')
"@ 2>&1
    Write-Host "  [DONE] Single-seed training complete." -ForegroundColor Green
}

# ======================================================================
#  3. Multi-seed training
# ======================================================================
if ($multiseed) {
    Write-Phase "Phase 3: Multi-Seed Training"

    # Read Pareto analysis to determine candidates
    $PARETO_PATH = Join-Path $EXPERIMENT_DIR "pareto_analysis.json"
    if (Test-Path $PARETO_PATH) {
        $pareto = Get-Content $PARETO_PATH | ConvertFrom-Json
        $CANDIDATES = $pareto.selected_candidates
    } else {
        Write-Host "  [WARN] No Pareto analysis found. Using default: MicroCNNSmall, DepthwiseMicroCNN" -ForegroundColor Yellow
        $CANDIDATES = @("MicroCNNSmall", "DepthwiseMicroCNN")
    }

    Write-Host "  Candidates: $($CANDIDATES -join ', ')" -ForegroundColor Green

    $SEEDS = @(42, 43, 44)
    $MULTI_RESULTS = @()

    foreach ($candidate in $CANDIDATES) {
        foreach ($s in $SEEDS) {
            Write-Host "  Training $candidate (seed $s)..." -ForegroundColor Green

            # Find the corresponding config
            $CONFIG_MAP = @{
                "MicroCNNSmall" = "micro_cnn_s.json"
                "MicroCNNExtraSmall" = "micro_cnn_xs.json"
                "DepthwiseMicroCNN" = "ds_micro_cnn.json"
            }
            $cfgFile = $CONFIG_MAP[$candidate]
            if (-not $cfgFile) {
                Write-Host "  [WARN] Unknown candidate: $candidate" -ForegroundColor Red
                continue
            }
            $cfgPath = Join-Path $ROOT "model/configs/$cfgFile"

            & $PYTHON "$ROOT/model/train.py" --config $cfgPath --run-dir "$ROOT/model/runs/${candidate}_seed${s}" 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [WARN] Training $candidate seed $s failed." -ForegroundColor Red
            }
        }
    }

    # Collect multi-seed results
    & $PYTHON -c @"
import json, os, glob

runs_dir = os.path.join(r'$ROOT', 'model', 'runs')
results = []
for candidate in $($CANDIDATES | ConvertTo-Json):
    for seed in [42, 43, 44]:
        run_dir_pattern = os.path.join(runs_dir, f'{candidate}_seed{seed}')
        # Also check normal run dir
        alt_dir = os.path.join(runs_dir, candidate)
        for d in [run_dir_pattern, alt_dir]:
            if not os.path.isdir(d):
                continue
            metrics_path = os.path.join(d, 'metrics.json')
            config_path = os.path.join(d, 'config.json')
            if os.path.exists(metrics_path) and os.path.exists(config_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                with open(config_path) as f:
                    cfg = json.load(f)
                results.append({
                    'model': candidate,
                    'seed': seed,
                    'run_dir': d,
                    'best_val_acc': metrics['best_val_acc'],
                    'best_val_loss': metrics['best_val_loss'],
                    'best_epoch': metrics['best_epoch'],
                })

out_path = os.path.join(r'$EXPERIMENT_DIR', 'multiseed_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved {len(results)} results to {out_path}')
"@ 2>&1
    Write-Host "  [DONE] Multi-seed training complete." -ForegroundColor Green
}

# ======================================================================
#  4. Quantization analysis (BN fusion + INT8 PTQ)
# ======================================================================
if ($quantize) {
    Write-Phase "Phase 4: Quantization Analysis"

    & $PYTHON -c @"
import json, os, sys, glob
sys.path.insert(0, os.path.join(r'$ROOT'))
sys.path.insert(0, os.path.join(r'$ROOT', 'model'))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from onn_model.models.compact_cnn import MicroCNNSmall, MicroCNNExtraSmall, DepthwiseMicroCNN
from onn_model.engine import _get_model, run_experiment
from onn_model.quantization import fuse_model_bn, check_bn_fusion_error, evaluate_ptq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load checkpoint
def find_best_checkpoint(model_name):
    runs_dir = os.path.join(r'$ROOT', 'model', 'runs')
    model_dir = os.path.join(runs_dir, model_name)
    if not os.path.isdir(model_dir):
        return None, None
    runs = sorted(os.listdir(model_dir))
    if not runs:
        return None, None
    latest = os.path.join(model_dir, runs[-1])
    best = os.path.join(latest, 'best.pt')
    config = os.path.join(latest, 'config.json')
    if os.path.exists(best) and os.path.exists(config):
        with open(config) as f:
            cfg = json.load(f)
        return best, cfg
    return None, None

# Create a small calibration set
calib_images = torch.randn(100, 1, 28, 28)
calib_labels = torch.randint(0, 10, (100,))
calib_loader = DataLoader(TensorDataset(calib_images, calib_labels), batch_size=20)

test_images = torch.randn(200, 1, 28, 28)
test_labels = torch.randint(0, 10, (200,))
test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=20)

models_to_check = ['MicroCNNSmall', 'MicroCNNExtraSmall', 'DepthwiseMicroCNN']
quant_results = []

for model_name in models_to_check:
    print(f'\n--- {model_name} ---')
    model = _get_model(model_name).to(device)
    model.eval()

    # FP32 baseline (synthetic)
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    fp32_acc = correct / total if total > 0 else 0.0
    print(f'  FP32 (synthetic): {fp32_acc:.4f}')

    # BN fusion
    fused = fuse_model_bn(model)
    x_test = torch.randn(20, 1, 28, 28, device=device)
    err = check_bn_fusion_error(model, fused, x_test)
    print(f'  BN fusion max_error: {err["max_error"]:.6f}  OK={err["output_ok"]}')

    # Fused model accuracy
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = fused(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    fused_acc = correct / total if total > 0 else 0.0
    print(f'  BN-fused (synthetic): {fused_acc:.4f}')

    # INT8 PTQ
    ptq_result = evaluate_ptq(model, test_loader, device, calib_loader, calib_batches=3)
    print(f'  INT8 PTQ (synthetic): {ptq_result["accuracy"]:.4f}')

    quant_results.append({
        'model': model_name,
        'fp32_accuracy': fp32_acc,
        'bn_fusion_max_error': err['max_error'],
        'bn_fusion_error_ok': err['output_ok'],
        'bn_fused_accuracy': fused_acc,
        'int8_ptq_accuracy': ptq_result['accuracy'],
        'int8_ptq_loss': ptq_result['loss'],
        'weight_saturated_fractions': ptq_result['weight_saturated_fractions'],
        'device': str(device),
    })

out_path = os.path.join(r'$EXPERIMENT_DIR', 'quantization_results.json')
with open(out_path, 'w') as f:
    json.dump(quant_results, f, indent=2)
print(f'\nQuantization results saved to {out_path}')
"@ 2>&1
    Write-Host "  [DONE] Quantization analysis complete." -ForegroundColor Green
}

Write-Phase "All requested phases completed."
