# M1-B 基线训练全流程脚本
# 在 PowerShell 中运行：
#   .\model\scripts\run_m1_baselines.ps1
#
# 如需 GPU，确保 conda env 中 torch 为 CUDA 版本。

Write-Host "=== M1-B 基线训练全流程 ===" -ForegroundColor Cyan

# ---------- 0. 激活环境 ----------
Write-Host "[1/7] 激活 Conda 环境..." -ForegroundColor Yellow
conda activate onn

# ---------- 1. 安装测试依赖 ----------
Write-Host "[2/7] 运行 pytest..." -ForegroundColor Yellow
python -m pytest model/tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ 测试失败，请检查后重试" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 测试通过" -ForegroundColor Green

# ---------- 2. Smoke test ----------
Write-Host "[3/7] BaselineCNN 烟雾测试..." -ForegroundColor Yellow
python model/train.py --config model/configs/baseline_cnn.json --smoke-test
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ BaselineCNN 烟雾测试失败" -ForegroundColor Red
    exit 1
}

Write-Host "[4/7] Tiny-ResNet 烟雾测试..." -ForegroundColor Yellow
python model/train.py --config model/configs/tiny_resnet.json --smoke-test
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Tiny-ResNet 烟雾测试失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 烟雾测试通过" -ForegroundColor Green

# ---------- 3. 完整训练 BaselineCNN ----------
Write-Host "[5/7] BaselineCNN 完整训练（15 epoch）..." -ForegroundColor Yellow
python model/train.py --config model/configs/baseline_cnn.json
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ BaselineCNN 训练失败" -ForegroundColor Red
    exit 1
}

# ---------- 4. 完整训练 Tiny-ResNet ----------
Write-Host "[6/7] Tiny-ResNet 完整训练（20 epoch）..." -ForegroundColor Yellow
python model/train.py --config model/configs/tiny_resnet.json
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Tiny-ResNet 训练失败" -ForegroundColor Red
    exit 1
}

# ---------- 5. 模型分析与活动统计 ----------
Write-Host "[7/7] 模型规模分析..." -ForegroundColor Yellow
python model/profile_model.py --model BaselineCNN --markdown
python model/profile_model.py --model TinyResNet --markdown
# 注意：活动分析需要手动指定 checkpoints 路径

Write-Host "=== M1-B 全流程完成 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：运行 analyze_activations.py 分析训练好的模型活动" -ForegroundColor White
