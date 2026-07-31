"""Collect multi-seed results and evaluate on test set."""
import json, os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from onn_model.data import get_test_loader
from onn_model.engine import _get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def find_run(model_name, seed):
    """Find the run directory for a model and seed."""
    runs_dir = f'model/runs/{model_name}'
    if not os.path.isdir(runs_dir):
        return None
    # Sort by time to get latest
    runs = sorted(os.listdir(runs_dir))
    if not runs:
        return None
    # Find the run that matches the seed by checking config
    for run in reversed(runs):
        config_path = os.path.join(runs_dir, run, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get('run_seed') == seed or cfg.get('seed') == seed:
                return os.path.join(runs_dir, run)
    # Fall back to latest
    return os.path.join(runs_dir, runs[-1])

def evaluate(model, config):
    """Evaluate model on test set."""
    model.eval()

    loader = get_test_loader(
        root=config.get('data_root', 'model/data'),
        batch_size=128, num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).float().sum().item()
        total_samples += labels.size(0)

    return total_correct / total_samples, total_loss / total_samples

candidates = ['MicroCNNSmall', 'DepthwiseMicroCNN']
seeds = [42, 43, 44]
all_results = []

for model_name in candidates:
    for seed in seeds:
        run_dir = find_run(model_name, seed)
        if not run_dir:
            print(f'No run found for {model_name} seed {seed}')
            continue

        best_path = os.path.join(run_dir, 'best.pt')
        config_path = os.path.join(run_dir, 'config.json')

        if not os.path.exists(best_path):
            print(f'No best.pt for {model_name} seed {seed} at {run_dir}')
            continue

        with open(config_path) as f:
            config = json.load(f)

        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model = _get_model(model_name).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        test_acc, test_loss = evaluate(model, config)

        result = {
            'model': model_name,
            'run_seed': seed,
            'best_val_acc': ckpt.get('best_val_acc', 0),
            'best_val_loss': ckpt.get('best_val_loss', float('inf')),
            'best_epoch': ckpt.get('best_epoch', -1),
            'test_acc': test_acc,
            'test_loss': test_loss,
            'run_dir': run_dir,
        }
        all_results.append(result)
        print(f'{model_name:<20} seed={seed}: val_acc={result["best_val_acc"]:.4f} (ep{result["best_epoch"]})  test_acc={test_acc:.4f}')

# Save
out_dir = 'experiments/model_compaction/m2_compact_models'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'multiseed_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary
print(f'\n{"="*80}')
print(f'{"Model":<25} {"Seed":>5} {"Val Acc":>10} {"Test Acc":>10} {"Best Ep":>8}')
print(f'{"="*80}')
for r in all_results:
    print(f'{r["model"]:<25} {r["run_seed"]:>5} {r["best_val_acc"]:>10.4f} {r["test_acc"]:>10.4f} {r["best_epoch"]:>8}')

# Aggregate
from statistics import mean, stdev
for model_name in candidates:
    mr = [r for r in all_results if r['model'] == model_name]
    val_accs = [r['best_val_acc'] for r in mr]
    test_accs = [r['test_acc'] for r in mr]
    best_eps = [r['best_epoch'] for r in mr]
    print(f'\n{model_name}:')
    print(f'  Val Acc: {mean(val_accs):.4f} ± {stdev(val_accs):.4f}')
    print(f'  Test Acc: {mean(test_accs):.4f} ± {stdev(test_accs):.4f}')
    if len(best_eps) > 1:
        print(f'  Best Epoch: {mean(best_eps):.1f} ± {stdev(best_eps):.1f}')
