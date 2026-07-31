"""Multi-seed training for Pareto-selected candidates."""
import json, os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from onn_model.data import get_test_loader
from onn_model.engine import _get_model, run_experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

candidates = [
    ('MicroCNNSmall', 'model/configs/micro_cnn_s.json'),
    ('DepthwiseMicroCNN', 'model/configs/ds_micro_cnn.json'),
]

seeds = [42, 43, 44]
all_results = []

for model_name, config_path in candidates:
    with open(config_path) as f:
        base_config = json.load(f)

    for run_seed in seeds:
        print(f'\n{"="*60}')
        print(f'Training {model_name} with run_seed={run_seed}')
        print(f'{"="*60}')

        config = base_config.copy()
        config['run_seed'] = run_seed
        config['seed'] = run_seed  # legacy compat

        state = run_experiment(config)

        # Evaluate on test set
        test_loader = get_test_loader(
            root=config.get('data_root', 'model/data'),
            batch_size=config.get('batch_size', 128),
            num_workers=0,
        )

        model = _get_model(model_name).to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).float().sum().item()
            total_samples += labels.size(0)

        test_acc = total_correct / total_samples
        test_loss = total_loss / total_samples

        result = {
            'model': model_name,
            'run_seed': run_seed,
            'best_val_acc': state.best_val_acc,
            'best_val_loss': state.best_val_loss,
            'best_epoch': state.best_epoch,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'config': config,
        }
        all_results.append(result)
        print(f'Best Val Acc: {state.best_val_acc:.4f} (epoch {state.best_epoch})')
        print(f'Test Acc: {test_acc:.4f}')

# Save results
out_dir = 'experiments/model_compaction/m2_compact_models'
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, 'multiseed_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary table
print(f'\n\n{"="*80}')
print(f'{"Model":<25} {"Seed":>5} {"Val Acc":>10} {"Test Acc":>10} {"Best Ep":>8}')
print(f'{"="*80}')
for r in all_results:
    print(f'{r["model"]:<25} {r["run_seed"]:>5} {r["best_val_acc"]:>10.4f} {r["test_acc"]:>10.4f} {r["best_epoch"]:>8}')

# Aggregate stats
print(f'\n\n{"="*80}')
print('Aggregate (mean ± std)')
from statistics import mean, stdev
for model_name, _ in candidates:
    model_results = [r for r in all_results if r['model'] == model_name]
    val_accs = [r['best_val_acc'] for r in model_results]
    test_accs = [r['test_acc'] for r in model_results]
    best_eps = [r['best_epoch'] for r in model_results]

    def fmt(vals):
        if len(vals) > 1:
            return f'{mean(vals):.4f} ± {stdev(vals):.4f}'
        return f'{vals[0]:.4f}'

    print(f'{model_name:<25} Val: {fmt(val_accs):>15}  Test: {fmt(test_accs):>15}  Best Ep: {fmt(best_eps)}')

print(f'\nResults saved to {out_dir}')
