"""Evaluate best checkpoints on test set."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from torch.utils.data import DataLoader
from onn_model.data import get_test_loader
from onn_model.engine import _get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

models_to_check = [
    ('MicroCNNSmall', 'MicroCNNSmall'),
    ('MicroCNNExtraSmall', 'MicroCNNExtraSmall'),
    ('DepthwiseMicroCNN', 'DepthwiseMicroCNN'),
]

results = []
for display_name, model_name in models_to_check:
    print(f'\n--- {display_name} ---')
    runs_dir = f'model/runs/{model_name}'
    runs = sorted(os.listdir(runs_dir))
    latest = os.path.join(runs_dir, runs[-1])
    best_path = os.path.join(latest, 'best.pt')
    config_path = os.path.join(latest, 'config.json')

    with open(config_path) as f:
        config = json.load(f)

    model = _get_model(model_name).to(device)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    loader = get_test_loader(root=config.get('data_root', 'model/data'), batch_size=128,
                             num_workers=config.get('num_workers', 0))

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

    test_acc = total_correct / total_samples
    test_loss = total_loss / total_samples
    print(f'Test Acc: {test_acc:.4f}  Test Loss: {test_loss:.4f}')

    results.append({
        'model': display_name,
        'test_acc': test_acc,
        'test_loss': test_loss,
    })

print('\n\nSummary:')
for r in results:
    print(f"{r['model']:<25} Test Acc: {r['test_acc']:.4f}  Test Loss: {r['test_loss']:.4f}")

os.makedirs('experiments/model_compaction/m2_compact_models', exist_ok=True)
with open('experiments/model_compaction/m2_compact_models/single_seed_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved.')
