"""Profile all five models and save results."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onn_model.profiling import profile_model
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.models.tiny_resnet import TinyResNet
from onn_model.models.compact_cnn import MicroCNNSmall, MicroCNNExtraSmall, DepthwiseMicroCNN

models = {
    'BaselineCNN': BaselineCNN(),
    'TinyResNet': TinyResNet(),
    'MicroCNNSmall': MicroCNNSmall(),
    'MicroCNNExtraSmall': MicroCNNExtraSmall(),
    'DepthwiseMicroCNN': DepthwiseMicroCNN(),
}

all_profiles = {}
for name, model in models.items():
    print(f'Profiling {name}...')
    result = profile_model(model, (1, 1, 28, 28))
    all_profiles[name] = result

os.makedirs('model/profiles', exist_ok=True)
fname_map = {
    'BaselineCNN': 'baseline_cnn_profile.json',
    'TinyResNet': 'tiny_resnet_profile.json',
    'MicroCNNSmall': 'micro_cnn_s_profile.json',
    'MicroCNNExtraSmall': 'micro_cnn_xs_profile.json',
    'DepthwiseMicroCNN': 'ds_micro_cnn_profile.json',
}
for name, result in all_profiles.items():
    with open(f'model/profiles/{fname_map[name]}', 'w') as f:
        json.dump(result, f, indent=2)

print()
header = f"{'Model':<25} {'Params':>10} {'MACs':>12} {'W8 bytes':>10} {'A8 peak':>10} {'PingPong':>10}"
print('=' * len(header))
print(header)
print('=' * len(header))
for name in ['BaselineCNN', 'TinyResNet', 'MicroCNNSmall', 'MicroCNNExtraSmall', 'DepthwiseMicroCNN']:
    r = all_profiles[name]
    fp = r['fpga_proxy']
    print(f'{name:<25} {r["total_params"]:>10,} {r["total_macs"]:>12,} {fp["total_weight_bytes_int8"]:>10} {fp["peak_activation_bytes_int8"]:>10} {fp["estimated_ping_pong_activation_bytes_int8"]:>10}')
print('=' * len(header))

os.makedirs('experiments/model_compaction/m2_compact_models', exist_ok=True)
summary = {}
for name in models:
    r = all_profiles[name]
    fp = r['fpga_proxy']
    summary[name] = {
        'params': r['total_params'],
        'trainable': r['trainable_params'],
        'macs': r['total_macs'],
        'w8_bytes': fp['total_weight_bytes_int8'],
        'w16_bytes': fp['total_weight_bytes_int16'],
        'a8_peak': fp['peak_activation_bytes_int8'],
        'a8_pingpong': fp['estimated_ping_pong_activation_bytes_int8'],
    }
with open('experiments/model_compaction/m2_compact_models/model_profiles.json', 'w') as f:
    json.dump({'profiles': all_profiles, 'summary_table': summary}, f, indent=2)
print('\nCombined profile saved.')
