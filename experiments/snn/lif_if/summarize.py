"""Aggregate validation-selected LIF runs and the frozen IF reference."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = load(REPO / "experiments/snn/conv_small/results/final_three_seed/summary.json")
    screening = {}
    for decay, suffix in ((0.75, "0p75"), (0.9, "0p9"), (0.95, "0p95")):
        item = load(HERE / f"results/decay_{suffix}_seed17/metrics.json")
        screening[str(decay)] = {
            "best_validation_accuracy_percent": item["best_validation_accuracy_percent"],
            "validation_effective_synaptic_additions_per_image": item[
                "workload_validation_per_sample"
            ]["effective_synaptic_additions"],
            "test_evaluations_during_screening": item["test_evaluations"],
        }
    selected = max(screening, key=lambda decay: screening[decay]["best_validation_accuracy_percent"])
    if selected != "0.9":
        raise RuntimeError("Expected the recorded validation selection to be decay=0.9")
    runs = {}
    for seed in (7, 17, 27):
        folder = HERE / f"results/decay_0p9_seed{seed}"
        validation = load(folder / "metrics.json")
        final_test = load(folder / "final_test.json")
        if validation["test_evaluations"] != 0 or final_test["test_evaluations"] != 1:
            raise RuntimeError(f"Invalid test evaluation record for seed {seed}")
        work = final_test["workload_test_per_sample"]
        runs[str(seed)] = {
            "best_epoch": validation["best_epoch"],
            "best_validation_accuracy_percent": validation["best_validation_accuracy_percent"],
            "test_accuracy_percent": final_test["test_accuracy_percent"],
            "effective_synaptic_additions_per_image": work["effective_synaptic_additions"],
            "input_spikes_per_image": work["input_spike_events"],
            "layer1_spikes_per_image": work["layer1_spike_events"],
            "layer2_spikes_per_image": work["layer2_spike_events"],
            "hidden_spikes_per_image": work["layer1_spike_events"] + work["layer2_spike_events"],
            "checkpoint_sha256": final_test["checkpoint_sha256"],
        }
    values = list(runs.values())
    avg = lambda field: statistics.mean(run[field] for run in values)
    snn = baseline["aggregate"]
    result = {
        "experiment": "8x8 frozen conv_small IF versus hidden-membrane LIF",
        "screening_seed": 17,
        "screening": screening,
        "selected_decay_by_validation_accuracy": float(selected),
        "selected_seeds": [7, 17, 27],
        "test_evaluations_per_selected_seed": 1,
        "parameters_each": 9872,
        "lif_runs": runs,
        "lif_aggregate": {
            "best_validation_accuracy_mean_percent": avg("best_validation_accuracy_percent"),
            "best_validation_accuracy_sample_std_pct_points": statistics.stdev(
                run["best_validation_accuracy_percent"] for run in values
            ),
            "test_accuracy_mean_percent": avg("test_accuracy_percent"),
            "test_accuracy_sample_std_pct_points": statistics.stdev(
                run["test_accuracy_percent"] for run in values
            ),
            "hidden_spikes_per_image_mean": avg("hidden_spikes_per_image"),
            "effective_synaptic_additions_per_image_mean": avg(
                "effective_synaptic_additions_per_image"
            ),
        },
        "if_frozen_reference": {
            "source": "experiments/snn/conv_small/results/final_three_seed/summary.json",
            "best_validation_accuracy_mean_percent": snn["best_validation_accuracy_mean_percent"],
            "test_accuracy_mean_percent": snn["test_accuracy_mean_percent"],
            "test_accuracy_sample_std_pct_points": snn["test_accuracy_sample_std_pct_points"],
            "hidden_spikes_per_image_mean": snn["hidden_spikes_per_image_mean"],
            "effective_synaptic_additions_per_image_mean": snn[
                "effective_synaptic_additions_per_image_mean"
            ],
        },
        "lif_minus_if_test_accuracy_pct_points": avg("test_accuracy_percent")
        - snn["test_accuracy_mean_percent"],
        "lif_minus_if_effective_synaptic_additions_per_image": avg(
            "effective_synaptic_additions_per_image"
        ) - snn["effective_synaptic_additions_per_image_mean"],
    }
    output = HERE / "results/summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
