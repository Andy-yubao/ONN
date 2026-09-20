"""Select compact hidden-state and readout widths without test-set tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.common.comparison_protocol import make_loaders, set_seed
from experiments.snn.conv_small.train import DEFAULT_DATA_DIR, frozen_t4_quantile_030_encoder
from model.snn.integer_reference import IntegerSNNReference, IntegerWidths
from model.snn.ptq import load_frozen_state


def _empty_metrics() -> dict:
    return {"saturations": {}, "ranges": {}, "threshold_margin": {}}


def _merge_metrics(total: dict, batch: dict) -> None:
    for name, count in batch["saturations"].items():
        total["saturations"][name] = total["saturations"].get(name, 0) + count
    for name, values in batch["ranges"].items():
        if name not in total["ranges"]:
            total["ranges"][name] = dict(values)
        else:
            total["ranges"][name]["min"] = min(total["ranges"][name]["min"], values["min"])
            total["ranges"][name]["max"] = max(total["ranges"][name]["max"], values["max"])
    for name, values in batch["threshold_margin"].items():
        if name not in total["threshold_margin"]:
            total["threshold_margin"][name] = dict(values)
        else:
            target = total["threshold_margin"][name]
            target["min"] = min(target["min"], values["min"])
            target["max"] = max(target["max"], values["max"])
            target["min_abs"] = min(target["min_abs"], values["min_abs"])
            target["abs_le_1"] += values["abs_le_1"]


def _configuration(guard_bits: int, readout_bits: int) -> dict:
    widths = IntegerWidths.for_guard_bits(guard_bits, readout_current=readout_bits)
    return {"guard_bits": guard_bits, "widths": vars(widths)}


@torch.inference_mode()
def compare_references(
    state: dict[str, torch.Tensor],
    loader: DataLoader,
    encoder,
    configurations: dict[str, tuple[int, int]],
    baseline_name: str,
) -> dict:
    references = {
        name: IntegerSNNReference(
            state,
            guard_bits=guard_bits,
            widths=IntegerWidths.for_guard_bits(
                guard_bits, readout_current=readout_bits
            ),
        )
        for name, (guard_bits, readout_bits) in configurations.items()
    }
    totals = {
        name: {
            "correct": 0,
            "prediction_mismatches_vs_baseline": 0,
            "spike_mismatches_vs_baseline": {"lif1": 0, "lif2": 0},
            "spike_values_compared": {"lif1": 0, "lif2": 0},
            "metrics": _empty_metrics(),
        }
        for name in configurations
    }
    sample_count = 0
    for images, labels in loader:
        times = encoder.first_spike_times(images)
        baseline_logits, baseline_trace, baseline_metrics = references[baseline_name].forward(
            times, return_trace=True, return_metrics=True
        )
        baseline_prediction = baseline_logits.argmax(1)
        batch_results = {
            baseline_name: (baseline_logits, baseline_trace, baseline_metrics)
        }
        for name, reference in references.items():
            if name != baseline_name:
                batch_results[name] = reference.forward(
                    times, return_trace=True, return_metrics=True
                )
        for name, (logits, trace, metrics) in batch_results.items():
            prediction = logits.argmax(1)
            totals[name]["correct"] += int((prediction == labels).sum().item())
            totals[name]["prediction_mismatches_vs_baseline"] += int(
                (prediction != baseline_prediction).sum().item()
            )
            for layer, key in (("lif1", "spikes1"), ("lif2", "spikes2")):
                totals[name]["spike_mismatches_vs_baseline"][layer] += sum(
                    int((left != right).sum().item())
                    for left, right in zip(trace[key], baseline_trace[key])
                )
                totals[name]["spike_values_compared"][layer] += sum(
                    value.numel() for value in trace[key]
                )
            _merge_metrics(totals[name]["metrics"], metrics)
        sample_count += labels.numel()

    output = {"sample_count": sample_count, "variants": {}}
    for name, values in totals.items():
        output["variants"][name] = {
            "accuracy_percent": 100.0 * values["correct"] / sample_count,
            "prediction_mismatches_vs_baseline": values[
                "prediction_mismatches_vs_baseline"
            ],
            "spike_mismatches_vs_baseline": values["spike_mismatches_vs_baseline"],
            "spike_values_compared": values["spike_values_compared"],
            "metrics": values["metrics"],
            "quantization": references[name].quantization_metadata(),
        }
    return output


@torch.inference_mode()
def collect_calibration(
    reference: IntegerSNNReference,
    loader: DataLoader,
    encoder,
    sample_limit: int,
) -> dict:
    total = _empty_metrics()
    seen = 0
    for images, _labels in loader:
        if seen >= sample_limit:
            break
        images = images[: sample_limit - seen]
        _logits, metrics = reference.forward(
            encoder.first_spike_times(images), return_metrics=True
        )
        _merge_metrics(total, metrics)
        seen += images.shape[0]
    return {"sample_count": seen, "metrics": total}


@torch.inference_mode()
def evaluate_test(reference, loader, encoder) -> float:
    correct = total = 0
    for images, labels in loader:
        prediction = reference.forward(encoder.first_spike_times(images)).argmax(1)
        correct += int((prediction == labels).sum().item())
        total += labels.numel()
    return 100.0 * correct / total


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _total_saturations(seed_results: dict, variant: str) -> int:
    return sum(
        sum(result["variants"][variant]["metrics"]["saturations"].values())
        for result in seed_results.values()
    )


def _total_prediction_mismatches(seed_results: dict, variant: str) -> int:
    return sum(
        result["variants"][variant]["prediction_mismatches_vs_baseline"]
        for result in seed_results.values()
    )


def _total_spike_mismatches(seed_results: dict, variant: str) -> int:
    return sum(
        sum(result["variants"][variant]["spike_mismatches_vs_baseline"].values())
        for result in seed_results.values()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Select formats without touching the test set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/state_quantization_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(17)
    train_loader, validation_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, 0
    )
    calibration_loader = DataLoader(
        train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    encoder, _encoder_metadata = frozen_t4_quantile_030_encoder()
    hidden_candidates = {
        "guard8": (8, 18),
        "guard4": (4, 18),
        "guard2": (2, 18),
    }
    results = {
        "selection_policy": {
            "calibration": "ordered training indices [0,1024)",
            "selection": "validation indices [55000,60000); test is not used for bit-width selection",
            "acceptance": "zero saturation, prediction mismatch, and hidden-spike mismatch on validation vs guard8",
            "test": "run once for the selected final format",
        },
        "hidden_candidates": {
            name: _configuration(*configuration)
            for name, configuration in hidden_candidates.items()
        },
        "hidden_validation_by_seed": {},
    }

    states = {}
    for seed in (7, 17, 27):
        state, _digest = load_frozen_state(seed)
        states[seed] = state
        compared = compare_references(
            state, validation_loader, encoder, hidden_candidates, "guard8"
        )
        results["hidden_validation_by_seed"][str(seed)] = compared
        print(f"hidden seed={seed}", {
            name: row["accuracy_percent"]
            for name, row in compared["variants"].items()
        }, flush=True)

    baseline_mean = _mean([
        row["variants"]["guard8"]["accuracy_percent"]
        for row in results["hidden_validation_by_seed"].values()
    ])
    hidden_aggregate = {}
    for name in hidden_candidates:
        mean_accuracy = _mean([
            row["variants"][name]["accuracy_percent"]
            for row in results["hidden_validation_by_seed"].values()
        ])
        hidden_aggregate[name] = {
            "mean_validation_accuracy_percent": mean_accuracy,
            "delta_vs_guard8_pct": mean_accuracy - baseline_mean,
            "total_validation_saturations": _total_saturations(
                results["hidden_validation_by_seed"], name
            ),
            "total_prediction_mismatches_vs_guard8": _total_prediction_mismatches(
                results["hidden_validation_by_seed"], name
            ),
            "total_spike_mismatches_vs_guard8": _total_spike_mismatches(
                results["hidden_validation_by_seed"], name
            ),
        }
    results["hidden_validation_aggregate"] = hidden_aggregate

    selected_hidden = "guard8"
    for name in ("guard2", "guard4"):
        row = hidden_aggregate[name]
        if (
            row["total_validation_saturations"] == 0
            and row["total_prediction_mismatches_vs_guard8"] == 0
            and row["total_spike_mismatches_vs_guard8"] == 0
        ):
            selected_hidden = name
            break
    selected_guard = hidden_candidates[selected_hidden][0]
    results["selected_hidden"] = selected_hidden

    readout_candidates = {
        f"{selected_hidden}_readout18": (selected_guard, 18),
        f"{selected_hidden}_readout17": (selected_guard, 17),
    }
    readout_baseline = f"{selected_hidden}_readout18"
    results["readout_candidates"] = {
        name: _configuration(*configuration)
        for name, configuration in readout_candidates.items()
    }
    results["readout_validation_by_seed"] = {}
    for seed in (7, 17, 27):
        compared = compare_references(
            states[seed], validation_loader, encoder, readout_candidates, readout_baseline
        )
        results["readout_validation_by_seed"][str(seed)] = compared
        print(f"readout seed={seed}", {
            name: row["accuracy_percent"]
            for name, row in compared["variants"].items()
        }, flush=True)

    compact_readout = f"{selected_hidden}_readout17"
    readout18_mean = _mean([
        row["variants"][readout_baseline]["accuracy_percent"]
        for row in results["readout_validation_by_seed"].values()
    ])
    readout17_mean = _mean([
        row["variants"][compact_readout]["accuracy_percent"]
        for row in results["readout_validation_by_seed"].values()
    ])
    readout17_saturations = _total_saturations(
        results["readout_validation_by_seed"], compact_readout
    )
    results["readout_validation_aggregate"] = {
        readout_baseline: {
            "mean_validation_accuracy_percent": readout18_mean,
            "delta_vs_readout18_pct": 0.0,
            "total_validation_saturations": _total_saturations(
                results["readout_validation_by_seed"], readout_baseline
            ),
            "total_prediction_mismatches_vs_readout18": 0,
        },
        compact_readout: {
            "mean_validation_accuracy_percent": readout17_mean,
            "delta_vs_readout18_pct": readout17_mean - readout18_mean,
            "total_validation_saturations": readout17_saturations,
            "total_prediction_mismatches_vs_readout18": _total_prediction_mismatches(
                results["readout_validation_by_seed"], compact_readout
            ),
        },
    }
    selected_readout = 17 if (
        readout17_saturations == 0
        and _total_prediction_mismatches(
            results["readout_validation_by_seed"], compact_readout
        ) == 0
    ) else 18
    results["selected_final_format"] = _configuration(selected_guard, selected_readout)
    results["selected_final_name"] = f"{selected_hidden}_readout{selected_readout}"

    if args.validation_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "selected": results["selected_final_format"],
            "test": "not run (--validation-only)",
        }, indent=2), flush=True)
        return

    results["calibration_by_seed"] = {}
    results["final_test_by_seed"] = {}
    for seed in (7, 17, 27):
        reference = IntegerSNNReference(
            states[seed],
            guard_bits=selected_guard,
            widths=IntegerWidths.for_guard_bits(
                selected_guard, readout_current=selected_readout
            ),
        )
        results["calibration_by_seed"][str(seed)] = collect_calibration(
            reference, calibration_loader, encoder, args.calibration_samples
        )
        accuracy = evaluate_test(reference, test_loader, encoder)
        results["final_test_by_seed"][str(seed)] = {
            "accuracy_percent": accuracy,
            "quantization": reference.quantization_metadata(),
        }
        print(f"final test seed={seed} accuracy={accuracy:.2f}%", flush=True)

    test_values = [
        row["accuracy_percent"] for row in results["final_test_by_seed"].values()
    ]
    results["final_test_aggregate"] = {
        "mean_accuracy_percent": _mean(test_values),
        "sample_std_pct": float(torch.tensor(test_values, dtype=torch.float64).std().item()),
        "delta_vs_fp32_mean_pct": _mean(test_values) - 93.46666666666665,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected": results["selected_final_format"],
        "test": results["final_test_aggregate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
