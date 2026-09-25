"""Audit completed software-only MNIST runs and summarize matched costs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEEDS = (7, 17, 27)
MODELS = ("cnn", "if", "snn")
TEST_SAMPLES = 10_000


def main() -> None:
    results = HERE / "results"
    summary: dict = {
        "protocol": "mnist28x28-sequential-v1",
        "seeds": list(SEEDS),
        "test_samples": TEST_SAMPLES,
        "operations_definition": {
            "dense_equivalent_macs_per_image": "All dense convolutions and readout; SNN sums four time steps.",
            "effective_synaptic_additions_per_image": (
                "SNN test_workload_total.effective_synaptic_additions / 10000; "
                "CNN not applicable. This is not equivalent to a dense MAC."
            ),
        },
        "test_interpretation": (
            "The decay=0.5 LIF and CNN tests were viewed before the beta/decay "
            "mistake was found; subsequent IF tests are post-test exploratory "
            "checks, not strictly independent held-out validation."
        ),
        "models": {},
    }
    source_hashes = set()
    calibrations = set()
    for name in MODELS:
        runs = []
        for seed in SEEDS:
            directory = results / f"{name}_seed{seed}"
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            config = metrics["config"]
            history = metrics["history"]
            assert len(history) == config["epochs"] == 20
            assert config["seed"] == seed and config["parameters"] == 20_432
            assert config["size"] == 28
            assert config["preprocessing"] == "ToTensor original 28x28"
            assert config["model"] == ("cnn" if name == "cnn" else "snn")
            if name != "cnn":
                assert config["membrane_decay"] == (1.0 if name == "if" else 0.5)
            assert metrics["test_evaluations"] == 1
            best = max(row["validation_accuracy"] for row in history)
            first = next(row["epoch"] for row in history if row["validation_accuracy"] == best)
            assert metrics["best_epoch"] == first
            assert metrics["best_validation_accuracy_percent"] == best
            checkpoint = directory / "best_model.pt"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
            source_hashes.add(json.dumps(config["source_sha256"], sort_keys=True))
            if name != "cnn":
                calibrations.add(json.dumps(config["encoder_calibration"], sort_keys=True))
            operations = metrics["test_workload_total"]
            additions = (operations["effective_synaptic_additions"] / TEST_SAMPLES
                         if name != "cnn" else None)
            runs.append({
                "seed": seed,
                "best_epoch": first,
                "validation_accuracy_percent": best,
                "test_accuracy_percent": metrics["final_test_accuracy_percent"],
                "parameters": config["parameters"],
                "dense_equivalent_macs_per_image": metrics["dense_macs_per_sample"],
                "effective_synaptic_additions_per_image": additions,
                "result_directory": str(directory.relative_to(REPO)).replace("\\", "/"),
            })
        summary["models"][name] = {
            "label": {"cnn": "matched CNN", "if": "IF SNN (decay=1.0)",
                      "snn": "exploratory LIF SNN (decay=0.5)"}[name],
            "runs": runs,
            "mean_validation_accuracy_percent": mean(run["validation_accuracy_percent"] for run in runs),
            "mean_test_accuracy_percent": mean(run["test_accuracy_percent"] for run in runs),
            "dense_equivalent_macs_per_image": runs[0]["dense_equivalent_macs_per_image"],
            "mean_effective_synaptic_additions_per_image": (
                mean(run["effective_synaptic_additions_per_image"] for run in runs)
                if name != "cnn" else None
            ),
        }
        assert len({run["dense_equivalent_macs_per_image"] for run in runs}) == 1
    assert len(source_hashes) == len(calibrations) == 1
    output = results / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"audited 9 runs; wrote {output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
