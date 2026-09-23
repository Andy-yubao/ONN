"""Compact, deterministic end-to-end vectors for the sparse RTL core."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms

from experiments.snn.conv_small.train import frozen_t4_quantile_030_encoder
from fpga.scripts.generate_phase1_vectors import DEFAULT_OUTPUT_DIR, ROOT, _make_reference


def words_to_steps(words: tuple[int, int, int, int]) -> torch.Tensor:
    return torch.tensor(
        [[[(word >> bit) & 1 for bit in range(64)] for word in words]],
        dtype=torch.int64,
    ).reshape(1, 4, 1, 8, 8)


def pack(values: torch.Tensor) -> int:
    return sum(int(value) << bit for bit, value in enumerate(values.reshape(-1).tolist()))


def expected_frame(reference, words: tuple[int, int, int, int]) -> dict:
    steps = words_to_steps(words)
    mem1 = mem2 = scores = None
    frame_add = frame_cycles = 0
    result_steps = []
    for t in range(4):
        input_spikes = steps[:, t]
        _, _, mem1, l1 = reference.conv1_if1_step(input_spikes, mem1)
        _, _, mem2, l2 = reference.conv2_if2_step(l1, mem2)
        _, scores = reference.readout_step(l2, scores, reference.temporal_coefficients[t])
        a1 = sum(
            16 * sum(0 <= y - ky + 1 < 8 and 0 <= x - kx + 1 < 8
                     for ky in range(3) for kx in range(3))
            for _, _, y, x in torch.nonzero(input_spikes, as_tuple=False).tolist()
        )
        a2 = sum(
            32 * sum(0 <= y - ky + 1 <= 6 and 0 <= x - kx + 1 <= 6
                     and (y - ky + 1) % 2 == 0 and (x - kx + 1) % 2 == 0
                     for ky in range(3) for kx in range(3))
            for _, _, y, x in torch.nonzero(l1, as_tuple=False).tolist()
        )
        additions = a1 + a2 + int(l2.sum().item()) * 10
        cycles = (1025 + 145 * int(input_spikes.sum().item()) + (1024 if t == 0 else 0)
                  + 545 + 289 * int(l1.sum().item()) + (512 if t == 0 else 0)
                  + 27 + 11 * int(l2.sum().item()) + (10 if t == 0 else 0))
        frame_add += additions
        frame_cycles += cycles
        result_steps.append({
            "input": words[t], "l1": pack(l1), "l2": pack(l2),
            "scores": scores.reshape(-1).tolist(),
            "step_add": additions, "step_cycles": cycles,
            "frame_add": frame_add, "frame_cycles": frame_cycles,
        })
    return {"class": int(scores.argmax(1).item()), "steps": result_steps}


def cases() -> list[tuple[str, tuple[int, int, int, int], dict]]:
    all_bits = (1 << 64) - 1
    selected = [
        ("all_zero", (0, 0, 0, 0), {"kind": "corner"}),
        ("single_spike", (1 << 27, 0, 0, 0), {"kind": "corner"}),
        ("corner_0", (1, 0, 0, 0), {"kind": "corner"}),
        ("corner_63", (0, 0, 0, 1 << 63), {"kind": "corner"}),
        ("center", (0, 1 << 36, 0, 0), {"kind": "corner"}),
        ("all_one_t0", (all_bits, 0, 0, 0), {"kind": "corner"}),
        ("repeated_pixel", (1 << 27,) * 4, {"kind": "corner"}),
        ("dense_each_step", (all_bits,) * 4, {"kind": "corner"}),
    ]
    transform = transforms.Compose([
        transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(ROOT / "data", train=True, download=False, transform=transform)
    encoder, _ = frozen_t4_quantile_030_encoder()
    # Training indices [55000,60000) are the held-out validation split.
    for index in range(55000, 55032):
        image, label = dataset[index]
        times = encoder.first_spike_times(image.unsqueeze(0))
        words = tuple(pack(times == t) for t in range(4))
        selected.append((f"mnist_val_{index}", words,
                         {"kind": "MNIST validation", "train_index": index, "label": int(label)}))
    rng = random.Random(170421)
    for index in range(8):
        # Independent per-step bitmaps also exercise repeated temporal firing.
        words = tuple(rng.getrandbits(64) & rng.getrandbits(64) for _ in range(4))
        selected.append((f"random_{index}", words,
                         {"kind": "random bitmap", "seed": 170421, "sequence": index}))
    return selected


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _make_reference()
    inputs = []
    outputs = []
    manifest = []
    for frame_id, (name, words, source) in enumerate(cases()):
        expected = expected_frame(reference, words)
        manifest.append({"frame_id": frame_id, "name": name, "source": source,
                         "input_words_hex": [f"{w:016X}" for w in words],
                         "class": expected["class"]})
        for t, step in enumerate(expected["steps"]):
            inputs.append(f"{frame_id * 4 + t} {int(t == 0)} {words[t]:016X}")
            outputs.append(
                f"{frame_id * 4 + t} {step['l1']:0256X} {step['l2']:0128X} "
                f"{step['step_add']} {step['step_cycles']} {step['frame_add']} "
                f"{step['frame_cycles']} {expected['class'] if t == 3 else 0}"
            )
            outputs.append(" ".join(str(v) for v in step["scores"]))
    (output_dir / "regression_inputs.txt").write_text(
        f"{len(inputs)}\n" + "\n".join(inputs) + "\n", encoding="ascii", newline="\n")
    (output_dir / "regression_expected.txt").write_text(
        "\n".join(outputs) + "\n", encoding="ascii", newline="\n")
    (output_dir / "regression_manifest.json").write_text(
        json.dumps({"format_version": 1, "split": "MNIST train indices 55000..59999 are validation",
                    "frames": manifest}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"frames": len(manifest), "steps": len(inputs)}


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
