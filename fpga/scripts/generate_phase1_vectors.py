"""Generate deterministic IF and Conv1+IF1 vectors from the shared reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from experiments.snn.conv_small.train import frozen_t4_quantile_030_encoder
from model.snn.export_params import (
    DEFAULT_EXPORT_DIR,
    load_deployment_state,
    read_int8_mem,
)
from model.snn.integer_reference import IntegerSNNReference, IntegerWidths
from model.snn.quantization import saturate_signed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "fpga" / "vectors"
VALIDATION_INDICES = (55_000, 55_001, 55_002, 55_003)


def _make_reference() -> IntegerSNNReference:
    state, _ = load_deployment_state()
    widths = IntegerWidths.for_guard_bits(4, readout_current=17)
    reference = IntegerSNNReference(state, guard_bits=4, widths=widths)
    exports = (
        ("conv1.weight", "conv1_weight.mem", (16, 1, 3, 3)),
        ("conv2.weight", "conv2_weight.mem", (32, 16, 3, 3)),
        ("readout.weight", "readout_weight.mem", (10, 512)),
    )
    for key, filename, shape in exports:
        exported = read_int8_mem(DEFAULT_EXPORT_DIR / filename, shape)
        if not torch.equal(exported, reference.weights[key].to(torch.int8)):
            raise RuntimeError(
                f"{filename} differs from the shared integer reference"
            )
    return reference


def _write_if_vectors(path: Path) -> int:
    directed = [
        (0, 0),
        (123, -456),
        (-321, 654),
        (939, 0),
        (940, 0),
        (938, 0),
        (1, 938),
        (2, 938),
        (-1, 940),
        (40_000, 0),
        (-40_000, 0),
        (32_767, 131_071),
        (-32_768, -131_072),
        (32_767, 100_000),
        (-32_768, -100_000),
        (2_000, 500),
    ]
    generator = torch.Generator().manual_seed(17)
    random_current = torch.randint(-60_000, 60_001, (128,), generator=generator)
    random_membrane = torch.randint(-131_072, 131_072, (128,), generator=generator)
    inputs = directed + list(zip(random_current.tolist(), random_membrane.tolist()))
    lines = [str(len(inputs))]
    for current, membrane in inputs:
        current_tensor = torch.tensor([current], dtype=torch.int64)
        membrane_tensor = torch.tensor([membrane], dtype=torch.int64)
        spikes, post, integrated = IntegerSNNReference._if_step(
            current_tensor, membrane_tensor, 939, 16, 18
        )
        current_sat = int(saturate_signed(current_tensor, 16).item())
        lines.append(
            f"{current} {membrane} {current_sat} {int(integrated.item())} "
            f"{int(post.item())} {int(spikes.item())}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return len(inputs)


def _pack_spikes(spikes: torch.Tensor) -> int:
    flat = spikes.to(torch.bool).reshape(-1)
    return sum(int(value) << index for index, value in enumerate(flat.tolist()))


def _artificial_times(kind: str) -> torch.Tensor:
    times = torch.full((1, 1, 8, 8), -1, dtype=torch.long)
    if kind == "single_pixel":
        times[0, 0, 3, 4] = 0
    elif kind == "multi_pattern":
        for y, x in ((0, 0), (0, 7), (2, 3), (3, 2), (3, 3), (4, 4), (7, 0), (7, 7)):
            times[0, 0, y, x] = 0
    elif kind != "all_zero":
        raise ValueError(kind)
    return times


def _load_validation_times() -> tuple[list[torch.Tensor], list[int]]:
    transform = transforms.Compose([
        transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(ROOT / "data", train=True, download=False, transform=transform)
    encoder, _ = frozen_t4_quantile_030_encoder()
    times = []
    labels = []
    for index in VALIDATION_INDICES:
        image, label = dataset[index]
        times.append(encoder.first_spike_times(image.unsqueeze(0)))
        labels.append(int(label))
    return times, labels


def _append_case(
    cases: list[dict],
    reference: IntegerSNNReference,
    name: str,
    input_spikes: torch.Tensor,
    membrane: torch.Tensor | None,
    *,
    frame_start: bool,
    source: dict,
) -> torch.Tensor:
    current, integrated, post, spikes = reference.conv1_if1_step(
        input_spikes, membrane
    )
    cases.append({
        "name": name,
        "frame_start": frame_start,
        "spike_word": _pack_spikes(input_spikes),
        "current": current.reshape(-1).tolist(),
        "integrated": integrated.reshape(-1).tolist(),
        "post_reset": post.reshape(-1).tolist(),
        "spikes": spikes.to(torch.int64).reshape(-1).tolist(),
        "source": source,
    })
    return post


def _write_conv_vectors(
    input_path: Path,
    expected_path: Path,
    manifest_path: Path,
    reference: IntegerSNNReference,
) -> int:
    cases: list[dict] = []
    for kind in ("all_zero", "single_pixel", "multi_pattern"):
        times = _artificial_times(kind)
        _append_case(
            cases,
            reference,
            kind,
            times == 0,
            None,
            frame_start=True,
            source={"kind": "artificial", "timestep": 0},
        )

    validation_times, labels = _load_validation_times()
    for offset, step in enumerate((0, 1, 2)):
        _append_case(
            cases,
            reference,
            f"mnist_val_{VALIDATION_INDICES[offset]}_t{step}",
            validation_times[offset] == step,
            None,
            frame_start=True,
            source={
                "kind": "MNIST validation",
                "train_index": VALIDATION_INDICES[offset],
                "label": labels[offset],
                "timestep": step,
                "state": "zeroed for isolated-timestep verification",
            },
        )

    membrane = None
    for step in range(4):
        membrane = _append_case(
            cases,
            reference,
            f"mnist_val_{VALIDATION_INDICES[3]}_stateful_t{step}",
            validation_times[3] == step,
            membrane,
            frame_start=(step == 0),
            source={
                "kind": "MNIST validation",
                "train_index": VALIDATION_INDICES[3],
                "label": labels[3],
                "timestep": step,
                "state": "carried from preceding timestep",
            },
        )

    input_lines = [str(len(cases))]
    expected_lines = []
    manifest_cases = []
    for case_index, case in enumerate(cases):
        input_lines.append(
            f"{case_index} {int(case['frame_start'])} {case['spike_word']:016X}"
        )
        for element_index, values in enumerate(zip(
            case["current"],
            case["integrated"],
            case["post_reset"],
            case["spikes"],
        )):
            current, integrated, post, spike = values
            expected_lines.append(
                f"{case_index} {element_index} {current} {integrated} {post} {spike}"
            )
        manifest_cases.append({
            "case_id": case_index,
            "name": case["name"],
            "frame_start": case["frame_start"],
            "spike_word_hex": f"{case['spike_word']:016X}",
            "input_spike_count": case["spike_word"].bit_count(),
            "output_spike_count": sum(case["spikes"]),
            "source": case["source"],
        })

    input_path.write_text("\n".join(input_lines) + "\n", encoding="ascii", newline="\n")
    expected_path.write_text(
        "\n".join(expected_lines) + "\n", encoding="ascii", newline="\n"
    )
    manifest_path.write_text(
        json.dumps({
            "format_version": 1,
            "element_order": "channel*64 + y*8 + x (PyTorch contiguous [16,8,8])",
            "input_bit_order": "bit y*8+x is input pixel [y,x]",
            "case_count": len(cases),
            "elements_per_case": 1024,
            "cases": manifest_cases,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(cases)


def _pack_flat_bits(values: torch.Tensor) -> int:
    flat = values.to(torch.bool).reshape(-1)
    return sum(int(value) << index for index, value in enumerate(flat.tolist()))


def _write_phase2_vectors(
    output_dir: Path,
    reference: IntegerSNNReference,
    validation_times: list[torch.Tensor],
    labels: list[int],
) -> int:
    cases: list[dict] = []

    def append(name: str, spikes1: torch.Tensor, membrane, frame_start: bool, source: dict):
        current, integrated, post, spikes2 = reference.conv2_if2_step(spikes1, membrane)
        cases.append({
            "name": name, "frame_start": frame_start,
            "input": _pack_flat_bits(spikes1), "current": current.reshape(-1).tolist(),
            "integrated": integrated.reshape(-1).tolist(),
            "post": post.reshape(-1).tolist(),
            "spikes": spikes2.to(torch.int64).reshape(-1).tolist(), "source": source,
        })
        return post

    zero = torch.zeros((1, 16, 8, 8), dtype=torch.bool)
    append("all_zero", zero, None, True, {"kind": "artificial"})
    single = zero.clone(); single[0, 3, 4, 5] = True
    append("single_spike", single, None, True, {"kind": "artificial"})
    multi = zero.clone()
    for c, y, x in ((0, 0, 0), (1, 7, 7), (4, 3, 2), (8, 4, 4), (15, 1, 6)):
        multi[0, c, y, x] = True
    append("multi_spike", multi, None, True, {"kind": "artificial"})

    mem1 = None
    mem2 = None
    real = validation_times[3]
    for step in range(4):
        _, _, mem1, spikes1 = reference.conv1_if1_step(real == step, mem1)
        mem2 = append(
            f"mnist_val_{VALIDATION_INDICES[3]}_stateful_t{step}", spikes1, mem2,
            step == 0,
            {"kind": "MNIST validation", "train_index": VALIDATION_INDICES[3],
             "label": labels[3], "timestep": step},
        )

    inputs = [str(len(cases))]
    expected = []
    manifest = []
    for case_id, case in enumerate(cases):
        inputs.append(f"{case_id} {int(case['frame_start'])} {case['input']:0256X}")
        for index, values in enumerate(zip(
            case["current"], case["integrated"], case["post"], case["spikes"]
        )):
            current, integrated, post, spike = values
            expected.append(
                f"{case_id} {index} {current} {integrated} {post} {spike}"
            )
        manifest.append({
            "case_id": case_id, "name": case["name"],
            "frame_start": case["frame_start"],
            "input_spike_count": case["input"].bit_count(),
            "output_spike_count": sum(case["spikes"]), "source": case["source"],
        })
    (output_dir / "conv2_inputs.txt").write_text(
        "\n".join(inputs) + "\n", encoding="ascii", newline="\n"
    )
    (output_dir / "conv2_expected.txt").write_text(
        "\n".join(expected) + "\n", encoding="ascii", newline="\n"
    )
    (output_dir / "conv2_manifest.json").write_text(
        json.dumps({"case_count": len(cases), "elements_per_case": 512,
                    "element_order": "channel*16+y*4+x", "cases": manifest},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return len(cases)


def _write_readout_vectors(
    output_dir: Path,
    reference: IntegerSNNReference,
    validation_times: list[torch.Tensor],
    labels: list[int],
) -> int:
    cases: list[dict] = []

    def append(name: str, spikes2: torch.Tensor, scores, step: int,
               frame_start: bool, source: dict):
        current, next_scores = reference.readout_step(
            spikes2, scores, reference.temporal_coefficients[step]
        )
        cases.append({
            "name": name, "frame_start": frame_start, "step": step,
            "input": _pack_flat_bits(spikes2),
            "current": current.reshape(-1).tolist(),
            "scores": next_scores.reshape(-1).tolist(), "source": source,
        })
        return next_scores

    zero = torch.zeros((1, 32, 4, 4), dtype=torch.bool)
    append("all_zero", zero, None, 0, True, {"kind": "artificial"})
    single = zero.clone(); single[0, 7, 2, 1] = True
    append("single_spike", single, None, 0, True, {"kind": "artificial"})
    multi = zero.clone()
    for c, y, x in ((0, 0, 0), (3, 3, 3), (9, 1, 2), (31, 2, 1)):
        multi[0, c, y, x] = True
    append("multi_spike", multi, None, 0, True, {"kind": "artificial"})

    mem1 = None
    mem2 = None
    scores = None
    real = validation_times[3]
    for step in range(4):
        _, _, mem1, spikes1 = reference.conv1_if1_step(real == step, mem1)
        _, _, mem2, spikes2 = reference.conv2_if2_step(spikes1, mem2)
        scores = append(
            f"mnist_val_{VALIDATION_INDICES[3]}_stateful_t{step}", spikes2,
            scores, step, step == 0,
            {"kind": "MNIST validation", "train_index": VALIDATION_INDICES[3],
             "label": labels[3], "timestep": step},
        )

    inputs = [str(len(cases))]
    expected = []
    manifest = []
    for case_id, case in enumerate(cases):
        inputs.append(
            f"{case_id} {int(case['frame_start'])} {case['step']} {case['input']:0128X}"
        )
        for class_index, (current, score) in enumerate(
            zip(case["current"], case["scores"])
        ):
            expected.append(f"{case_id} {class_index} {current} {score}")
        manifest.append({
            "case_id": case_id, "name": case["name"], "step": case["step"],
            "frame_start": case["frame_start"],
            "input_spike_count": case["input"].bit_count(), "source": case["source"],
        })
    (output_dir / "readout_inputs.txt").write_text(
        "\n".join(inputs) + "\n", encoding="ascii", newline="\n"
    )
    (output_dir / "readout_expected.txt").write_text(
        "\n".join(expected) + "\n", encoding="ascii", newline="\n"
    )
    (output_dir / "readout_manifest.json").write_text(
        json.dumps({"case_count": len(cases), "classes_per_case": 10,
                    "cases": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return len(cases)


def _write_core_vectors(
    output_dir: Path,
    reference: IntegerSNNReference,
    validation_times: list[torch.Tensor],
    labels: list[int],
) -> int:
    inputs = []
    l1_lines = []
    l2_lines = []
    readout_lines = []
    sparse_lines = []
    frames = []
    step_id = 0
    random_generator = torch.Generator().manual_seed(1704)
    frame_sources = [
        (
            validation_times[offset],
            {"kind": "MNIST validation", "train_index": VALIDATION_INDICES[offset],
             "label": labels[offset]},
        )
        for offset in (2, 3)
    ]
    frame_sources.extend(
        (
            torch.randint(-1, 4, (1, 1, 8, 8), generator=random_generator),
            {"kind": "deterministic random latency map", "seed": 1704,
             "sequence": sequence},
        )
        for sequence in range(2)
    )
    for times, source in frame_sources:
        logits, trace = reference.forward(times, return_trace=True)
        frame_additions = 0
        frame_cycles = 0
        for step in range(4):
            input_spikes = times == step
            inputs.append(f"{step_id} {int(step == 0)} {_pack_spikes(input_spikes):016X}")
            for index, values in enumerate(zip(
                trace["conv1_current"][step].reshape(-1).tolist(),
                trace["mem1_integrated"][step].reshape(-1).tolist(),
                trace["mem1_post_reset"][step].reshape(-1).tolist(),
                trace["spikes1"][step].to(torch.int64).reshape(-1).tolist(),
            )):
                l1_lines.append(f"{step_id} {index} {' '.join(str(v) for v in values)}")
            for index, values in enumerate(zip(
                trace["conv2_current"][step].reshape(-1).tolist(),
                trace["mem2_integrated"][step].reshape(-1).tolist(),
                trace["mem2_post_reset"][step].reshape(-1).tolist(),
                trace["spikes2"][step].to(torch.int64).reshape(-1).tolist(),
            )):
                l2_lines.append(f"{step_id} {index} {' '.join(str(v) for v in values)}")
            for class_index, (current, score) in enumerate(zip(
                trace["readout_current"][step].reshape(-1).tolist(),
                trace["weighted_logits"][step].reshape(-1).tolist(),
            )):
                readout_lines.append(
                    f"{step_id} {class_index} {current} {score}"
                )
            conv1_additions = 0
            for _, _, input_y, input_x in torch.nonzero(input_spikes, as_tuple=False).tolist():
                valid_positions = sum(
                    0 <= input_y - kernel_y + 1 < 8 and
                    0 <= input_x - kernel_x + 1 < 8
                    for kernel_y in range(3) for kernel_x in range(3)
                )
                conv1_additions += 16 * valid_positions
            spikes1 = trace["spikes1"][step]
            conv2_additions = 0
            for _, _, input_y, input_x in torch.nonzero(spikes1, as_tuple=False).tolist():
                valid_positions = sum(
                    0 <= input_y - kernel_y + 1 <= 6 and
                    0 <= input_x - kernel_x + 1 <= 6 and
                    (input_y - kernel_y + 1) % 2 == 0 and
                    (input_x - kernel_x + 1) % 2 == 0
                    for kernel_y in range(3) for kernel_x in range(3)
                )
                conv2_additions += 32 * valid_positions
            spikes2 = trace["spikes2"][step]
            readout_additions = int(spikes2.sum().item()) * 10
            input_events = int(input_spikes.sum().item())
            layer1_events = int(spikes1.sum().item())
            layer2_events = int(spikes2.sum().item())
            conv1_cycles = 1025 + 145 * input_events + (1024 if step == 0 else 0)
            conv2_cycles = 545 + 289 * layer1_events + (512 if step == 0 else 0)
            readout_cycles = 27 + 11 * layer2_events + (10 if step == 0 else 0)
            total_additions = conv1_additions + conv2_additions + readout_additions
            total_cycles = conv1_cycles + conv2_cycles + readout_cycles
            frame_additions += total_additions
            frame_cycles += total_cycles
            sparse_lines.append(
                f"{step_id} {conv1_additions} {conv2_additions} {readout_additions} "
                f"{total_additions} {conv1_cycles} {conv2_cycles} {readout_cycles} "
                f"{total_cycles} {frame_additions} {frame_cycles}"
            )
            step_id += 1
        frames.append({
            "last_step_id": step_id - 1,
            "source": source,
            "expected_class": int(logits.argmax(1).item()),
            "logits": logits.reshape(-1).tolist(),
            "sparse_synaptic_additions": frame_additions,
            "sparse_compute_cycles": frame_cycles,
        })

    (output_dir / "core_inputs.txt").write_text(
        str(len(inputs)) + "\n" + "\n".join(inputs) + "\n",
        encoding="ascii", newline="\n",
    )
    for filename, lines in (
        ("core_l1_expected.txt", l1_lines),
        ("core_l2_expected.txt", l2_lines),
        ("core_readout_expected.txt", readout_lines),
        ("core_sparse_expected.txt", sparse_lines),
    ):
        (output_dir / filename).write_text(
            "\n".join(lines) + "\n", encoding="ascii", newline="\n"
        )
    (output_dir / "core_frames.txt").write_text(
        str(len(frames)) + "\n" + "\n".join(
            f"{frame['last_step_id']} {frame['expected_class']}" for frame in frames
        ) + "\n", encoding="ascii", newline="\n",
    )
    (output_dir / "core_manifest.json").write_text(
        json.dumps({"step_count": len(inputs), "frames": frames},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return len(inputs)


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _make_reference()
    if_count = _write_if_vectors(output_dir / "if_vectors.txt")
    conv_count = _write_conv_vectors(
        output_dir / "conv1_inputs.txt",
        output_dir / "conv1_expected.txt",
        output_dir / "manifest.json",
        reference,
    )
    validation_times, labels = _load_validation_times()
    conv2_count = _write_phase2_vectors(
        output_dir, reference, validation_times, labels
    )
    readout_count = _write_readout_vectors(
        output_dir, reference, validation_times, labels
    )
    core_steps = _write_core_vectors(
        output_dir, reference, validation_times, labels
    )
    return {
        "if_vectors": if_count,
        "conv1_cases": conv_count,
        "conv2_cases": conv2_count,
        "readout_cases": readout_count,
        "core_steps": core_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
