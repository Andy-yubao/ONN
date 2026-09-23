"""PC validation CLI for the Basys3 SNN UART link."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fpga.basys3.protocol import (
    PacketError, Result, STATUS_OK, decode_request, decode_result,
    encode_request, encode_result, read_packet,
)
from fpga.scripts.generate_regression_vectors import expected_frame, pack
from fpga.scripts.generate_phase1_vectors import ROOT, _make_reference


def mnist_case(index: int, split: str) -> tuple[tuple[int, int, int, int], int]:
    from torchvision import datasets, transforms
    from experiments.snn.conv_small.train import frozen_t4_quantile_030_encoder

    if split == "validation" and not 55000 <= index < 60000:
        raise ValueError("validation index must be in [55000, 60000)")
    if split == "train" and not 0 <= index < 55000:
        raise ValueError("training index must be in [0, 55000)")
    if split == "test" and not 0 <= index < 10000:
        raise ValueError("test index must be in [0, 10000)")
    transform = transforms.Compose([
        transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(ROOT / "data", train=split != "test", download=False,
                             transform=transform)
    image, label = dataset[index]
    encoder, _ = frozen_t4_quantile_030_encoder()
    times = encoder.first_spike_times(image.unsqueeze(0))
    return tuple(pack(times == step) for step in range(4)), int(label)


class MockTransport:
    """Packet-level stand-in using the same integer reference as the vector generator."""

    def __init__(self, reference):
        self.reference = reference
        self.response = b""

    def write(self, data: bytes) -> int:
        sequence, words = decode_request(data)
        expected = expected_frame(self.reference, words)
        last = expected["steps"][-1]
        self.response = encode_result(Result(
            sequence, STATUS_OK, expected["class"], tuple(last["scores"]),
            last["frame_add"], last["frame_cycles"],
        ))
        return len(data)

    def read(self, size: int) -> bytes:
        chunk, self.response = self.response[:size], self.response[size:]
        return chunk

    def close(self) -> None:
        pass


def validate_one(transport, reference, sequence: int,
                 words: tuple[int, int, int, int]) -> tuple[dict, Result | None]:
    expected = expected_frame(reference, words)
    last = expected["steps"][-1]
    try:
        request = encode_request(sequence, words)
        if transport.write(request) != len(request):
            raise OSError("short UART write")
        actual = decode_result(read_packet(transport))
        if actual.sequence != sequence:
            raise PacketError("response sequence mismatch")
        errors = []
        if actual.status != STATUS_OK:
            errors.append(f"FPGA status {actual.status}")
        if actual.prediction != expected["class"]:
            errors.append("prediction mismatch")
        if actual.scores != tuple(last["scores"]):
            errors.append("score mismatch")
        if actual.synaptic_additions != last["frame_add"]:
            errors.append("synaptic-add mismatch")
        if actual.compute_cycles != last["frame_cycles"]:
            errors.append("compute-cycle mismatch")
        return {"pass": not errors, "errors": errors}, actual
    except (PacketError, TimeoutError, OSError) as exc:
        return {"pass": False, "errors": [f"communication error: {exc}"],
                "communication_error": True}, None


def artificial_cases() -> list[tuple[str, tuple[int, int, int, int], int | None]]:
    return [
        ("all_zero", (0, 0, 0, 0), None),
        ("single_spike", (1 << 27, 0, 0, 0), None),
        ("corner_spike", (1, 0, 0, 0), None),
        ("dense", ((1 << 64) - 1, 0, 0, 0), None),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "serial"), default="mock")
    parser.add_argument("--port", help="USB-UART COM port, required for serial")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--level", choices=("artificial", "golden", "batch"), default="golden")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--indices", type=int, nargs="*", help="MNIST indices within the selected split")
    parser.add_argument("--output", type=Path, help="write JSON result")
    args = parser.parse_args()

    reference = _make_reference()
    if args.level == "artificial":
        samples = artificial_cases()
    else:
        start = 55000 if args.split == "validation" else 0
        indices = args.indices if args.indices is not None else (
            [start + 2, start + 3] if args.level == "golden"
            else list(range(start, start + 20))
        )
        samples = [(f"{args.split}_{index}", *mnist_case(index, args.split))
                   for index in indices]
    if args.backend == "mock":
        transport = MockTransport(reference)
    else:
        if not args.port:
            parser.error("--port is required for serial backend")
        try:
            import serial
        except ImportError as exc:
            parser.error(f"pyserial is required for serial backend: {exc}")
        transport = serial.Serial(args.port, args.baud, timeout=args.timeout,
                                  write_timeout=args.timeout)
        transport.reset_input_buffer()
    summary = {"backend": args.backend, "split": args.split if args.level != "artificial" else None,
               "total_samples": len(samples), "prediction_mismatch": 0,
               "score_mismatch": 0, "counter_mismatch": 0, "communication_error": 0,
               "fpga_correct": 0 if args.backend == "serial" else None,
               "mock_correct": 0 if args.backend == "mock" else None,
               "reference_correct": 0, "labeled_samples": 0,
               "cases": []}
    try:
        for number, (name, words, label) in enumerate(samples):
            checked, actual = validate_one(transport, reference, number % 256, words)
            golden = expected_frame(reference, words)
            errors = checked["errors"]
            summary["communication_error"] += int(
                checked.get("communication_error", False) or
                (actual is not None and actual.status != STATUS_OK)
            )
            summary["prediction_mismatch"] += int("prediction mismatch" in errors)
            summary["score_mismatch"] += int("score mismatch" in errors)
            summary["counter_mismatch"] += int(any("mismatch" in error and
                    ("add" in error or "cycle" in error) for error in errors))
            if label is not None:
                summary["labeled_samples"] += 1
                summary["reference_correct"] += int(golden["class"] == label)
                correct = int(actual is not None and actual.status == 0 and
                              actual.prediction == label)
                key = "fpga_correct" if args.backend == "serial" else "mock_correct"
                summary[key] += correct
            case = {"name": name, "label": label, "golden_class": golden["class"],
                    "result": checked, "fpga": asdict(actual) if actual else None}
            summary["cases"].append(case)
            print(f"{'PASS' if checked['pass'] else 'FAIL'} {name}: " +
                  ("; ".join(errors) if errors else f"class={golden['class']}"))
    finally:
        transport.close()
    count = summary["labeled_samples"]
    summary["fpga_accuracy"] = summary["fpga_correct"] / count if count and args.backend == "serial" else None
    summary["mock_accuracy"] = summary["mock_correct"] / count if count and args.backend == "mock" else None
    summary["reference_accuracy"] = summary["reference_correct"] / count if count else None
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    return 0 if all(case["result"]["pass"] for case in summary["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
