"""UART waveform test vectors from the packet codec and integer reference."""

from __future__ import annotations

import json

from fpga.basys3.host import mnist_case
from fpga.basys3.protocol import Result, checksum, encode_request, encode_result
from fpga.scripts.generate_phase1_vectors import DEFAULT_OUTPUT_DIR, _make_reference
from fpga.scripts.generate_regression_vectors import expected_frame


def main() -> None:
    reference = _make_reference()
    zero = (0, 0, 0, 0)
    single = (1 << 27, 0, 0, 0)
    mnist_a, _ = mnist_case(55002, "validation")
    mnist_b, _ = mnist_case(55003, "validation")
    requests = [
        encode_request(10, zero),
        encode_request(11, single),  # sent while first frame is busy; deliberately dropped
        encode_request(12, mnist_a),
        encode_request(13, zero),
        encode_request(16, zero),  # valid checksum but invalid timestep index
        encode_request(14, mnist_b),  # follows a board reset
        encode_request(15, single),
    ]
    requests[3] = requests[3][:-1] + bytes((requests[3][-1] ^ 0x01,))
    malformed_index = bytearray(requests[4])
    malformed_index[6] = 7
    malformed_index[-1] = checksum(malformed_index[2:-1])
    requests[4] = bytes(malformed_index)
    responses = []
    for sequence, words in ((10, zero), (12, mnist_a), (14, mnist_b), (15, single)):
        frame = expected_frame(reference, words)
        last = frame["steps"][-1]
        responses.append(encode_result(Result(
            sequence, 0, frame["class"], tuple(last["scores"]),
            last["frame_add"], last["frame_cycles"])))
    responses.insert(2, encode_result(Result(13, 1, 0, (0,) * 10, 0, 0)))
    responses.insert(3, encode_result(Result(16, 2, 0, (0,) * 10, 0, 0)))
    responses.append(encode_result(Result(10, 2, 0, (0,) * 10, 0, 0)))
    directory = DEFAULT_OUTPUT_DIR
    (directory / "uart_requests.mem").write_text(
        "\n".join(f"{byte:02X}" for request in requests for byte in request) + "\n",
        encoding="ascii", newline="\n")
    (directory / "uart_responses.mem").write_text(
        "\n".join(f"{byte:02X}" for response in responses for byte in response) + "\n",
        encoding="ascii", newline="\n")
    debug_frame = expected_frame(reference, mnist_a)
    debug_last = debug_frame["steps"][-1]
    debug_counts = tuple(count for step in debug_frame["steps"]
                         for count in (step["l1"].bit_count(), step["l2"].bit_count()))
    debug_response = encode_result(Result(
        12, 0, debug_frame["class"], tuple(debug_last["scores"]),
        debug_last["frame_add"], debug_last["frame_cycles"], debug_counts))
    (directory / "uart_debug_response.mem").write_text(
        "\n".join(f"{byte:02X}" for byte in debug_response) + "\n",
        encoding="ascii", newline="\n")
    (directory / "uart_manifest.json").write_text(json.dumps({
        "requests": ["zero", "busy dropped single", "MNIST validation 55002",
                     "bad checksum", "bad timestep index",
                     "MNIST validation 55003 after reset", "single"],
        "response_sequences": [10, 12, 13, 16, 14, 15, 10],
        "extra_case": "first ten bytes of request 0, then receiver timeout",
        "uart_clocks_per_bit": 8,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("7 full request waveforms plus 1 partial, 7 expected responses")


if __name__ == "__main__":
    main()
