"""Basys3 SNN USB-UART packet codec (wire protocol version 1)."""

from __future__ import annotations

from dataclasses import dataclass

MAGIC = b"\xA5\x5A"
VERSION = 1
REQUEST = 0x01
RESULT = 0x81
DEBUG_RESULT = 0x82
REQUEST_LENGTH = 36
RESULT_LENGTH = 50
DEBUG_LENGTH = 66
STATUS_OK = 0
STATUS_BAD_CHECKSUM = 1
STATUS_BAD_HEADER = 2
STATUS_BUSY = 3


class PacketError(ValueError):
    pass


def checksum(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


def packet(sequence: int, kind: int, payload: bytes) -> bytes:
    if not 0 <= sequence <= 255 or len(payload) > 255:
        raise ValueError("sequence or payload out of range")
    body = bytes((VERSION, sequence, kind, len(payload))) + payload
    return MAGIC + body + bytes((checksum(body),))


def unpack(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < 7 or data[:2] != MAGIC:
        raise PacketError("missing packet sync")
    version, sequence, kind, length = data[2:6]
    if version != VERSION or len(data) != 7 + length:
        raise PacketError("invalid version or packet length")
    if checksum(data[2:]) != 0:
        raise PacketError("checksum mismatch")
    return sequence, kind, data[6:-1]


def encode_request(sequence: int, words: tuple[int, int, int, int]) -> bytes:
    if len(words) != 4 or any(not 0 <= word < (1 << 64) for word in words):
        raise ValueError("request requires four 64-bit spike words")
    payload = b"".join(bytes((step,)) + word.to_bytes(8, "little")
                       for step, word in enumerate(words))
    return packet(sequence, REQUEST, payload)


def decode_request(data: bytes) -> tuple[int, tuple[int, int, int, int]]:
    sequence, kind, payload = unpack(data)
    if kind != REQUEST or len(payload) != REQUEST_LENGTH:
        raise PacketError("wrong request type or length")
    if any(payload[9 * step] != step for step in range(4)):
        raise PacketError("invalid timestep ordering")
    return sequence, tuple(int.from_bytes(payload[9 * t + 1:9 * t + 9], "little")
                           for t in range(4))


@dataclass(frozen=True)
class Result:
    sequence: int
    status: int
    prediction: int
    scores: tuple[int, ...]
    synaptic_additions: int
    compute_cycles: int
    debug_counts: tuple[int, ...] = ()


def encode_result(result: Result) -> bytes:
    if len(result.scores) != 10 or len(result.debug_counts) not in (0, 8):
        raise ValueError("result requires ten scores and zero or eight debug counts")
    payload = bytes((result.status, result.prediction))
    payload += b"".join(int(score).to_bytes(4, "little", signed=True)
                        for score in result.scores)
    payload += result.synaptic_additions.to_bytes(4, "little")
    payload += result.compute_cycles.to_bytes(4, "little")
    payload += b"".join(value.to_bytes(2, "little") for value in result.debug_counts)
    return packet(result.sequence, DEBUG_RESULT if result.debug_counts else RESULT, payload)


def decode_result(data: bytes) -> Result:
    sequence, kind, payload = unpack(data)
    if kind not in (RESULT, DEBUG_RESULT):
        raise PacketError("wrong result type")
    if len(payload) != (DEBUG_LENGTH if kind == DEBUG_RESULT else RESULT_LENGTH):
        raise PacketError("wrong result length")
    return Result(
        sequence, payload[0], payload[1],
        tuple(int.from_bytes(payload[2 + 4 * i:6 + 4 * i], "little", signed=True)
              for i in range(10)),
        int.from_bytes(payload[42:46], "little"),
        int.from_bytes(payload[46:50], "little"),
        tuple(int.from_bytes(payload[50 + 2 * i:52 + 2 * i], "little")
              for i in range(8)) if kind == DEBUG_RESULT else (),
    )


def read_packet(transport) -> bytes:
    """Resynchronize at the magic marker and reject malformed frames."""
    while True:
        first = transport.read(1)
        if not first:
            raise TimeoutError("UART response timeout")
        if first == MAGIC[:1]:
            second = transport.read(1)
            if second == MAGIC[1:]:
                break
            if not second:
                raise TimeoutError("UART response timeout")
    header = transport.read(4)
    if len(header) != 4:
        raise PacketError("short packet header")
    length = header[3]
    if length not in (RESULT_LENGTH, DEBUG_LENGTH):
        raise PacketError("unexpected packet length")
    tail = transport.read(length + 1)
    if len(tail) != length + 1:
        raise PacketError("short packet payload")
    return MAGIC + header + tail
