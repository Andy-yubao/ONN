from __future__ import annotations

import unittest

from fpga.basys3.protocol import (
    PacketError, Result, checksum, decode_request, decode_result, encode_request,
    encode_result, read_packet,
)


class BytesTransport:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, count: int) -> bytes:
        result, self.data = self.data[:count], self.data[count:]
        return result


class PacketTest(unittest.TestCase):
    def test_request_roundtrip_and_bit_order(self):
        words = (1, 1 << 63, 0x0102030405060708, 0)
        encoded = encode_request(255, words)
        self.assertEqual(len(encoded), 43)
        self.assertEqual(encoded[7:15], b"\x01\x00\x00\x00\x00\x00\x00\x00")
        self.assertEqual(decode_request(encoded), (255, words))

    def test_reject_malformed(self):
        encoded = bytearray(encode_request(4, (0,) * 4))
        encoded[-1] ^= 1
        with self.assertRaises(PacketError):
            decode_request(bytes(encoded))
        encoded = bytearray(encode_request(4, (0,) * 4))
        encoded[6] = 1
        encoded[-1] = checksum(encoded[2:-1])
        with self.assertRaises(PacketError):
            decode_request(bytes(encoded))

    def test_result_signed_and_debug(self):
        result = Result(3, 0, 7, (-1048576, -1, 0, 1048575) + (0,) * 6,
                        12345, 67890, tuple(range(8)))
        encoded = encode_result(result)
        self.assertEqual(len(encoded), 73)
        self.assertEqual(decode_result(read_packet(BytesTransport(b"garbage" + encoded))), result)


if __name__ == "__main__":
    unittest.main()
