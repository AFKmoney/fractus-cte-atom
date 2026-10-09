"""Atomizer spans adapted to Fractus integer ticks.

ATOM does not assign token IDs. Fractus CTE (HF thefinalboss/fractus-cte)
does: ``observe`` is an embedding and ``output_head`` is tied to it, so every
tick is one integer in ``[0, vocab_size)``.

This adapter keeps the Atomizer contract (byte spans, structural boundaries,
exact UTF-8 round-trip, no BPE, no GPT-2) and emits a closed id stream the
existing engine can train from scratch:

    0..255              raw payload byte
    256                 PAD
    257                 BOS
    258                 EOS
    259..265            Atomizer boundary marker (after each packet)

Boundary ids follow ``Atomizer._BOUNDARY_CODES`` (1..7) shifted by 258.
Decode drops every id >= 256 and rebuilds UTF-8 from the surviving bytes.
The x8 GPT-2 checkpoints (vocab 50257) are not loadable into this head.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .atomizer import Atomizer

PAD = 256
BOS = 257
EOS = 258
BOUNDARY_BASE = 258  # id = BOUNDARY_BASE + Atomizer code (1..7)
VOCAB_SIZE = 266  # 256 bytes + pad/bos/eos + 7 boundary markers

_BOUNDARY_NAME = {code: name for name, code in Atomizer._BOUNDARY_CODES.items()}


class AtomFractusTokenizer:
    """Atomizer segmentation, Fractus id stream. No learned vocabulary."""

    VERSION = "atom-fractus-v1"
    vocab_size = VOCAB_SIZE

    def __init__(
        self,
        max_span_bytes: int = 32,
        pack_mode: str = "linguistic",
        phase_period: float = 256.0,
    ) -> None:
        self.atomizer = Atomizer(
            max_span_bytes=max_span_bytes,
            pack_mode=pack_mode,
            phase_period=phase_period,
        )
        self.max_span_bytes = max_span_bytes
        self.pack_mode = pack_mode

    def encode(self, text: str, add_specials: bool = True) -> List[int]:
        packets = self.atomizer.encode(text, reset=True)
        ids: List[int] = [BOS] if add_specials else []
        for packet in packets:
            ids.extend(packet.payload)
            code = Atomizer._BOUNDARY_CODES.get(packet.boundary, 0)
            if code:
                ids.append(BOUNDARY_BASE + code)
        if add_specials:
            ids.append(EOS)
        return ids

    def encode_bytes(self, raw: bytes, add_specials: bool = True) -> List[int]:
        packets = self.atomizer.encode_bytes(raw, reset=True, flush=True)
        ids: List[int] = [BOS] if add_specials else []
        for packet in packets:
            ids.extend(packet.payload)
            code = Atomizer._BOUNDARY_CODES.get(packet.boundary, 0)
            if code:
                ids.append(BOUNDARY_BASE + code)
        if add_specials:
            ids.append(EOS)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytes(i for i in ids if 0 <= int(i) < 256)
        return raw.decode("utf-8", errors="replace")

    def payload_bytes(self, ids: Sequence[int]) -> bytes:
        return bytes(i for i in ids if 0 <= int(i) < 256)

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]
