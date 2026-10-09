"""ATOM-native streaming atomization.

This module deliberately does not build a vocabulary and does not assign token
IDs.  Raw UTF-8 bytes are treated as observations.  The atomizer groups the
observations into reversible spans and emits an ``AtomPacket`` containing the
surface bytes plus deterministic local/contextual features.  A downstream
ATOM compiler turns that packet into the eight toroidal atom properties.

The important distinction from the legacy GPT-2 path is:

    raw stream + local buffer + current context -> packet -> toroidal atom

The packet is an event, not a permanent vocabulary entry.  Two equal surface
spans can have different features when their preceding stream context differs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import string
from typing import Iterable, Iterator

import torch


@dataclass(frozen=True)
class AtomPacket:
    """One reversible, context-conditioned input event for ATOM.

    ``payload`` is kept as bytes so that round-tripping does not depend on a
    lossy Unicode replacement.  ``features`` is a fixed-size observation used
    by the token-to-atom compiler; it is not a token ID or a vocabulary row.
    """

    payload: bytes
    start: int
    end: int
    boundary: str
    level: int
    phase: float
    duration: float
    features: torch.Tensor

    @property
    def length(self) -> int:
        return len(self.payload)

    @property
    def text(self) -> str:
        return self.payload.decode("utf-8", errors="replace")

    def to(self, device: str | torch.device) -> "AtomPacket":
        return AtomPacket(
            payload=self.payload,
            start=self.start,
            end=self.end,
            boundary=self.boundary,
            level=self.level,
            phase=self.phase,
            duration=self.duration,
            features=self.features.to(device),
        )


class Atomizer:
    """A small stateful compiler from a byte stream to atom packets.

    Segmentation is online and reversible.  It uses structural boundaries
    (whitespace, punctuation, line breaks and bounded span duration), not a
    learned BPE vocabulary.  The packet features contain a byte observation,
    span statistics and a rolling context sketch.  The latter makes the same
    surface span context-sensitive without introducing token IDs.

    ``max_span_bytes`` is a safety bound, not a vocabulary merge size.  The
    boundary is always emitted at the end of the stream, so every byte is
    represented exactly once.

    Pack modes
    ----------
    ``eager``
        Flush on every whitespace / punctuation / newline (legacy word-ish packs).
    ``linguistic``
        Prefer longer dialogue-aligned spans: accumulate toward ``max_span_bytes``,
        flush on newlines (Utilisateur:/Assistant: lines), and cut at the last
        whitespace/punctuation when the bound is hit — still pure byte-span,
        no BPE / HF.
    """

    VERSION = "atomizer-v1-byte-span"
    FEATURE_DIM = 320
    _PUNCTUATION = set(bytes(string.punctuation, "ascii"))
    _BOUNDARY_CODES = {
        "whitespace": 1,
        "punctuation": 2,
        "newline": 3,
        "class_transition": 4,
        "max_span": 5,
        "eos": 6,
        "generated": 7,
    }
    _PACK_MODES = ("eager", "linguistic")

    def __init__(
        self,
        max_span_bytes: int = 32,
        phase_period: float = 256.0,
        pack_mode: str = "linguistic",
    ) -> None:
        if max_span_bytes < 1:
            raise ValueError("max_span_bytes must be at least 1")
        if phase_period <= 0:
            raise ValueError("phase_period must be positive")
        mode = str(pack_mode).strip().lower()
        if mode not in self._PACK_MODES:
            raise ValueError(f"pack_mode must be one of {self._PACK_MODES}, got {pack_mode!r}")
        self.max_span_bytes = int(max_span_bytes)
        self.phase_period = float(phase_period)
        self.pack_mode = mode
        # Soft flush threshold for linguistic mode (~3/4 of max): pack longer
        # spans that still end on whitespace/punctuation when nearly full.
        self._soft_flush_bytes = max(2, (self.max_span_bytes * 3) // 4)
        self.reset()

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    def reset(self) -> None:
        """Reset only the streaming context, not the configuration."""
        self.position = 0
        self.previous_length = 0
        self.context_group = torch.zeros(16, dtype=torch.float32)
        self.packet_count = 0
        self._buffer = bytearray()
        self._buffer_start = 0
        self._last_soft_cut = 0  # exclusive end index inside buffer

    @staticmethod
    def _is_continuation_byte(value: int) -> bool:
        return 0x80 <= value <= 0xBF

    @classmethod
    def _ends_at_utf8_boundary(cls, buffer: bytearray) -> bool:
        """Return whether a max-span cut would preserve UTF-8 code points."""
        if not buffer:
            return True
        last = buffer[-1]
        if last < 0x80:
            return True
        if cls._is_continuation_byte(last):
            continuation_count = 0
            for value in reversed(buffer):
                if not cls._is_continuation_byte(value):
                    break
                continuation_count += 1
            if continuation_count >= len(buffer):
                return False
            lead = buffer[-continuation_count - 1]
            if 0xC2 <= lead <= 0xDF:
                expected = 1
            elif 0xE0 <= lead <= 0xEF:
                expected = 2
            elif 0xF0 <= lead <= 0xF4:
                expected = 3
            else:
                return True
            return continuation_count >= expected
        # A leading multi-byte byte at the end is incomplete.
        return not (0xC2 <= last <= 0xF4)

    @staticmethod
    def _class(value: int) -> str:
        if value in (9, 10, 13, 32):
            return "space"
        if 48 <= value <= 57:
            return "digit"
        if 65 <= value <= 90 or 97 <= value <= 122:
            return "alpha"
        if value < 32:
            return "control"
        if value in Atomizer._PUNCTUATION:
            return "punctuation"
        if Atomizer._is_continuation_byte(value):
            return "continuation"
        return "other"

    @classmethod
    def _boundary_reason(cls, buffer: bytearray) -> str | None:
        if not buffer:
            return None
        current = buffer[-1]
        if current == 10 or current == 13:
            return "newline"
        if current in cls._PUNCTUATION:
            return "punctuation"
        if current in (9, 32) and any(byte not in (9, 10, 13, 32) for byte in buffer[:-1]):
            return "whitespace"
        # A structural switch is useful for mixed identifiers/numbers, but do
        # not split UTF-8 continuation bytes in the middle of a code point.
        if len(buffer) >= 2 and not cls._is_continuation_byte(current):
            previous = buffer[-2]
            previous_class = cls._class(previous)
            current_class = cls._class(current)
            if previous_class != current_class and {previous_class, current_class} <= {"alpha", "digit"}:
                return "class_transition"
        return None

    @classmethod
    def _is_soft_boundary(cls, reason: str | None) -> bool:
        return reason in {"whitespace", "punctuation", "class_transition"}

    def _best_linguistic_cut(self, buffer: bytearray) -> int | None:
        """Exclusive end index of preferred WS/punct cut inside buffer, or None."""
        if len(buffer) < 2:
            return None
        # Prefer an already-tracked soft cut if valid.
        if 2 <= self._last_soft_cut < len(buffer):
            prefix = bytearray(buffer[: self._last_soft_cut])
            if self._ends_at_utf8_boundary(prefix):
                return self._last_soft_cut
        # Search backward for whitespace / punctuation (keep ≥1 byte after cut).
        for i in range(len(buffer) - 2, 0, -1):
            value = buffer[i]
            if value in (9, 10, 13, 32) or value in self._PUNCTUATION:
                cut = i + 1
                prefix = bytearray(buffer[:cut])
                if cut >= 2 and self._ends_at_utf8_boundary(prefix):
                    return cut
        return None

    @staticmethod
    def _entropy(hist: torch.Tensor) -> float:
        positive = hist[hist > 0]
        if positive.numel() == 0:
            return 0.0
        return float(-(positive * positive.log2()).sum().item() / 8.0)

    @staticmethod
    def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        denom = float(left.norm().item() * right.norm().item())
        if denom <= 1e-12:
            return 0.0
        return float(torch.dot(left, right).item() / denom)

    def _make_features(self, payload: bytes, start: int, boundary: str, level: int) -> torch.Tensor:
        values = torch.tensor(list(payload), dtype=torch.long)
        hist = torch.bincount(values, minlength=256).float()
        hist = hist / max(float(len(payload)), 1.0)
        # Keep the low-nibble groups for the rolling context sketch.  Unlike
        # a high-nibble grouping, this distinguishes nearby ASCII symbols such
        # as ``a`` and ``b`` while remaining only 16 values wide.
        grouped = hist.reshape(16, 16).sum(dim=0)

        first = torch.zeros(16, dtype=torch.float32)
        last = torch.zeros(16, dtype=torch.float32)
        first[payload[0] >> 4] = 1.0
        last[payload[-1] >> 4] = 1.0

        whitespace = sum(value in (9, 10, 13, 32) for value in payload) / len(payload)
        digits = sum(48 <= value <= 57 for value in payload) / len(payload)
        alpha = sum(65 <= value <= 90 or 97 <= value <= 122 for value in payload) / len(payload)
        punctuation = sum(value in self._PUNCTUATION for value in payload) / len(payload)
        control = sum(value < 32 for value in payload) / len(payload)
        transitions = sum(
            self._class(left) != self._class(right)
            for left, right in zip(payload, payload[1:])
        ) / max(len(payload) - 1, 1)
        phase = 2.0 * math.pi * (start % self.phase_period) / self.phase_period
        context_similarity = self._safe_cosine(grouped, self.context_group)
        boundary_code = self._BOUNDARY_CODES.get(boundary, 0)
        stats = torch.tensor(
            [
                len(payload) / self.max_span_bytes,
                whitespace,
                digits,
                alpha,
                punctuation,
                control,
                self._entropy(hist),
                transitions,
                payload[0] / 255.0,
                payload[-1] / 255.0,
                math.sin(phase),
                math.cos(phase),
                self.previous_length / self.max_span_bytes,
                (context_similarity + 1.0) / 2.0,
                boundary_code / max(self._BOUNDARY_CODES.values()),
                level / 3.0,
            ],
            dtype=torch.float32,
        )
        # 256 byte observations + first/last nibbles + 16 scalar observables +
        # 16 rolling context values = 320 features.
        return torch.cat((hist, first, last, stats, self.context_group), dim=0)

    def _level_for(self, boundary: str, payload: bytes) -> int:
        if boundary == "newline" or b"\n" in payload:
            return 3
        if boundary == "punctuation" or any(byte in self._PUNCTUATION for byte in payload):
            return 2
        return 1

    def _emit(self, payload: bytes, start: int, boundary: str) -> AtomPacket:
        level = self._level_for(boundary, payload)
        phase = 2.0 * math.pi * (start % self.phase_period) / self.phase_period
        features = self._make_features(payload, start, boundary, level)
        packet = AtomPacket(
            payload=payload,
            start=start,
            end=start + len(payload),
            boundary=boundary,
            level=level,
            phase=phase,
            duration=float(len(payload)),
            features=features,
        )
        grouped = features[:256].reshape(16, 16).sum(dim=0)
        self.context_group.mul_(0.85).add_(0.15 * grouped)
        self.previous_length = len(payload)
        self.position = packet.end
        self.packet_count += 1
        return packet

    def _flush_pending(self, boundary: str = "eos") -> list[AtomPacket]:
        if not self._buffer:
            return []
        packet = self._emit(bytes(self._buffer), self._buffer_start, boundary)
        self._buffer.clear()
        self._buffer_start = self.position
        self._last_soft_cut = 0
        return [packet]

    def _flush_prefix(self, cut: int, boundary: str) -> list[AtomPacket]:
        """Emit buffer[:cut]; keep the remainder in the pending buffer."""
        if cut <= 0 or cut > len(self._buffer):
            raise ValueError("invalid linguistic cut")
        payload = bytes(self._buffer[:cut])
        remainder = bytearray(self._buffer[cut:])
        packet = self._emit(payload, self._buffer_start, boundary)
        self._buffer = remainder
        self._buffer_start = self.position
        self._last_soft_cut = 0
        return [packet]

    def encode_bytes(
        self,
        raw: bytes,
        reset: bool = True,
        flush: bool = True,
    ) -> list[AtomPacket]:
        """Atomize bytes and guarantee exact byte reconstruction.

        ``flush=False`` keeps a partial UTF-8/span buffer for the next call;
        this is what makes ``stream`` genuinely incremental.
        """
        if reset:
            self.reset()
        packets: list[AtomPacket] = []
        linguistic = self.pack_mode == "linguistic"
        for value in raw:
            if not self._buffer:
                self._buffer_start = self.position
            self._buffer.append(value)
            reason = self._boundary_reason(self._buffer)
            if linguistic:
                if reason == "newline":
                    packets.extend(self._flush_pending(reason))
                    continue
                if self._is_soft_boundary(reason):
                    self._last_soft_cut = len(self._buffer)
                    # Near-full soft flush: end a long span on WS/punct (FR dialogue).
                    if len(self._buffer) >= self._soft_flush_bytes:
                        packets.extend(self._flush_pending(reason))
                        continue
                # Byte-tick (max_span 1): pure byte stream; UTF-8 composition is
                # the model's job, so mid-codepoint splits are allowed by design.
                if len(self._buffer) >= self.max_span_bytes and (
                    self.max_span_bytes == 1 or self._ends_at_utf8_boundary(self._buffer)
                ):
                    cut = self._best_linguistic_cut(self._buffer)
                    if cut is not None and 2 <= cut < len(self._buffer):
                        # Prefer WS/punct cut; label by the byte that ended the prefix.
                        cut_byte = self._buffer[cut - 1]
                        if cut_byte in (9, 10, 13, 32):
                            label = "whitespace"
                        elif cut_byte in self._PUNCTUATION:
                            label = "punctuation"
                        else:
                            label = "max_span"
                        packets.extend(self._flush_prefix(cut, label))
                    else:
                        packets.extend(self._flush_pending("max_span"))
                continue
            # Eager (legacy): flush on every structural boundary.
            if reason is not None:
                packets.extend(self._flush_pending(reason))
            elif len(self._buffer) >= self.max_span_bytes and (
                self.max_span_bytes == 1 or self._ends_at_utf8_boundary(self._buffer)
            ):
                packets.extend(self._flush_pending("max_span"))
        if flush:
            packets.extend(self._flush_pending("eos"))
        return packets

    def encode(self, text: str, reset: bool = True) -> list[AtomPacket]:
        return self.encode_bytes(text.encode("utf-8"), reset=reset, flush=True)

    def packet_from_payload(self, payload: bytes, boundary: str = "generated") -> AtomPacket:
        """Commit a predicted surface packet to the atomizer context.

        Byte-tick contract (ARCHITECTURE.md): ``"generated"`` is OOD and is
        never written -- a train-like boundary is inferred from the bytes so
        train and generate share the same feature distribution.
        """
        if not payload:
            raise ValueError("generated atom payload cannot be empty")
        if self._buffer:
            raise ValueError("cannot inject a generated packet while a raw span is pending")
        if boundary == "generated":
            boundary = self._infer_commit_boundary(bytes(payload))
        return self._emit(bytes(payload), self.position, boundary)

    @classmethod
    def _infer_commit_boundary(cls, payload: bytes) -> str:
        last = payload[-1]
        if last in (10, 13):
            return "newline"
        if last in cls._PUNCTUATION:
            return "punctuation"
        if last in (9, 32):
            return "whitespace"
        return "max_span"

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "max_span_bytes": self.max_span_bytes,
            "phase_period": self.phase_period,
            "pack_mode": self.pack_mode,
            "position": self.position,
            "previous_length": self.previous_length,
            "context_group": self.context_group.clone(),
            "packet_count": self.packet_count,
            "buffer": bytes(self._buffer),
            "buffer_start": self._buffer_start,
            "last_soft_cut": self._last_soft_cut,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version", self.VERSION) != self.VERSION:
            raise ValueError(f"unsupported atomizer version: {state.get('version')}")
        self.position = int(state.get("position", 0))
        self.previous_length = int(state.get("previous_length", 0))
        self.context_group = state.get("context_group", torch.zeros(16)).clone().float()
        self.packet_count = int(state.get("packet_count", 0))
        self._buffer = bytearray(state.get("buffer", b""))
        self._buffer_start = int(state.get("buffer_start", self.position))
        self._last_soft_cut = int(state.get("last_soft_cut", 0))
        # pack_mode / max_span_bytes stay as constructed (CLI wins on resume).

    def __call__(self, text: str) -> list[AtomPacket]:
        return self.encode(text)

    def stream(self, chunks: Iterable[str]) -> Iterator[AtomPacket]:
        """Atomize consecutive chunks without resetting the stream context."""
        first = True
        for chunk in chunks:
            for packet in self.encode_bytes(
                chunk.encode("utf-8"),
                reset=first,
                flush=False,
            ):
                first = False
                yield packet
        for packet in self._flush_pending("eos"):
            yield packet
