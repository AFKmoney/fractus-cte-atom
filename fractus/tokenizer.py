"""Fractus tokenizer. This fork's tokenizer is the Atomizer.

The class name stays FractusTokenizer because the engine, the corpus
builder, the sandbox and the probes import it. The implementation is
ATOM byte-spans, not GPT-2 BPE. vocab is 266. There is no merge table.
"""

from typing import List, Optional

from .atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer

DEFAULT_VOCAB_SIZE = VOCAB_SIZE


class FractusTokenizer:
    """Atomizer spans as Fractus integer ids. Same encode/decode contract."""

    def __init__(self, tokenizer=None, max_span_bytes: int = 32, pack_mode: str = "linguistic"):
        self._atom = tokenizer if isinstance(tokenizer, AtomFractusTokenizer) else AtomFractusTokenizer(
            max_span_bytes=max_span_bytes, pack_mode=pack_mode
        )
        self.vocab_size = VOCAB_SIZE

    @classmethod
    def gpt2_compatible(cls) -> "FractusTokenizer":
        """Name kept so old call sites run. Returns the Atom tokenizer.

        GPT-2 BPE is not the tokenizer of this fork. Loading vocab 50257
        here would silently train the wrong head.
        """
        return cls()

    @classmethod
    def build_or_load(cls, path: Optional[str] = None) -> "FractusTokenizer":
        return cls()

    def encode(self, text: str) -> List[int]:
        return self._atom.encode(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return self._atom.encode_batch(texts)

    def decode(self, ids: List[int]) -> str:
        return self._atom.decode(ids)

    def save(self, path: Optional[str] = None):
        return None
