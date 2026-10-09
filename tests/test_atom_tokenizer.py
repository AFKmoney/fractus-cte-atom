"""Round-trip and id-range lock for the Atom→Fractus adapter."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer


def test_roundtrip_french_and_code():
    tok = AtomFractusTokenizer(max_span_bytes=32, pack_mode="linguistic")
    text = "Bonjour Philippe.\n2+2=4. def tick(): return h"
    ids = tok.encode(text)
    assert ids[0] == 257 and ids[-1] == 258
    assert max(ids) < VOCAB_SIZE
    assert tok.decode(ids) == text


def test_no_gpt2_range():
    tok = AtomFractusTokenizer()
    ids = tok.encode("fractus")
    assert all(i < 266 for i in ids)
    assert any(i >= 259 for i in ids)  # at least one boundary marker
