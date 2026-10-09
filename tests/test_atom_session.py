"""One pass through every Atom-plugged Fractus mechanism."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.atom_session import AtomSession


def test_all_mechanisms_fire():
    session = AtomSession(scale="smoke")
    report = session.checklist()
    assert report["vocab"] == 266
    assert report["tied"] is True
    assert report["routing_blocks"] == 2
    assert report["train"]["ce"] > 0
    assert report["memories"] == 1
    assert report["recalled"] is True
    assert report["kb"] == 1
    assert report["rag_hit"] is True
    grown = session.grow_body(dict(n_layers=3, n_experts=6))
    assert "n_layers" in grown
    assert session.engine.vocab_size == 266
    print(report)
    print(grown)


if __name__ == "__main__":
    test_all_mechanisms_fire()
    print("plugged ok")
