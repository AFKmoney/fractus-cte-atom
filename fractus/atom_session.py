"""Single plug of the original Fractus CTE body onto the Atom id stream.

The parent files are the mechanisms. This module is the only place that
constructs them together for vocab 266. A mechanism not reached from here
is not on the Atom path.

Wired:
  AtomFractusTokenizer, ContinuousThoughtEngine,
  apply_kuramoto_routing_fix, PersistentMemory, CognitiveModes,
  KnowledgeBase + RAGEngine, grow_atom, maybe_grow,
  generate_greedy_prefix (speech gate: no ban, no temperature),
  generate_with_surgery (available, not the gate).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer
from .cognitive_modes import CognitiveModes
from .continuous_engine import ContinuousThoughtEngine
from .decode_surgery import generate_with_surgery
from .generate_aligned import generate_greedy_prefix
from .grow import grow_atom, grow_summary
from .kuramoto_fix import apply_kuramoto_routing_fix
from .memory import PersistentMemory
from .rag import KnowledgeBase, RAGEngine

SMOKE = dict(
    d_model=64, n_heads=4, d_head=16, n_levels=1,
    n_oscillators=4, coupling_rank=2,
    n_experts=4, top_k=2, expert_d_ff=64, siren_rank=8,
    n_layers=2,
)
ONE_B = dict(
    d_model=1280, n_heads=20, d_head=64, n_levels=2,
    n_oscillators=16, coupling_rank=8,
    n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
    n_layers=16,
)


class AtomSession:
    def __init__(self, scale: str = "smoke", apply_fix: bool = True):
        if scale not in ("smoke", "1b"):
            raise ValueError(scale)
        self.tokenizer = AtomFractusTokenizer()
        cfg = dict(SMOKE if scale == "smoke" else ONE_B)
        self.engine = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, **cfg)
        self.scale = scale
        self.routing = apply_kuramoto_routing_fix(self.engine, log=lambda *_: None) if apply_fix else {}
        self.memory = PersistentMemory(d_model=self.engine.d_model, max_memories=64)
        self.engine.attach_memory(self.memory)
        self.modes = CognitiveModes(
            n_oscillators=self.engine.blocks[0].kuramoto.N, n_modes=4
        )
        self.kb = KnowledgeBase(d_model=self.engine.d_model)
        self.rag = RAGEngine(self.engine, self.tokenizer, self.kb)
        self.opt = torch.optim.AdamW(self.engine.parameters(), lr=3e-4, weight_decay=0.01)

    def train_chunk(self, ids: torch.Tensor) -> dict:
        """One teacher-forced chunk. Carries thought, counts expert hits, may grow."""
        if int(ids.max()) >= VOCAB_SIZE:
            raise ValueError("id outside Atom vocab 266")
        self.engine.train()
        chunk = ids.view(1, -1)
        target = torch.cat([ids[1:], ids[-1:]])
        logits, lb = self.engine.tick_chunk_train(chunk)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target)
        loss = ce + 0.01 * lb
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), 1.0)
        self.opt.step()
        grew = self.engine.maybe_grow(min_ticks_between_grows=1, max_experts=64, imbalance_threshold=0.0)
        phases = self.engine.blocks[0].kuramoto_phases.detach().reshape(1, -1)
        samples = phases.repeat(8, 1)
        samples = samples + 0.05 * torch.randn_like(samples)
        self.modes.fit(samples, n_iters=5)
        mode = self.modes.classify(phases.reshape(-1))
        return {
            "ce": float(ce.detach()),
            "lb": float(lb.detach()),
            "grew": bool(grew),
            "mode": mode["mode"],
            "experts": self.engine.blocks[0].moe.n_experts,
        }

    def remember(self, text: str) -> None:
        self.engine.eval()
        ids = self.tokenizer.encode(text)[:32]
        with torch.no_grad():
            self.engine.reset_thought(1)
            self.engine.tick_chunk_train(torch.tensor([ids], dtype=torch.long))
        self.memory.consolidate(self.engine.thought_state, context=text[:80], importance=1.0)
        emb = self.engine.observe.weight[ids].mean(dim=0).detach().cpu().numpy()
        self.kb.add(text, emb, source="atom")

    def speak(self, prompt: str, n: int = 8) -> str:
        text, _ = generate_greedy_prefix(self.engine, self.tokenizer, prompt, max_new=n, window=64)
        return text

    def speak_surgery(self, prompt: str, n: int = 4) -> str:
        return generate_with_surgery(self.engine, self.tokenizer, prompt, max_new=n)

    def grow_body(self, new_config: dict) -> dict:
        old = self.engine
        self.engine = grow_atom(old, new_config)
        self.engine.attach_memory(self.memory)
        self.opt = torch.optim.AdamW(self.engine.parameters(), lr=3e-4, weight_decay=0.01)
        return grow_summary(old, self.engine)

    def checklist(self) -> dict:
        ids = torch.tensor(self.tokenizer.encode("Fractus pense. Atom coupe."), dtype=torch.long)
        step = self.train_chunk(ids[:16])
        self.remember("Fractus pense en continu.")
        recalled = self.memory.recall(self.engine.thought_state.flatten(), top_k=1)
        hit = self.rag.retriever.retrieve("Fractus", self.tokenizer, top_k=1)
        said = self.speak("Bonjour", n=4)
        return {
            "vocab": self.engine.vocab_size,
            "tied": self.engine.output_head.weight.data_ptr() == self.engine.observe.weight.data_ptr(),
            "routing_blocks": self.routing.get("blocks"),
            "train": step,
            "memories": len(self.memory),
            "recalled": bool(recalled),
            "kb": len(self.kb),
            "rag_hit": bool(hit),
            "speech_gate": "greedy_prefix",
            "spoke": said,
        }
