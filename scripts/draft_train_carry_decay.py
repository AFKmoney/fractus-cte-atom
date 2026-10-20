"""DRAFT (written, NOT launched, NOT wired into run_v4.py / the running pod): v4 trainer with a decaying carry.

Why (docs/LONG_CARRY.md): the carry (attn_S, attn_z) is a plain running sum, so linear attention averages
uniformly over every id since the last BOS; at grid chunk 8 ~94 % of the denominator phi(q).z comes from the carry and
the current chunk is diluted. v4 610M is worse on chunk 7 with the full carry (2.498) than with only 1 chunk of carry
(2.343). Training the forgetting in is the clean fix; inference-only decay is a stop-gap.

Modes (env):
  CARRY_MODE=fixed   CARRY_GAMMA=0.8   -> exactly the existing trainer knob CARRY_DECAY (bound_carry() multiplies
                                          attn_S, attn_z by gamma after every step = once per 128-id chunk). This script
                                          only sets CARRY_DECAY and execs the unmodified v4 trainer.
  CARRY_MODE=learned CARRY_GAMMA=0.8   -> per-block, per-head learned decay gamma_h = sigmoid(carry_logit_h), init
                                          logit(CARRY_GAMMA). Applied where the chunk READS the carry (_build_carry), so
                                          gamma gets gradient through the current chunk's attention. New parameter
                                          blocks.*.carry_logit; old checkpoints load with these keys missing (filled from
                                          the init). Inference must apply the same patch (fractus/carry_policy.py
                                          'learned' would read carry_logit from the checkpoint).
The v4 trainer file itself is not modified; patches are applied in-process before it runs.
Usage (example, not run):  CARRY_MODE=learned CARRY_GAMMA=0.8 <usual v4 env> python scripts/draft_train_carry_decay.py
"""
import os, sys, math, runpy
import torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MODE = os.environ.get('CARRY_MODE', 'fixed'); GAMMA = float(os.environ.get('CARRY_GAMMA', '0.8'))
TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fast4gpu_atom_v4.py')

if MODE == 'fixed':
    os.environ['CARRY_DECAY'] = str(GAMMA)
elif MODE == 'learned':
    os.environ['CARRY_DECAY'] = '1.0'   # no extra fixed decay on top
    from fractus import continuous_engine as ce
    _init, _build = ce.CTEBlock.__init__, ce.CTEBlock._build_carry
    def __init__(self, *a, **k):
        _init(self, *a, **k)
        self.carry_logit = nn.Parameter(torch.full((self.attn.n_heads,), math.log(GAMMA / (1 - GAMMA))))
    def _build_carry(self, B):
        S, z = _build(self, B)                                     # (B*nL*nH, dH, dH), (B*nL*nH, dH)
        nL, nH = self.attn.n_levels, self.attn.n_heads
        g = torch.sigmoid(self.carry_logit).to(S.dtype).repeat(B * nL)   # order (B, nL, nH) flattened, nH fastest
        return S * g[:, None, None], z * g[:, None]
    def _store_carry(self, B, S_final, z_final):
        # the stored carry is the decayed past + this chunk (S_final already includes g*S0): keep as is, detached
        _store(self, B, S_final, z_final)
    _store = ce.CTEBlock._store_carry
    ce.CTEBlock.__init__, ce.CTEBlock._build_carry = __init__, _build_carry
    _lsd = ce.ContinuousThoughtEngine.load_state_dict
    def load_state_dict(self, sd, strict=True, **kw):
        missing = [k for k in self.state_dict() if k.endswith('carry_logit') and k not in sd]
        sd = dict(sd); own = self.state_dict()
        for k in missing: sd[k] = own[k]
        return _lsd(self, sd, strict=strict, **kw)
    ce.ContinuousThoughtEngine.load_state_dict = load_state_dict
else:
    raise SystemExit(f'CARRY_MODE {MODE!r}')

if __name__ == '__main__':
    sys.argv = [TRAINER] + sys.argv[1:]
    runpy.run_path(TRAINER, run_name='__main__')
