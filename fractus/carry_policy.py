"""Inference-only carry decay for the linear-attention carry (docs/LONG_CARRY.md). No weight change.

The block carry (attn_S, attn_z) is a plain running sum of phi(k) v^T and phi(k) since the cold start / BOS, so
y_t = phi(q_t) S_t / phi(q_t) z_t is a UNIFORM average over every past id: at grid chunk 8 ~94 % of the denominator
comes from the carry and the current chunk is diluted. set_carry_decay(eng, g) makes every chunk read g * carry
(S and z scaled jointly, so the S/z ratio of the past is kept; this is NOT a norm bound/clamp of attn_S/attn_z).
Applied at every chunk boundary -> exponential forgetting with weight g**age_in_chunks. g = 1.0 is exactly the
trained function (default). Works for every decoder (greedy, beam, min-p) since it sits in CTEBlock._build_carry.
"""


def set_carry_decay(eng, gamma: float = 1.0) -> None:
    gamma = float(gamma)
    eng._carry_gamma = gamma
    for blk in eng.blocks:
        if not getattr(blk, '_carry_policy_patched', False):
            orig = blk._build_carry

            def _build_carry(B, blk=blk, orig=orig, eng=eng):
                g = getattr(eng, '_carry_gamma', 1.0)
                if g != 1.0:
                    blk.attn_S = blk.attn_S * g
                    blk.attn_z = blk.attn_z * g
                return orig(B)

            blk._build_carry = _build_carry
            blk._carry_policy_patched = True
