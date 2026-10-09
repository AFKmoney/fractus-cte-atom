"""EBVT probe for Fractus — instrumentation non-invasive (hooks forward).

Transposition EBVT -> Fractus (voir EBVT-001 / white papers du projet EBVT).
L'arbre d'activation de Fractus est un arbre "caterpillar" :

    chaine des n_layers blocs (hubs) + n_experts spokes experts par bloc.

Le signal de cut b' (le "usage cut balance") est le CUT d'utilisation :

    spoke (bloc b, expert e) : b' = C[b,e]
        (le cut de l'expert = ses events d'activation sur la fenetre)
    arete de chaine (bloc k-1 | bloc k) : b' = min(L_k, M - L_k)
        (L_k = events dans les blocs 0..k-1 ; M = total d'events)

EBVT mesure (definition exacte du projet, sur l'arbre d'activation) :

    V_arch = somme des paires d'aretes adjacentes {e,f} |b'(e) - b'(f)|

Identite layer-cake exacte (verifiee a chaque report — "suspect the
instrument") : pour t = 1..H, P_t = # paires adjacentes avec exactement une
arete dans H_t = {e : b'(e) >= t}, alors

    V_arch = somme_t P_t          (H = max b')

Metriques :
    H           = max b'   (hauteur de la hierarchie de specialisation :
                            le cut d'usage le plus profond)
    persistence = #t avec P_t > 0 / H
                  (le profile tient-il sur toutes les echelles ? La reference
                   de forme du theoreme principal EBVT est la double etoilee
                   equilibree : une frontiere centrale SOUTENUE sur tous les
                   seuils — profile persistant.)
    P_1         = largeur (frontiere active/inactif au seuil minimal)
    support     = # experts utilises
    dominance   = max count / total

Reference de forme : la double etoilee equilibree est la forme qui maximise
la variation de cut (theoreme principal EBVT, white paper). Ici la forme
"double etoilee d'activation" = deux blocs adjacents portant des cut d'usage
equilibres : l'arete de chaine centrale devient le cut le plus profond,
soutenu sur tous les seuils. (L'extremum sur le scaffold 16x128 est un
nouveau probleme ouvert de la famille — a declarer, pas a deviner.)

Le profile P_t est comparable d'un checkpoint a l'autre : H qui croit +
profile persistant = la deuxieme horloge du contenu, structurelle,
independante de topic_hits.

Non-invasif : register_forward_hook sur chaque PhaseRoutedMoE (API
officielle PyTorch, desactivable). Le hook re-calcule _compute_gates sur
les memes phases que le forward (deterministe, sans RNG) => selection
top-k IDENTIQUE a celle du moteur par construction. Zero ligne modifiee
dans le package fractus/.

Fenetre : appelez probe.reset() juste APRES le engine.reset_thought(1) de
la generation (c'est la meme convention que le compteur natif
engine._expert_hits du bloc 0, dont le report fait le cross-check).
"""

import os
import sys

import numpy as np
import torch


class EBVTError(RuntimeError):
    pass


class EBVTProbe:
    """Compteur d'activations expert + metriques EBVT sur l'arbre d'activation."""

    def __init__(self, engine, verbose: bool = False):
        self.engine = engine
        self.blocks = list(engine.blocks)
        self.n_layers = len(self.blocks)
        self.n_experts = self.blocks[0].moe.n_experts
        for i, blk in enumerate(self.blocks):
            if blk.moe.n_experts != self.n_experts:
                raise EBVTError(
                    f"bloc {i} a {blk.moe.n_experts} experts "
                    f"(attendu {self.n_experts}) — growth detecte, re-attacher le probe")
        self.counts = np.zeros((self.n_layers, self.n_experts), dtype=np.int64)
        self._hooks = []
        self._verbose = verbose
        self.active = False
        for bi, blk in enumerate(self.blocks):
            self._hooks.append(blk.moe.register_forward_hook(self._make_hook(bi)))
        self.active = True

    # ------------------------------------------------------------- hooks --
    def _make_hook(self, bi: int):
        def hook(module, inp, out):
            if not self.active:
                return
            with torch.no_grad():
                gates = module._compute_gates(inp[1])          # (B, L, E)
                idx = gates.topk(module.top_k, dim=-1).indices  # (B, L, K)
                flat = idx.flatten().cpu().numpy()
                if flat.size:
                    u, c = np.unique(flat, return_counts=True)
                    self.counts[bi, u] += c
        return hook

    def detach(self):
        """Retire les hooks (desactive le comptage)."""
        self.active = False
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def native_counter(self):
        """Compteur natif du moteur (bloc 0 seul), ou None si absent."""
        eh = getattr(self.engine, "_expert_hits", None)
        if eh is None or eh.numel() != self.n_experts:
            return None
        return eh.detach().cpu().numpy().astype(np.int64).ravel().copy()

    def reset(self):
        """Vide la fenetre. A appeler juste apres engine.reset_thought(1).

        NB : reset_thought N'EFFACE PAS le compteur natif (il l'init seulement
        s'il est absent) — c'est un compteur CUMULATIF du moteur. Le report
        fait donc un DIFF : snapshot ici, difference au moment du report.
        """
        self.counts.fill(0)
        self._native_base = self.native_counter()

    # --------------------------------------------------- math pure (arbre) --
    @staticmethod
    def tree_metrics(C) -> dict:
        """Metriques EBVT de la matrice d'activation C (n_layers, n_experts).

        Arbre "caterpillar" : chaine des blocs + spokes experts.
        b'(spoke b,e) = C[b,e] ; b'(chaine k-1|k) = min(L_k, M - L_k).
        V = somme des paires d'aretes adjacentes |b'(e) - b'(f)|.
        Identite layer-cake V = somme_t P_t verifiee (EBVTError si casse).
        """
        C = np.asarray(C, dtype=np.int64)
        if C.ndim != 2:
            raise EBVTError(f"C doit etre (layers, experts), got {C.shape}")
        B, E = C.shape
        M = int(C.sum())
        row = C.sum(axis=1)                            # events par bloc
        # b' des aretes de chaine (B-1 aretes) : equilibre events gauche/droite.
        if B > 1 and M > 0:
            L = np.cumsum(row)                          # L[k] = events blocs 0..k
            chain = np.minimum(L[:-1], M - L[:-1])      # arete (k-1 | k)
        else:
            chain = np.zeros(B - 1, dtype=np.int64)

        # b' incidentes a chaque bloc (tableau 1D par hub).
        inc = []
        for b in range(B):
            parts = [C[b]]
            if b > 0:
                parts.append(chain[b - 1:b])
            if b < B - 1:
                parts.append(chain[b:b + 1])
            inc.append(np.concatenate(parts))

        # --- V direct : pour chaque hub, somme des paires |x_j - x_i| ---
        # x trie croissant, d elements : somme_{i<j} |x_j - x_i| =
        # somme_i (2i - d + 1) x_i   (formule du white paper, index 0)
        V = 0
        for b in range(B):
            x = np.sort(inc[b])
            d = x.size
            V += int((x * (2 * np.arange(d) - d + 1)).sum())

        H = int(max(int(C.max()), int(chain.max())))
        base = {
            "layers": B, "experts": E, "total": M,
            "block_events": [int(v) for v in row.tolist()],
            "chain_b": [int(v) for v in chain.tolist()],
        }
        if M == 0 or H == 0:
            base.update({"V": V, "H": H, "P1": 0, "persistence": 0.0,
                         "support": int((C > 0).sum()), "dominance": 0.0,
                         "empty": True, "profile": [], "top12": []})
            return base

        # --- Layer-cake : P_t = somme_v h_t(v)(d(v) - h_t(v)) ---
        ks = np.arange(1, H + 1)                        # (H,)
        P = np.zeros(H, dtype=np.int64)
        for b in range(B):
            x = inc[b]
            d = x.size
            act = (x[None, :] >= ks[:, None]).sum(axis=1)   # (H,)
            P += act * (d - act)
        V_cake = int(P.sum())
        if V_cake != V:
            raise EBVTError(
                f"layer-cake self-check FAILED : V_direct={V} "
                f"vs V_cake={V_cake} — instrument en defaut")

        # Profile downsample : premiers t + grille uniforme (<= 32 pts) + dernier.
        idx = np.unique(np.concatenate([
            np.arange(min(8, H)),
            np.linspace(0, H - 1, 24).astype(int),
            [H - 1],
        ]))
        profile = [[int(t + 1), int(P[t])] for t in idx]

        flat = C.reshape(-1)
        top_idx = np.argsort(-flat)[:12]
        top12 = []
        for i in top_idx:
            if flat[i] > 0:
                top12.append([int(i // E), int(i % E), int(flat[i])])

        base.update({
            "V": V,
            "H": H,
            "P1": int(P[0]),
            "persistence": round(float((P > 0).sum()) / max(H, 1), 3),
            "support": int((C > 0).sum()),
            "dominance": round(float(flat.max()) / max(M, 1), 3),
            "profile": profile,
            "top12": top12,
            "empty": False,
        })
        return base

    # ------------------------------------------------------------- report --
    def report(self, with_native_crosscheck: bool = True) -> dict:
        """Metriques EBVT de la fenetre courante (+ cross-check natif bloc 0)."""
        m = self.tree_metrics(self.counts)
        if with_native_crosscheck:
            # Le compteur natif est CUMULATIF (reset_thought ne l'efface pas,
            # voir reset()) : on compare le DIFF sur la fenetre au compte du
            # probe sur le bloc 0.
            native_now = self.native_counter()
            base = getattr(self, "_native_base", None)
            if native_now is None:
                m["native_b0"] = {"match": "no_native_counter"}
            else:
                native_win = native_now - base if base is not None else native_now
                probe_b0 = self.counts[0]
                m["native_b0"] = {
                    "match": bool(np.array_equal(native_win, probe_b0)),
                    "native_total": int(native_win.sum()),
                    "probe_total": int(probe_b0.sum()),
                    "max_abs_diff": int(np.abs(native_win - probe_b0).max()),
                }
        return m


# -------------------------------------------------------------- selftest --
def _selftest() -> int:
    """Verifications d'instrument (sans model) : identite layer-cake sur
    matrices aleatoires + cas figures (vide, spike, double-etoilee)."""
    rng = np.random.default_rng(42)
    fails = 0

    # 1. Identite V_direct == V_cake sur 200 matrices aleatoires.
    n_ok = 0
    for trial in range(200):
        B = int(rng.integers(2, 17))
        E = int(rng.integers(1, 129))
        C = rng.integers(0, 40, size=(B, E))
        try:
            m = EBVTProbe.tree_metrics(C)
            assert m["V"] >= 0
            # H = max b' >= max C ; l'identite layer-cake (si-dessus) est
            # l'invariant crucial.
            assert m["H"] >= int(C.max())
            assert m["H"] <= max(int(C.sum() // 2), int(C.max()))
            n_ok += 1
        except EBVTError as e:
            print(f"  FAIL identite trial {trial} : {e}")
    print(f"  [1/4] identite layer-cake : {n_ok}/200 ok")
    fails += 200 - n_ok

    # 2. Matrice nulle => V == 0, H == 0 (baseline nulle).
    m = EBVTProbe.tree_metrics(np.zeros((16, 128), dtype=np.int64))
    ok = m["V"] == 0 and m["H"] == 0
    detail = f"FAIL V={m['V']} H={m['H']}"
    print(f"  [2/4] vide => V==0, H==0 : {'ok' if ok else detail}")
    fails += 0 if ok else 1

    # 3. Spike (un expert chaud, 50 events) : b' = 50 sur un spoke, 0 partout
    #    ailleurs (chain : min(50, 0) = 0) => V = 50 * 128 = 6400, H = 50.
    C = np.zeros((16, 128), dtype=np.int64)
    C[0, 0] = 50
    m = EBVTProbe.tree_metrics(C)
    ok = m["V"] == 6400 and m["H"] == 50
    detail = f"FAIL V={m['V']} H={m['H']}"
    print(f"  [3/4] spike => V==6400, H==50 : {'ok' if ok else detail}")
    fails += 0 if ok else 1

    # 4. Double-etoilee d'activation (2 blocs adjacents, cut equilibres) >
    #    single-hub a masse egale : l'arete de chaine centrale (b' = M/2)
    #    devient le cut le plus profond, la forme a variation maximale.
    M = 256
    C_a = np.zeros((16, 128), dtype=np.int64)
    C_a[0, :64] = 4                        # single hub : 256 events, bloc 0
    C_d = np.zeros((16, 128), dtype=np.int64)
    C_d[7, :64] = 2                        # double etoilee : 2x128, blocs 7|8
    C_d[8, :64] = 2
    V_a = EBVTProbe.tree_metrics(C_a)["V"]
    V_d = EBVTProbe.tree_metrics(C_d)["V"]
    ok = V_d > V_a > 0
    print(f"  [4/4] double-etoilee {V_d} > single-hub {V_a} : "
          f"{'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    print("SELFTEST", "PASS" if fails == 0 else f"FAIL ({fails})")
    return fails


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
