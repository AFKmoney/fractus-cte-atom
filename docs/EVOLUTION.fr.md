# Fractus — journal d’évolution (horodaté)

Chronologie du **système**, pas de la mythologie.  
Repos : `thefinalboss/fractus-cte`, `fractus-p0`, `fractus-vorax`, `fractus-datasets`.  
Dernière mise à jour : **2026-08-27**.

Les comptes de tokens sont des **snapshots de manifeste**, pas l’horloge live du pod.

---

## 2026-04 — graine

- Ligne mathématique K3 / CM (séparée).
- Idée Fractus : la pensée est continue, accumulante, oscillatoire — pas un seul forward.

## 2026-07 — architectures en parallèle

- CTE : attention linéaire + Kuramoto + MoE phasé, multi-blocs.
- Palimpseste-max : mémoire hypervectorielle append-only, CPU, HF.
- Vorax (conception) : cerveau CTE scellé + organes qui **écrivent** (`.kn`) sans réentraîner le 1B.

## 2026-08-12 — doctrine d’entraînement

- Note Chinchilla : pas d’empilement d’epochs « parce que le papier le dit ». Une passe réelle + ingest Vorax.

## 2026-08-13 → 15 — premier burn 4×5090

- ~1k tok/s/GPU.
- Loss qui descend dans les 30.
- Découvertes : collapse mono-token, train ≠ gen length-1, chirurgie mid-train, mean-merge same-shape.
- Space HF : échecs torch / `audioop` / Gradio sur CPython 3.13.

## 2026-08-16 — panne

- Incident pod. Reprise sur le dernier `.pt` HF, pas à 0.
- Recette 8 GPU.

## 2026-08-17 → 18 — phase 2, freeze, Vorax

- Shards npy depuis `fractus-datasets`.
- Freeze ≈ **15–17M tokens/GPU** sur shards ~430M (~**4 %** de cette passe).
- Vorax repo : compiler, traces, Hebbian, spawn.

## 2026-08-19 — goulot Kuramoto

- Arc de routing ~25°/360°, \(r \approx 0.01\), experts morts.
- Pas « attendre les tokens ». Température de gate, échelle \(\omega\), LB vraie.
- Doc mini-CPU : un petit Fractus ≠ progrès 1B ; merge mini→1B invalide.

## 2026-08-21 — mauvais cluster

- Mix 3090 / A4000 / 5060 Ti / A2000 12GB. 12GB ne tient pas le 1B B=2.
- Pas d’offre 8× sur QuickPod à ce moment. Crédit laissé ; attendre du 24GB+ identique.

## 2026-08-22 → 26 — cœur ouvert + x8

- Attention chunked / cumsum, CE chunkée, `BLOCK_CKPT`, ckpt atomiques.
- White paper v2 + `arxiv/main.tex`.
- X8_MANIFEST : ~**35M tokens/GPU**, status `x8_running`.

## 2026-08-27 — P0, v4, DiffusionBlocks, lignée nommée

- Repo `fractus-p0` : routing P0, trainer v4, PREFIX vs CARRY mesuré, DiffusionBlocks natif (expérience).
- Phrase officielle : Fractus = **RNN d’attention linéaire**, pas un transformer.
- Ouvert : decay sur \(S\), DDP vs merge MoE, MI(\(\bar\theta\), token) vs tick, PPL apparié, split d’éval par source.

---

## Ce que l’évolution n’est pas

- Pas un AGI livré.
- Pas 4,2B tokens digérés.
- Pas « merge = 8× plus malin ».
- Pas « Kuramoto réglé ».

## Suite ops

x4 24GB+ homogène > attendre un x8 fantôme.  
Resume manifeste. Corps P0+v4. Gate PREFIX unique@40.  
Vorax pour le savoir. Baseline transformer sur held-out propre avant tout papier d’ampleur.
