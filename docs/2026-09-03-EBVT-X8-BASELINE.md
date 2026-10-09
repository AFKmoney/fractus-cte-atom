# EBVT sur Fractus — baseline X8 merged (103M tokens)

Date : 2026-09-03 — point 0 de la courbe d'échelle (103M → 430M → … → 4.2B).

## L'instrument

`scripts/ebvt_probe.py` — transposition du EBVT (white papers du projet
EBVT, `EBVT-001`) sur l'arbre d'activation de Fractus.

Graphe : arbre « caterpillar » — chaine des 16 blocs (hubs) + 128 spokes
experts par bloc. Signal de cut b' = le CUT d'utilisation :

- spoke (b, e) : b' = C[b,e] (events d'activation de l'expert sur la fenetre)
- arete de chaine (k-1 | k) : b' = min(L_k, M − L_k) (L_k = events blocs 0..k-1)

Metriques (definitions exactes du projet EBVT) :

- **V** = somme des paires d'aretes adjacentes |b'(e) − b'(f)|
- **H** = max b' (hauteur de la hierarchie de specialisation)
- **P_t** = # paires adjacentes avec exactement une arete b' ≥ t ;
  identite layer-cake **V = Σ_t P_t** (self-check a chaque report,
  EBVTError si casse — 200/200 matrices aleatoires au selftest)
- persistence = #t avec P_t>0 / H ; P1 ; support (# experts utilises) ;
  dominance = max count / total

Non-invasif : forward-hooks `register_forward_hook` sur les 16
`PhaseRoutedMoE` ; le hook re-calcule `_compute_gates(phases).topk(k)` sur
les memes phases que le forward → selection **identique au moteur par
construction**. Zero ligne modifiee dans le package `fractus/`.

Calibration : le compteur natif du moteur (`engine._expert_hits`, bloc 0
seul) sert de reference de cross-check. ATTENTION : ce compteur est
**CUMULATIF** — `reset_thought()` ne l'efface pas (il l'initialise seulement
s'il est absent ; il n'est remis a zero que par `maybe_grow`). Le probe
travaille donc en convention DIFF : snapshot du compteur natif au
`probe.reset()`, difference comparee aux comptes du probe au `report()`.

## Protocole baseline

- ckpt : `checkpoints/x8run/FRACTUS_1B_X8_MERGED.pt` (103.3M tokens/cerveau,
  8 cerveaux, ~826M total — 20 % du budget 4.2B)
- chargement : `ContinuousThoughtEngine.from_pretrained` +
  `apply_kuramoto_routing_fix` (etat frais, tel que le space le charge)
- fenetre : 1 generation `generate_with_surgery` (config du harnais :
  moe T≥3.0, leak 0.9+0.08·N, ban 20, freq 1.2, top-150 @T1.15), prompt
  5 tokens + 48 nouveaux, seed 42, CPU, torch 2.5.1
- `probe.reset()` juste apres `reset_thought(1)` (meme convention de fenetre
  que le compteur natif)
- reproductibilite : **3 executions independantes → chiffres identiques**
  (V, H, P1, pers, supp, dom, profil), le texte genere variant d'une
  execution a l'autre → le routage est dans un **regime d'attracteur
  quasi-deterministe** a 103M (seul l'echantillonnage final des tokens est
  bruité ; les selections top-2 des experts sont robustes). C'est une
  propriete favorable pour en faire une horloge : le paysage EBVT est stable.

## Resultats (point 0)

```
M (total events)   = 1696  (= (5 prompt + 48 gen) x 16 blocs x top-2)
V                  = 1,948,686
H                  = 848
P1                 = 8,054
persistence        = 1.0
support            = 34 / 2048 experts (1.7 %)
dominance          = 0.031  (expert max = 53 events)
block_events       = 106 x 16 (uniforme par construction : chaque bloc
                     traite les 53 ticks, 2 selections par tick)
cross-check natif  = match exact (bloc 0, max_abs_diff = 0)
```

Profil P_t (downsample, t → P_t) :

```
t:   1    2     8     37     74     111    221    332    442    553    663   774  848
P: 8054  7933  7933  7812   3840   3330   2818   2306   1794   1282    770   258  258
```

Verification interne : le plateau final 258 = exactement 2 blocs centraux ×
129 paires frontiere (les 2 seules aretes de chaine actives au-dessus de
t=742, chacune en frontiere avec ses 128 spokes + la arete de chaine
voisine inactives). La numerique de l'instrument est doublement verifiee
(selftest + donnees reelles).

## Interpretation honnete (a lire avant de tirer des conclusions)

1. **H est epingle par le protocole ici** : H = M/2 = 848, car l'arete de
   chaine centrale porte min(L, M−L) = 848 et tous les blocs traitent les
   memes ticks. A longueur de generation fixe, H ne discrie pas le contenu.
   La discrimination sur la courbe d'echelle viendra de : **V, support,
   dominance, et la forme du profil au-dessus du plancher de chaine (258)**.
   (H redevient interessant quand le protocole change de longueur — la
   generation longue 192 tok fera croitre M, et H suivra M/2 : c'est un
   controle, pas une mesure de contenu.)
2. **persistence = 1.0 est aussi epingle** : la chaine de 15 aretes aux
   b' distincts et croissants vers le centre met une frontiere a chaque
   seuil t ≤ 848. La question de contenu n'est pas « y a-t-il une frontiere »
   mais « combien de frontiere y a-t-il a chaque echelle » = le profil P_t.
3. **Ce que la baseline dit du cerveau a 103M** : l'usage des experts est
   **shallow et dispersif** — 1.7 % des experts actifs, le plus utilise
   n'ayant que 3.1 % des events, et le profil cascade (7933 → 3840 →
   3330 → … → 258) montre une hierarchie de specialisation faible : il y a
   une structure (loin de V=0, loin d'un spike unique), mais peu de
   profondeur. C'est le portrait d'un cerveau qui babille, coherent avec
   l'etat connu (babillage mesure, topic_hits 0.2–0.3).

## Le kill-test (engagement, consigne ici)

Si, de 103M → 430M (prochain merge), **le support stagne (≲40/2048), la
dominance ne baisse pas, et le profil ne s'approfondit pas au-dessus du
plancher de chaine** — pendant que topic_hits monte — alors EBVT mesure la
chose wrong : les phrases seraient porte par un autre mecanisme que la
specialisation expert, et l'horloge structurelle est a revoir (pas le
contenu du claim « 4B pour parler »).

Prediction attendue a 430M si l'hypothese de specialisation est bonne :
support en hausse nette, dominance en baisse, cascade du profil qui se
remplit vers les hauts t (plus d'experts a 20+ events), V en hausse.

## Reproductibilite

```
# selftest (numpy seul, sans model)
python3 scripts/ebvt_probe.py --selftest

# smoke sur le model reel (load ~120 s, 3 generations ~2-3 min sur CPU
# sandbox 2 vCPU ; chiffres identiques attendus)
PYTHONPATH=<repo space> python3 scripts/ebvt_smoke.py <path ckpt x8run>

# harnais complet (les 6 segments portent maintenant leur EBVT)
PYTHONPATH=<repo space> python3 scripts/probe_ciel_ouvert.py <path ckpt x8run>
```

Fichiers : `scripts/ebvt_probe.py` (instrument + selftest),
`scripts/ebvt_smoke.py` (validation sur model reel),
`scripts/probe_ciel_ouvert.py` (patche : import + attach + reset/report
dans gen_surgical, S1, battery, S6 continu — zeronisation `fractus/`).
