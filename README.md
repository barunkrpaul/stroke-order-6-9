# Stroke order resolves the six and nine ambiguity

Code and results for the paper *Stroke order resolves the six and nine ambiguity in
rotation-invariant handwritten digit recognition* by Barun Kr. Paul, Ujjal Kumar Das and
Kinsuk Giri (submitted to Pattern Analysis and Applications).

The paper shows that any rotation-invariant image feature gives the same value to a 6 and
to the 9 obtained by turning it through 180 degrees, and proposes a descriptor of the
recorded pen trajectory (signed turning angles, normalised lengths, scaled distances and
signed areas) whose invariance is exact and whose ordering separates the pair. On the
writer-independent split of Pendigits it gives 98.75 per cent at every one of 36 test
rotations and 98.60 per cent on the digits 6 and 9.

## Layout

| Folder | Contents |
|---|---|
| `src/` | one script per experiment, the descriptor (`feats.py`), the renderer and the notebook |
| `results/` | the JSON files written by the scripts; every number in the paper comes from them |
| `paper_assets/` | the LaTeX tables, the macro file and the figures generated from `results/` |

## Running

```bash
pip install -r requirements.txt
cd src
python get_data.py          # Pendigits and MNIST
python final2.py            # proposed descriptor, 5 seeds, ablations, McNemar
python j_makefigs.py        # rebuilds paper_assets/ from the JSON files
```

The other scripts reproduce the remaining tables, viz. `sweep.py` and `imgexp.py`
(trajectory and image baselines), `zern.py` (exact moment descriptor), `j1_mnist.py`
and `j1b_mnist.py` (MNIST control), `j2_seq.py` (sequence models), `j3_order.py`
(stroke order ablation), `j4_robust.py` (noise, missing points, reflection),
`j5_sci.py` (winding-direction rule on MNIST skeletons) and `j6_offline2.py`
(recovery from offline images). The experiments were run on two CPU cores; no GPU is
needed. Each script writes its JSON file in `src/`; move it to `results/` to replace
the stored copy. `j_makefigs.py` reads from `results/` (or the folder named by the
`RES` environment variable) and writes to `paper_assets/` (or `OUT`).

## Data

Pendigits: Alpaydin E, Alimoglu F (1996), UCI Machine Learning Repository,
https://doi.org/10.24432/C5MG6K. MNIST: LeCun et al. (1998). Neither dataset is
redistributed here.

## Licence

MIT, see `LICENSE`.
