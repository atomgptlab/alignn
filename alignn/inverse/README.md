# ALIGNN-CSP — inverse design with pure-PyTorch ALIGNN

A conditional diffusion model that generates crystal structures, using ALIGNN
as the denoiser. Given what you want (a composition, a target property, a
diffraction pattern, a micrograph — or several at once), it produces a
lattice and atomic positions.

Nothing here depends on DGL: the denoiser is built from the pure-torch ALIGNN
layers in `alignn.models.alignn_atomwise_pure`.

## How it works

Two coupled diffusion processes run on the same timestep:

**Lattice.** A lattice matrix is only defined up to a global rotation, so
rather than diffusing `L` we diffuse the rotation-invariant symmetric part
`S = (L Lᵀ)^{1/2}`, in its matrix logarithm, scaled by `N^{-1/3}`. Any point
in that 6-dimensional space maps back to a valid non-degenerate cell through
`expm`, so noise can never produce a broken lattice. Standard DDPM with a
cosine schedule.

**Fractional coordinates.** These live on the torus `[0,1)³`, so they use the
wrapped-normal score-matching process of DiffCSP with a geometric σ ladder
(0.005 → 0.5). The network predicts the σ-scaled score, which tends to `-z` as
σ→0 and to `0` as the distribution approaches uniform — well conditioned at
both ends.

Atom types are held fixed, which makes generation *crystal structure
prediction*: place a known set of atoms in an unknown cell.

### The denoiser

Everything the network sees is invariant under the full symmetry group — RBFs
of minimum-image distances, Fourier features of fractional differences, and
bond-angle cosines — so no equivariant machinery is needed. The graph is
fully connected within the cell (cells are small, and at high noise a radius
graph is meaningless anyway), with periodic self-edges so that a one-atom cell
still has pair information. The ALIGNN line graph is built over each atom's
`knn` nearest edges, which is what carries **bond angles** — the three-body
information distance-only denoisers (CSPNet, CDVAE, FlowMM) never see.

One subtlety worth knowing about. The coordinate score is *not* read off a
`MLP(node_feature) → 3` head. ALIGNN's edge-gated convolution lets an edge
feature only *gate* a source-node feature, so the aggregated message carries
magnitude much more readily than orientation, and such a head does not learn
at all — ALIGNN-FF avoids the issue entirely by never emitting a vector
(forces come from differentiating a scalar energy). Instead the score is
assembled from the edge vectors themselves:

```
score_i = Σ_j w_ij(h_i, h_j, y_ij) · Δf_ij
```

which is direction-correct by construction and invariant to a global shift of
all coordinates.

## Conditioning

Conditioning is pluggable. Each modality is a `Conditioner` producing one
`(B, hidden)` vector; `MultiModalConditioner` sums the active ones. Every
modality owns a learned *null* embedding and is dropped independently during
training, which is what lets one checkpoint serve any subset of conditioning
at sampling time via classifier-free guidance.

| type | input | encoder |
|---|---|---|
| `scalar` | one number — Tc, band gap, formation energy | sinusoidal embedding + MLP |
| `composition` | 118-long element count/fraction vector | MLP |
| `vector` | 1-D signal — XRD pattern, DOS, spectrum | dilated 1-D CNN, mean+max pooled |
| `image` | 2-D map — STEM/HAADF, diffraction image | strided 2-D CNN |

The `vector` encoder is a dilated CNN rather than a dense layer because
diffraction information lives in peak *positions and spacings*; the mean+max
pooling keeps the sharp Bragg peaks that mean-pooling alone washes out. The
`image` encoder standardises per-image, since detector gain and exposure vary.

Declare them with a plain dict:

```python
from alignn.inverse.model import ALIGNNCSP

model = ALIGNNCSP(
    denoiser_config={"hidden_features": 256, "alignn_layers": 3},
    conditioner_spec={
        "Tc":          {"type": "scalar", "mean": 3.68, "std": 4.75},
        "composition": {"type": "composition"},
        "xrd":         {"type": "vector", "input_length": 1000},
        "stem":        {"type": "image"},
    },
)
```

Modalities are looked up by name in the batch, except `composition`, which is
derived from the atom types already present. A modality missing from a batch
contributes its null embedding, so a partially-labelled dataset trains without
special-casing.

At sampling time, pick which ones to guide on:

```python
sample(model, schedule, normalizer, batch, guidance=2.0,
       active_modalities=["xrd"])          # structure from a pattern alone
```

## Relax and rank

`relax_rank.py` refines candidates with the pretrained ALIGNN force field.
This matters because reconstruction benchmarks score against DFT-relaxed
references, which sit at local minima of the potential energy surface: a
diffusion sample lands *near* such a minimum, and the force field walks it the
rest of the way. The resulting energy also gives a physically meaningful way
to choose among several candidates for one composition — the standard
crystal-structure-prediction recipe, and something a standalone generator
cannot do.

## Usage

```bash
# 1. build a split (needs pymatgen)
python scripts/atombench/prepare_data.py --output runs/data_jarvis

# 2. train
python -m alignn.inverse.train_csp \
    --data-dir runs/data_jarvis --output runs/model --epochs 3000

# 3. generate + write an AtomBench CSV
python scripts/atombench/generate_benchmark.py \
    --checkpoint runs/model/best_model.pt --data-dir runs/data_jarvis \
    --output-csv runs/bench/alignn_csp.csv \
    --num-candidates 8 --relax cell --rank energy

# 4. score
bash scripts/atombench/score.sh runs/bench/alignn_csp.csv
```

`scripts/atombench/run_ablation.sh` runs the four pipeline variants (raw /
rank / relax / full) so you can see what each stage contributes.

## What actually moved the numbers

Findings from the AtomBench runs, recorded so they are not rediscovered:

**The candidate pool is the strongest single lever.** Going from 1 sample per
target to 32 (with a cheap energy prescreen, then relaxing the top 4) took
match rate from 0.223 to 0.524 on JARVIS Supercon-3D. Relaxation alone, on one
sample, barely moved it (0.243); selection is doing most of the work and
relaxation sharpens what selection picks (RMSD 0.29 -> 0.056).

**Symmetrisation fixes the lattice-angle metric almost entirely.** Angle MAE is
measured after Niggli reduction, which is discontinuous — a nearly-cubic cell
off by half a degree can reduce to a different basis and contribute an
enormous error. Idealising to the detected space group took angle MAE from
15.9 to 8.4 and KLD from 0.030 to 0.018, with match rate unchanged. Choose the
tolerance on validation (see `symmetrize_predictions.py --sweep`).

**The line graph buys score-fitting, and probably precision, not recall.**
Against a control that deletes the angular channel and spends the same budget
on pair-graph depth (nine convolution blocks in both arms, parameters matched
to within 1%), six models per arm — three seeds on each of two hardware
platforms:

| | A: line graph | B: no line graph | p |
|---|---|---|---|
| denoising val loss | **2.011 ± 0.018** | 2.351 ± 0.007 | 0.002 |
| coordinate RMSD (Å) | 0.031 ± 0.012 | 0.044 ± 0.011 | 0.18 |
| ccRMSD | 0.506 ± 0.014 | 0.522 ± 0.019 | 0.13 |
| match rate | 0.4709 ± 0.0286 | 0.4709 ± 0.0367 | 1.00 |

The loss gap is unambiguous — the twelve runs do not overlap, and the gap
reproduces to three decimals on a second machine. Downstream, RMSD and ccRMSD
trend the right way but do not reach significance at this sample size, and
match rate does not move at all. A first pass with only three seeds showed
RMSD at 0.030 ± 0.001 vs 0.048 ± 0.013 and looked decisive; that tightness was
a small-sample artifact. Angles cost 2.4x per training step. Reproduce with
`--alignn-layers 3 --gcn-layers 3` against `--alignn-layers 0 --gcn-layers 9`.

**Validation loss only partly tracks generation quality.** A fine-tuned model
with a clearly better val loss (1.90 vs 2.07) scored *worse* on every benchmark
metric except RMSD, and the ablation above shows a 14.5% loss gap producing a
non-significant RMSD gap and zero change in match rate. Loss predicts how precisely atoms are
placed, not how often the right structure is found.

**Report error bars.** Across fifteen independently trained models the match
rate on 103 JARVIS targets spanned 0.437-0.524, a spread of nine structures. Single
runs on splits this size can invert a comparison; only order-of-magnitude coordinate
differences were safely resolved here.

**Basis-permutation augmentation hurt on the small split.** Despite being a
correct symmetry of the problem, it consistently cost a little accuracy on
847 training crystals — the model spends capacity covering 48 relabellings
of each crystal. It is `--augment 1` by default for larger corpora; set
`--augment 0` to reproduce the best small-data result.

**Pretraining helps where the target split is large.** On Alexandria
(6603 training crystals) pretraining on 65k JARVIS crystals improved match
rate 0.413 -> 0.485; on the 847-crystal JARVIS split it did not help.

### A leakage caveat worth knowing

18.4% of the JARVIS Supercon-3D test targets, and 15.4% of the Alexandria
ones, have a StructureMatcher-identical counterpart elsewhere in JARVIS
dft_3d under a *different* material id. Any model pretrained on a corpus
drawn from that database can reach those targets by recall rather than
generation, so pretrained results should be reported alongside the
non-leaked subset. `check_pretrain_leakage.py` measures the overlap and
`filter_leaked.py` produces the restricted CSV.

### Reading the training log

Watch the per-noise-level line, not the aggregate coordinate loss:

```
frac vs baseline: [1-300) 1.032->0.766  [300-600) 1.032->0.798  ...
```

Each bucket shows the predict-zero baseline and what the model achieves. The
aggregate is a poor signal — most of it comes from small-σ steps whose target
is near-unit-variance noise, so a model that has learned a great deal and one
that has learned nothing look nearly identical.
