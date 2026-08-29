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

**Fractional coordinates.** These live on the torus `[0,1)³`, so they use a
wrapped-normal score-matching process with a geometric σ ladder
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

## Generating a structure

`ALIGNNGenerator` follows the same shape as `AlignnUnifiedCalculator`: build it
once (which loads the diffusion model and the force field a single time), then
call it repeatedly.

```python
from alignn.inverse.generate import ALIGNNGenerator

gen = ALIGNNGenerator(num_steps=200)        # default model, cached on first use

result = gen.generate("NbN", prop=16.0, num_candidates=8)
print(result.best)          # GeneratedStructure(NNb, 2 atoms, E=-17.91 eV/atom, relaxed=True)
print(result.atoms)         # jarvis Atoms, ready for anything else in ALIGNN
print(result.to_poscar())
```

Released models are listed in the ALIGNN 2.0 registry and downloaded on first
use. With no argument you get `DEFAULT_MODEL` (`csp_supercon_alex`, trained on
the larger of the two superconductor benchmarks); pass `model="name"` to pick
another or `checkpoint=` for a local file.

| model | trained on | notes |
|---|---|---|
| `csp_supercon_jarvis` | JARVIS Supercon-3D, from scratch | best single benchmark run (match 0.524) |
| `csp_supercon_jarvis_pt` | same, fine-tuned from the base | best coordinate accuracy (RMSD 0.023 Å) |
| `csp_supercon_alex` **(default)** | Alexandria DS-A/B | match 0.485, RMSD 0.028 Å |
| `csp_pretrain_dft3d` | 65k dft_3d crystals | composition-only base, for fine-tuning |

```python
from alignn.pretrained import list_alignn2_models
list_alignn2_models("generative")
```

The composition accepts a formula, a counts dict, or an explicit atom list;
`formula_units` asks for a larger cell; `prop=None` leaves the property
unconditioned. Every candidate stays available on `result.candidates`, ordered
best first.

```python
gen.generate("Nb3Sn", prop=18.0)                    # formula
gen.generate({"Mg": 1, "B": 2}, prop=39.0, formula_units=2)   # 6-atom cell
gen.generate(["Fe", "Fe", "O", "O", "O"])           # explicit atoms, no target
gen.generate("NbN", prop=16.0, active_modalities=["composition"])
```

Roughly 35 s per call on one GPU at 200 steps with 8 candidates, of which the
single relaxation is most of it; a poor candidate that the optimiser struggles
with can push that to a couple of minutes. Two knobs matter:

- **`num_candidates` is nearly free.** Cost scales with denoising steps, not
  batch size — 32 candidates take the same wall time as 1 (24 s at 1000 steps,
  4.6 s at 200, 1.3 s at 50). It is also the strongest lever on quality, which
  is why it defaults to 8.
- **`relax_top` (default 1)** decides how many candidates get relaxed. A
  single-point energy costs ~0.3 s and a relaxation ~100x that, so all
  candidates are screened cheaply and the budget is spent on the best.

Relaxation runs in-process on one reused force field. Set `relax_workers` above
1 only for large batches — a pool reloads the force field per worker, and
because it uses the spawn start method the caller must then be under an
`if __name__ == "__main__":` guard.

> The step count trades speed for fidelity. The models were trained at 1000
> steps; whether 200 or 50 preserves benchmark quality has not been measured.

## Explicit bond-angle diffusion (optional)

Everything above treats bond angles as an *input feature*: ALIGNN computes
them, the line graph propagates them, and they help the network predict where
the atoms should go. They are never themselves something the model denoises.
This section adds the option of making them one, and of letting the triplet
topology change continuously while it does. All of it is off by default —
`ALIGNNCSPDenoiser()` with no arguments is byte-for-byte the model described
above, and a test asserts that switching the angle head on leaves `eps_frac`
and `eps_lattice` unchanged.

The hypothesis under test is narrow: **does explicit three-body geometric
denoising improve crystal generation when embedded in ALIGNN's
atom–bond–line-graph hierarchy?** It is a hypothesis, not a claim — nothing
here has been trained yet.

### The angular channel

The model gains one output head, reading the line-graph feature `z` of the
same backbone that already produces the coordinate score and the lattice
noise. Per triplet it predicts the angular displacement the forward process
introduced,

```
delta_ijk = wrap( theta_ijk(f_t, L_t) - theta_ijk(f_0, L_0) )
```

trained with a wrapped smooth-L1 loss (`beta = 0.1 pi`). Both the target and
the loss are FoldingDiff's, which runs DDPM-style corruption and denoising
directly on protein bond and dihedral angles and shows that a model with an
explicit angular channel reproduces natural angular distributions.

**One honest deviation.** FoldingDiff can diffuse a genuinely persistent
`theta_t` because a protein backbone's internal-coordinate list is fixed:
residue *i* always has the same three angles. A crystal being denoised from
noise has no such list — the triplet set is a function of the coordinates and
changes as they move, so there is no independent angular state to noise. What
is implemented is the closest well-defined thing: the angular *target* is
computed on the triplet representation that exists at the current step, with
both angles evaluated on the same periodic-image identity
`(i, j, k, n_ji, n_jk)` so the difference measures the corruption of one
triplet rather than a change of neighbour. Angles are an explicit denoising
objective with their own head and their own loss; they are not an
independently-noised variable. No new SDE, angle manifold or schedule was
invented to paper over the difference.

The angular representation itself is untouched — ALIGNN's Gaussian RBF over
`cos(theta)` — so that the experiment measures the *objective*, not a change
of basis.

### Continuous topology

At large `t` the coordinates are close to uniform, and a hard neighbour-rank
rule for "does this triplet exist" is unjustified: ranks swap constantly and
the line graph jumps. `topology="radius"` replaces the kNN rule with a radius
candidate set in which every pair carries a smooth relevance

```
s_ij  = u(r_ij ; r_c)        DimeNet's polynomial envelope: u, u' and u''
                             all vanish at r_c
s_ijk = s_ji * s_jk          ReaxFF's treatment of valence angles: an angle
                             switches off smoothly as either bond dissociates
```

`u` is not re-derived — it is `CutoffPolynomial`, already in this repository
for the smooth property model. The weight multiplies the gate *before both*
sums of the edge-gated convolution's normalised average, which is the only
placement that makes a zero-weight edge exactly equivalent to a deleted one;
scaling the numerator alone would renormalise the survivors and would not be
continuous. Because `s` is exactly zero at and beyond `r_c`, restricting the
sparse graph to pairs inside `r_c` drops only terms already contributing
nothing, so a triplet can enter or leave without any jump — which is what the
tests check by sweeping an atom through the cutoff and watching the output.

The graph is rebuilt from the current coordinates and lattice on every
forward pass, so the topology follows the geometry through reverse diffusion
on its own. There is no connectivity annealing schedule, no graph-temperature
term and no learned bondness network; `r_ij(t)` evolving is the whole
mechanism.

`radius_cutoff` defaults to 5 Å — between this repository's own three-body
cutoff (3.5 Å) and DimeNet's molecular cutoff (5 Å), and close to the radius
the baseline's 12 nearest neighbours actually span, which keeps the kNN
comparison fair.

### Ablations

Ablations are the point, not an afterthought: a single "it improved" number
cannot separate the three claims. `alignn/inverse/ablations.py` holds the
configurations, `scripts/atombench/run_angle_ablation.sh` runs them with the
same split, optimiser, epoch budget and seed list across every arm.

| | angular objective | topology | angle → bond coupling |
|---|---|---|---|
| **A0** baseline | – | kNN | – |
| **A1** | yes | kNN | yes |
| **A2** | – | smooth radius | – |
| **A3** proposed | yes | smooth radius | yes |
| **A4** control | yes | smooth radius | **cut** |
| **A6** | yes | smooth radius | yes, Fourier angle basis |

Which contrast answers what:

- **A0 vs A1** — does explicit angular denoising help on its own?
- **A0 vs A2** — or is the gain just from a better-behaved noisy graph?
- **A4 vs A3** — is the benefit genuine coupling, or would auxiliary
  supervision on a shared trunk do as well? A4 keeps the angular features
  evolving and supervised but zeroes their contribution to the bond
  representation, so its coordinate/lattice pathway provably cannot see them
  (a test perturbs the angle embedding and asserts A4's output does not move
  while A3's does).
- **A1 vs A3** — the design brief's A5: hard kNN against the smooth radius
  graph. It is a comparison between existing arms, not a seventh
  configuration.
- **A3 vs A6** — does the angular basis matter? Run this *last*; mixing a
  basis change into the primary experiment would make attribution impossible.

```bash
bash scripts/atombench/run_angle_ablation.sh runs/data_jarvis runs/ablation 0 1 2
```

Given the spread already recorded above — match rate across fifteen
independently trained models spanned 0.437–0.524 — one seed per arm cannot
settle any of these. The runner defaults to three.

### Evaluation

Every existing AtomBench metric is preserved. Two mechanism metrics are added,
defined before the runs so that a favourable one cannot be picked afterwards:

- **bond-angle distributions**, generated against held-out real structures
  (KL, Jensen-Shannon, and a 1-D Wasserstein distance reported in degrees) —
  FoldingDiff's own diagnostic;
- **relaxation displacement**, how far a sample moves to reach its nearest
  ALIGNN-FF local minimum, with the volume change and energy drop — the
  proximity-to-local-minimum evaluation MatterGen uses. If explicit angular
  denoising produces locally coherent geometry, its samples should need less
  repair.

```bash
python scripts/atombench/angle_eval.py runs/ablation/A3_s0/bench.csv --relax
```

### What is deliberately not here

DimeNet's joint spherical Fourier–Bessel distance–angle basis is **not**
implemented; A6 substitutes the Fourier basis on theta this repository
already ships. The full SBF needs spherical Bessel zeros and a new
dependency, and the design brief sequences that ablation last, so it is
deferred rather than half-done. Also absent, deliberately: cross-attention,
separate networks per variable, a learned bondness classifier, a
time-dependent graph-temperature schedule, Jacobian angle forces, and any
loss whose role cannot be traced to one of the papers below.

Note that atom types are *not* diffused — the generator is conditioned on
composition — so the state is really `(F, L)` and, with this extension,
`(F, L, Theta)`. The `A` in the ablation names follows the design brief's
notation.

### References

| | |
|---|---|
| ALIGNN — atom graph + line graph, angles update bonds update atoms | Choudhary & DeCost, *npj Comput. Mater.* **7**, 185 (2021), [10.1038/s41524-021-00650-1](https://doi.org/10.1038/s41524-021-00650-1) |
| FoldingDiff — diffusion directly on bond angles; wrapped noise, wrapped smooth-L1, angle-distribution evaluation | Wu *et al.*, *Nat. Commun.* **15**, 1059 (2024), [10.1038/s41467-024-45051-2](https://doi.org/10.1038/s41467-024-45051-2) |
| Torsional Diffusion — diffusion on angular configuration spaces | Jing *et al.*, NeurIPS 2022 |
| ReaxFF — continuous distance-dependent bond order; angle terms vanish as either bond dissociates | van Duin *et al.*, *J. Phys. Chem. A* **105**, 9396 (2001), [10.1021/jp004368u](https://doi.org/10.1021/jp004368u) |
| DimeNet — cutoff envelope whose value and first two derivatives vanish at `r_c` | Gasteiger, Groß & Günnemann, ICLR 2020, [arXiv:2003.03123](https://arxiv.org/abs/2003.03123) |
| MatterGen — one score network denoising crystal variables jointly; proximity-to-relaxed evaluation | Zeni *et al.*, *Nature* **639**, 624 (2025), [10.1038/s41586-025-08628-5](https://doi.org/10.1038/s41586-025-08628-5) |
| DiffCSP — joint equivariant diffusion of lattice and fractional coordinates | Jiao *et al.*, NeurIPS 2023 |
| CrystalDiT — the warning against unnecessary multi-stream architectural complexity | Yi *et al.*, AAAI 2026, [10.1609/aaai.v40i2.37121](https://doi.org/10.1609/aaai.v40i2.37121) |

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

If you have an existing editable install from before `alignn/inverse` was
added, re-run `pip install -e .` — the editable finder is generated at install
time and will not see a new subpackage.

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
