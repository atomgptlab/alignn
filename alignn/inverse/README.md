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

## Explicit bond-angle diffusion — reference

> **Written by Claude (Anthropic)** on the `angle-diffusion` branch, at the
> request of the repository owner. This section is meant as the working reference for
> the group: it states exactly what each ablation contains, which part of each
> design decision comes from published work and which part is ours. Every
> equation below was checked against the code in `alignn/inverse/`, and the
> numbers quoted from the graph construction were measured, not estimated.
> Nothing here has been trained. Three bibliographic details are flagged
> inline as needing a check against the primary sources before they go into a
> manuscript.

### Overview: what this is, why it might matter, what we are testing

**The one-sentence version.** ALIGNN's distinguishing feature is that it looks
at bond *angles*, not just bond lengths; our generative model already uses
angles to *read* a noisy crystal, but never asks the network what the angles
*should be*. This branch makes bond angles a thing the model denoises, and
asks whether that helps.

**Where angles currently sit.** ALIGNN-CSP is a diffusion model over
fractional coordinates and the lattice, conditioned on composition
[DiffCSP, MatterGen]. Two processes run on a shared timestep: the lattice
follows DDPM in a rotation-invariant log-symmetric representation
[DDPM, iDDPM], and the coordinates follow a wrapped-normal score process on
the torus [NCSN, ScoreSDE]. The denoiser is ALIGNN [ALIGNN], so it builds a
line graph and propagates bond angles alongside bond lengths. But angles enter
only as an *input feature*. The network is asked "given this noisy structure,
which way should each atom move?" and it consults the angles to answer. It is
never asked "given this noisy structure, what should the angles be?" Angles
are an input, never an output.

**Why anyone should care.** What separates a real crystal from a plausible
cloud of atoms is mostly *local coordination geometry* — tetrahedra,
octahedra, specific angular motifs — and that is a three-body property. A
purely coordinate-space objective supervises each atom's displacement more or
less independently and only reaches angular structure indirectly, through
whatever correlations the network happens to learn. There is a direct
precedent for doing better: FoldingDiff [FoldingDiff] generates protein
backbones by representing them in internal coordinates and running the
diffusion process *on the angles themselves*, and it reproduces natural
bond-angle distributions in a way coordinate-space models do not. Torsional
diffusion [TorsDiff] makes the same general point for molecular conformers:
when the interesting degrees of freedom are angular, define the process on the
angular space.

**The specific puzzle in our own data.** The ablation already recorded in this
README is the reason this is worth compute rather than idle curiosity. Deleting
the line graph and spending the same parameter budget on pair-graph depth costs
a large, unambiguous amount of denoising validation loss — 2.011 ± 0.018
against 2.351 ± 0.007 across six models per arm, p = 0.002, reproducing on two
machines — and yet leaves match rate *exactly* unchanged at 0.4709. Angular
information demonstrably helps the network fit the score, and demonstrably does
not help it find the right structure more often. One reading of that gap is
that angles are being used to interpolate rather than to constrain: the network
reads them, gets a better local fit, and still lands in the wrong basin. If
that reading is right, *supervising* angles rather than merely observing them
is the natural intervention, and this branch is the test of it.

**The second idea: connectivity should not click.** There is a separate,
smaller problem visible in the current code. To build the line graph, we choose
each atom's twelve nearest bonds and call those the ones that form angles. That
is a sensible rule for a real crystal. It is a poor rule in the middle of
reverse diffusion, when coordinates are near-uniform: neighbour ranks swap
constantly, so the set of triplets — and therefore the function the network is
computing — jumps discontinuously from one denoising step to the next, for
reasons that have nothing to do with chemistry. Physical chemistry solved this
problem long ago. ReaxFF [ReaxFF] gives bonds a continuously varying order and
lets an angular energy term fade smoothly to zero as either of its bonds
dissociates; DimeNet [DimeNet] gives graph neural networks a cutoff envelope
whose value and first two derivatives all vanish at the cutoff radius, so an
edge can enter or leave the neighbour list without any discontinuity. We adopt
both, so that the effective line graph *crystallises* as the geometry does
rather than flickering.

**What we are testing, stated as falsifiable questions.**

1. Does an explicit three-body denoising *objective* improve crystal
   generation, over and above angles being an input feature?
2. Does continuously varying interaction topology matter in the high-noise
   regime, independently of any angular objective?
3. If something improves, is it because the angular representation is
   *coupled back* into the coordinate/lattice pathway, or would any auxiliary
   task on a shared trunk have done the same?

**What a negative result looks like, agreed in advance.** If A2 (smooth
topology, no angular objective) captures the whole gain, then the angular
objective is not the story and we should say so. If A4 (angular objective with
the coupling severed) matches A3, then the benefit is generic auxiliary
supervision and the three-body claim fails even if the benchmark number goes
up. Both outcomes are publishable and both are cheap to reach; the ablation
suite is designed so that we cannot avoid learning which one we are in.

**Scope discipline.** This stays an ALIGNN. There is no transformer, no second
network, no cross-attention, no learned bond classifier — CrystalDiT
[CrystalDiT] is a recent reminder that multi-stream architectural complexity
is usually not what buys crystal-generation performance, and MatterGen
[MatterGen] denoises every crystal variable through one shared score network.
The angular channel is one extra output head on the representation that
already exists.

### Notation

```
N            atoms in a structure (batched: all crystals concatenated)
f ∈ [0,1)^3  fractional coordinates, per atom
L ∈ R^3x3    lattice matrix, rows are lattice vectors, cart = f L
t            diffusion timestep, shared by the lattice and coordinate processes
i, j, k      atom indices; a triplet is the angle at the shared atom
(i→j)        a directed pair ("bond"), a node of the line graph
r_ij         minimum-image Cartesian vector from i to j
θ_ijk        interior bond angle at j, in [0, π]
s_ij         smooth pair relevance in [0,1]
s_ijk        smooth triplet relevance in [0,1]
h, y, z      atom, bond and triplet features inside the network
```

Atom types are **not** diffused: the generator is conditioned on composition,
so the state is `(f, L)` and, with this branch, `(f, L, Θ)`. The `A` in the
ablation names follows the design brief's `(A, F, L, Θ)` notation and should
not be read as a claim that species are generated.

### What A0 already contains

Stating this precisely matters, because every arm is a delta against it.

**The two inherited processes.** The lattice is diffused not as `L` but as
`log S` where `S = (L Lᵀ)^{1/2}`, scaled by `N^{-1/3}` and flattened to a
Frobenius-preserving 6-vector; this is rotation-invariant, and `expm` maps any
point in `R^6` back to a valid cell, so noise cannot produce a degenerate
lattice. Standard DDPM [DDPM] on that 6-vector with the cosine ᾱ schedule
[iDDPM]; the network predicts ε. Fractional coordinates live on the torus, so
they use a wrapped-normal variance-exploding score process [NCSN] with a
geometric σ ladder from 0.005 to 0.5, `f_t = wrap(f_0 + σ_t z)`; the network
predicts the σ-scaled score, computed as a softmax-weighted mean over 11
periodic images, and sampling uses a predictor–corrector scheme [ScoreSDE].
Conditioning uses classifier-free guidance [CFG] with each modality dropped
independently. **None of this is touched by any arm below.**

**The graph.** The pair graph is *dense* within each cell — every ordered pair
including self-pairs, resolved to the minimum image over the 27 offsets in
{−1,0,1}³, with self-pairs forced off the zero image so a one-atom cell still
carries information about its own translates. Cells are small, and at high
noise a radius graph on the pair channel would be arbitrary, so density is a
deliberate choice rather than an oversight.

**The line graph.** A bond may participate in triplets if it is among the `knn`
= 12 shortest bonds incident on its destination atom (`_knn_mask`, ranked by
destination). For each surviving bond `A = (u→v)` and each surviving bond
`B = (v→w)`, one line-graph edge is emitted; the shared atom is `v`. Triplets
are **not** deduplicated and back-tracking triplets `i→j→i` are **not**
excluded — measured on a 3-atom dense graph, 9 of 27 triplets are
back-tracking, each contributing exactly `cos θ = −1`. This is inherited
behaviour and is identical in every arm, but it matters when reading generated
angle histograms, which will therefore carry a spike at 180°.

**The angular feature.** `cos θ` from `torch_bond_cosines`, expanded in 40
fixed Gaussian RBFs over [−1, 1], then two MLP layers to hidden width — ALIGNN's
own representation [ALIGNN], unchanged in every primary arm.

**The convolution.** Edge-gated graph convolution [GatedGCN] as ALIGNN uses it:

```
m_ij  = W_s h_i + W_d h_j + W_e y_ij
σ_ij  = sigmoid(m_ij)
h'_i  = W_1 h_i + ( Σ_j σ_ij ⊙ W_2 h_j ) / ( Σ_j σ_ij + 1e-6 )
y'_ij = y_ij + SiLU(LayerNorm(m_ij))
```

The same class is instantiated twice per ALIGNN layer: once on the atom graph
(atoms ← bonds) and once on the line graph (bonds ← triplets), which is how
`Θ → L(G) → G` propagation happens. Note `LayerNorm`, not `BatchNorm` — the
normalisation is per element, so no cross-edge statistic exists. That fact is
load-bearing for the continuity argument in §"Continuous topology" below.

**The heads.** The coordinate score is *not* an MLP on the node feature. Because
the edge-gated convolution lets an edge feature only gate a source-node feature,
an aggregated message carries magnitude far more readily than direction, and
such a head does not learn. Instead the score is assembled from the edge vectors
themselves, `score_i = Σ_j w_ij(h_i, h_j, y_ij) · Δf_ij`, which is
direction-correct by construction and invariant to a global shift. The lattice
head is an MLP on the mean-pooled atom feature.

### Addition 1 — the angular denoising channel

#### What we take from FoldingDiff

FoldingDiff [FoldingDiff] establishes that bond angles can be the variables of
a diffusion model rather than a derived quantity: it represents a protein
backbone in internal coordinates, corrupts those angles with wrapped noise,
trains a network to predict the angular noise, and shows the resulting samples
reproduce natural angular distributions. We take three specific things:

1. **that angles can be a denoising target at all** — the conceptual move;
2. **the wrapped residual** — an angular error is only ever defined modulo
   2π, so the loss must wrap before it penalises;
3. **the loss functional form** — smooth L1 on the wrapped residual, with
   `β = 0.1π`.

#### What we implement

The network gains exactly one head: a two-layer MLP on the line-graph feature
`z` of the shared backbone, zero-initialised on the output layer so training
starts from a silent prediction (matching how the existing coordinate and
lattice heads are initialised). Per triplet it predicts a scalar `δ̂_ijk`.

The target is the angular displacement the forward process actually produced:

```
θ_t   = θ_ijk( f_t , L_t )      angle at the noised geometry
θ_0   = θ_ijk( f_0 , L_0 )      angle at the clean geometry, same triplet
δ     = wrap( θ_t − θ_0 ) ∈ [−π, π)
```

and the loss is the relevance-weighted wrapped smooth L1

```
L_ang = Σ_T  s_ijk · SmoothL1_β( wrap( δ̂ − δ ) )  /  Σ_T  s_ijk ,   β = 0.1π
```

added to the existing objective with a fixed weight `angle_weight` (default
1.0, alongside `frac_weight` 10.0 and `lattice_weight` 1.0).

#### Deviation 1 (important, and unavoidable) — the angular state is induced, not persistent

**This is the main methodological caveat of the branch and should be stated in
any write-up.** FoldingDiff can diffuse a genuinely persistent `Θ_t` because a
protein backbone's internal-coordinate list is *fixed*: residue *i* always has
the same three angles, so the angles are legitimate independent state
variables. A crystal being denoised from noise has no such list — the set of
triplets is a *function of the coordinates* and changes as they move. There is
therefore no independent angular variable to noise, and no forward process
`q(Θ_t | Θ_0)` to write down.

**What we do instead, and why.** The angular *target* is computed on the
triplet representation that exists at the current step. Angles remain an
explicit denoising objective with their own head and their own loss; they are
not an independently-noised variable. Concretely, the process is
`(f, L)`-driven and `Θ` is read off it, so the model is trained to report the
angular component of a corruption it is simultaneously being asked to undo in
Cartesian terms. This is the fallback the design brief specifies for exactly
this situation, and it is chosen over the alternative of inventing an angular
SDE with no literature standing.

**Consequence to be honest about.** Because `Θ` is induced, `L_ang` is not an
independent diffusion loss with its own ELBO interpretation; it is a
geometrically-structured auxiliary objective on the same forward process. Any
claim in a paper should be phrased as "explicit angular supervision derived
from the joint process", not "we diffuse angles".

#### Deviation 2 (ours) — fixed periodic-image identity for the target

**Not from any cited source.** A triplet's identity in a periodic crystal is
`(i, j, k, n_ji, n_jk)`, where the `n` are integer cell offsets. When we
resolve the noised geometry to minimum image we record the integer offset
actually chosen,

```
n = offset_argmin − round( f_t[dst] − f_t[src] )
```

and compute `θ_0` by applying *that same* `n` to the clean coordinates,
`Δf_0 = f_0[dst] − f_0[src] + n`, rather than re-running minimum-image
resolution on the clean structure. The reason is that the two give different
answers at large `t`: re-resolving would compare the noised angle at one
neighbour against the clean angle at a *different* neighbour, so the target
would mix "this triplet bent" with "this is now a different triplet". Fixing
`n` makes `δ` measure the corruption of one specific triplet. As σ → 0 the two
constructions coincide, and a test asserts `|δ| < 1e−4` when `f_t = f_0`.

#### Deviation 3 (ours) — relevance weighting of the loss

**Not from FoldingDiff**, which has no per-angle weights because its angle set
is fixed. Ours is `Σ s·ℓ / Σ s`. This is what makes the objective continuous
when a triplet enters or leaves the sparse graph: a triplet at the cutoff has
weight zero and contributes nothing on either side of the boundary. In kNN
arms all weights are 1 and the expression reduces to a plain mean.

#### A note on wrapping

Bond angles are in `[0, π]`, so `θ_t − θ_0 ∈ [−π, π]` already, and `wrap` is
the identity except exactly at the boundary. It is applied anyway so the
objective is the wrapped one by construction rather than by an argument about
ranges, and so the same helper serves a genuinely circular variable if a
dihedral channel is ever added. `acos` is clamped at `±(1 − 1e−7)` to keep the
gradient finite for collinear triplets, which — see the back-tracking note
above — are common; the cost is ≈ 0.03° of accuracy at the poles.

### Addition 2 — continuously weighted topology

#### What we take from DimeNet

DimeNet [DimeNet] introduces the polynomial envelope

```
u(x) = 1 + a x^p + b x^(p+1) + c x^(p+2) ,   x = r / r_c
a = −(p+1)(p+2)/2 ,  b = p(p+2) ,  c = −p(p+1)/2
u(x) = 0 for x ≥ 1
```

for which `u(0) = 1` and `u(1) = u'(1) = u''(1) = 0`. We take the envelope
exactly, and we do not re-derive it: this repository already ships it as
`CutoffPolynomial` for the smooth property model, and we import that class.

**Convention note, worth stating precisely because the two differ.** This
repository parameterises by `coeff`, which *is* the paper's `p`; the widely
used reference implementation parameterises by `envelope_exponent` and sets
`p = exponent + 1`. We use `coeff = 5`, i.e. `p = 5`. Both satisfy the
vanishing conditions (verified by autograd in the test suite); they differ only
in how fast the envelope decays — at `r = r_c/2`, `u = 0.7734` for `p = 5`
against `0.8555` for `p = 6`. Also note DimeNet's *code* folds a factor `1/x`
into its envelope so that it can multiply a Bessel basis; the polynomial above
is the paper's `u(d)`, which is the form we want for a weight in `[0,1]`.

#### What we take from ReaxFF

ReaxFF [ReaxFF] makes bond order a continuous function of interatomic
distance and multiplies every valence-angle energy term by switching factors
for *both* of its constituent bonds, so an angular interaction disappears
smoothly when either bond dissociates and appears smoothly when one forms. We
take the product structure of that gate and nothing else — no ReaxFF
parameters, no bond-order function, no element-specific tables:

```
s_ij  = u( r_ij ; r_c )                pair relevance   [DimeNet]
s_ijk = s_ji · s_jk                    triplet relevance [ReaxFF]
```

The behaviour we want during reverse diffusion follows directly: as `r_ji`
grows, `s_ji → 0` and every triplet through that bond fades out; as some other
`r_jl` shrinks, new triplets fade in. The effective line graph changes
continuously because `r(t)` changes, with no annealing schedule, no
graph-temperature term and no learned bondness network.

#### Deviation 4 (ours) — where the gate is applied, and why the placement is forced

**Not specified by either source.** ReaxFF gates a physical energy term;
DimeNet's envelope multiplies a radial basis. Neither says what to do with a
*learned, normalised, gated message*. In the edge-gated convolution the
aggregation is a normalised average, and the weight must therefore multiply
`σ_ij` **before both sums**:

```
h'_i = W_1 h_i + ( Σ_j w_ij σ_ij ⊙ W_2 h_j ) / ( Σ_j w_ij σ_ij + 1e-6 )
```

This is the only placement with the property we need. Setting `w = 0` makes the
term contribute nothing to numerator *and* denominator, so the result is
*exactly* what the same layer computes on a graph with that edge physically
removed — verified by a test that compares the two. Weighting the numerator
alone would renormalise the surviving messages and would be discontinuous in
exactly the situation the whole construction exists to avoid. Because
`LayerNorm` is per element, no batch statistic can smuggle the deleted edge
back in.

Three places carry the weight, all with the same scalar:

| where | weight | why |
|---|---|---|
| atom-graph messages | `s_ij` | pair channel fades with distance |
| line-graph messages | `s_ijk` | triplet channel fades with either bond |
| coordinate-score head, `w_ij ← w_ij · s_ij` | `s_ij` | a pair leaving the cutoff must stop pushing the atom smoothly, not abruptly |

#### Deviation 5 (ours) — the sparse graph is an exact truncation, not an approximation

**Not from either source.** Because `s` is *exactly* zero at and beyond `r_c`
— not merely small — restricting the sparse line graph to pairs with
`s_ij > 0` removes only terms already contributing exactly nothing. The radius
graph is therefore an exact sparsification of a gated dense graph, not an
approximation of it, and triplet insertion/deletion is invisible by
construction rather than by tolerance. The test suite sweeps an atom straight
through `r_c` in 121 steps, confirms the triplet count really does change
during the sweep, and asserts no step in the output exceeds 20× the median
step.

The pair graph itself stays dense in the edge list, with `s_ij` doing the
truncation numerically. That is a deliberate choice: it keeps the sparse
structure identical to A0's, so nothing about batching or indexing differs
between arms, and it removes any risk of an atom being left with no pair
edges at all in a large cell.

#### Deviation 6 (ours) — the graph is rebuilt every forward pass

Every distance, every envelope value, the `allowed` mask and the entire line
graph are recomputed inside `forward()` from the current `(f, L)`. During
reverse diffusion that means the topology is re-derived at each of the T
denoising steps at no extra bookkeeping cost. The pair index is composition-
determined and is built once. This satisfies the "rebuild the radius graph
during reverse diffusion" requirement without any sampler changes.

**Compute cost, measured.** On a 16-atom 7 Å cell the kNN construction yields
2304 triplets and the radius construction at `r_c = 5 Å` yields 3262 — about
1.4×. Triplet count scales as `Σ_j deg(j)²`, so this ratio grows with cell
size and with `r_c`; budget for it when matching wall-clock across arms.

#### The choice of `r_c`

`radius_cutoff` defaults to 5.0 Å. Justification, in order of weight: it is
close to the radius that the baseline's twelve nearest neighbours actually
span in a dense crystal, which is what makes A1 vs A3 a comparison of
*smoothness* rather than of *interaction range*; it sits just above this
repository's own three-body cutoff of 3.5 Å; and it matches DimeNet's
molecular cutoff. It is a free parameter and should be held fixed across every
arm of a comparison.

### The arms

Each arm is a keyword dict in `alignn/inverse/ablations.py`. They differ only
in the switches named; width, depth, schedule, optimiser, splits and seeds come
from the training script and must be identical across a comparison.

#### A0 — baseline

`angle_diffusion=False, topology="knn", gate_pair_messages=False,
angle_feedback=True`

The model that exists on `develop`, unchanged, as described in "What A0 already
contains". Angles are an input feature; the outputs are the coordinate score
and the lattice noise and nothing else. Its first job is to reproduce the
published numbers — match 0.485 on Alexandria, 0.524 on JARVIS Supercon-3D —
before any comparison is trusted. Its second job is as the correctness anchor
for the branch: a test loads A0's state dict into A1 and asserts `eps_frac`
and `eps_lattice` come out bit-identical, which is what establishes that the
angular channel is an *addition* to the model rather than a perturbation of
it. A0 is also what a user gets from `ALIGNNCSPDenoiser()` with no arguments,
so every released checkpoint keeps loading and behaving exactly as before.

#### A1 — angular denoising only

`angle_diffusion=True`, graph left at baseline.

Adds the angle head and `L_ang`; changes nothing about the graph. Same kNN
membership rule, same triplets, same 40-bin cosine RBF, same convolutions,
same everything, plus one two-layer MLP. Because the architectural delta is a
single head that feeds no other computation in the forward direction, a
difference between A0 and A1 is attributable to the *objective* and to nothing
else — this is the arm that answers question 1 in isolation, and it is
deliberately built on the baseline's topology so that question 1 and question
2 cannot contaminate each other. The angular information does reach the
structural heads, but only through ALIGNN's ordinary path, and only because
the loss reshapes the shared trunk. One caveat to report alongside the result:
adding a loss term changes the gradient balance, so `angle_weight` must be
fixed across arms and stated, or the comparison silently becomes a
hyperparameter search.

#### A2 — smooth topology only

`topology="radius", gate_pair_messages=True`, no angle head.

Replaces the hard kNN membership rule with a radius candidate set, gates
triplet messages by `s_ijk`, and gates atom-graph messages and the per-edge
terms of the coordinate score by `s_ij`. No angle is ever denoised; the
outputs are still just the coordinate score and the lattice noise. This arm
exists to kill the boring explanation. Making the high-noise graph continuous
is a substantial change to the model's inductive bias on its own — at large
`t` the coordinates are near-uniform and neighbour ranks swap constantly — so
if A3 beats A0 and A2 beats A0 by the same margin, the angular objective
contributed nothing and the honest conclusion is that the topology did the
work. A2 answers question 2 with the angular channel held off.

#### A3 — the proposed model

`angle_diffusion=True, topology="radius", gate_pair_messages=True,
angle_feedback=True`

Both mechanisms together, and the arm the hypothesis is about. The angular
channel is *coupled*, not bolted on: `z` reaches the coordinate and lattice
heads through ALIGNN's own hierarchy, because the line-graph convolution's
triplet feature gates the bond→bond message and the updated bond features then
gate the bond→atom message on the next layer — `Θ → L(G) → G → (f, L)` heads,
exactly the mechanism ALIGNN already provides [ALIGNN]. No Jacobian
correction, no geometric force term, no hand-designed angle-to-coordinate
update. A3 beating A0 is the headline number, but on its own it says only that
the combination helps; which half is responsible is settled by A1, A2 and A4,
not by A3.

#### A4 — the coupling control

`angle_diffusion=True, topology="radius", gate_pair_messages=True,
angle_feedback=False`

Identical to A3, including parameter count, except that the triplet weight
entering the *bond aggregation* is set to zero. Mechanically, inside the
line-graph convolution:

```
m_T = W_s m[lg_src] + W_d m[lg_dst] + W_e z         still contains z
σ_T = sigmoid(m_T) · 0                              gate forced to zero
y'  = SiLU(LayerNorm(W_1 m + 0/(0+1e-6))) + m       bonds see no angles
z'  = SiLU(LayerNorm(m_T)) + z                      angles still evolve
```

So the angular features are still computed, still updated by every ALIGNN
layer, still feed the angle head, and the angular loss still back-propagates
into the shared trunk through `m_T` — that last part is deliberate, because
gradient flow into a shared trunk is precisely what auxiliary multitask
learning *is*. What is removed is the forward path by which the angular latent
reaches the bond representation, and hence the atom, coordinate and lattice
pathway. A4 is therefore "angles supervised alongside the model" and A3 is
"angles wired into it", which is exactly the contrast question 3 asks about.
If A3 ≈ A4, the mechanism claim fails even if the benchmark number improved.

Two things to keep straight. First, the cut is verified rather than asserted: a
test perturbs the angle-embedding weights and confirms A3's coordinate score
moves while A4's stays at *exactly* zero. Second, **A4's structural trunk is
not identical to A0's** — it is A3's with one aggregation zeroed, which is a
different function from A0's full triplet aggregation. A4 is a control for A3
and must not be read as a second baseline.

#### A5 — hard kNN versus smooth radius

**Not a configuration.** This is a comparison between arms that already exist,
which is why there is no `"A5"` key. It is answered twice: A1 against A3 with
the angular objective on, and A0 against A2 with it off. Running both is worth
the compute because they can disagree informatively — if continuous topology
matters *only* when something is being denoised on the triplets, then A0↔A2
moves little while A1↔A3 moves a lot, and that pattern is itself evidence for
the coupling story rather than for topology as a standalone improvement.

#### A6 — angular basis

`angle_basis="fourier"` on top of A3.

Swaps ALIGNN's 40-bin Gaussian RBF over `cos θ` for the learnable Fourier
basis on `θ` this repository already ships (`FourierAngular`, order
`(triplet_bins − 1)//2`, giving `1 + 2·order` features). Sequenced last on
purpose: changing the angular representation at the same time as introducing
the angular objective would make attribution impossible, which is why every
primary arm keeps ALIGNN's basis untouched [ALIGNN].

**State its limit plainly.** DimeNet's joint spherical Fourier–Bessel `(d, θ)`
basis is **not** implemented. DimeNet reports that a joint distance–angle basis
is a stronger inductive bias than the raw angle [DimeNet], and that remains the
interesting version of this question, but it requires spherical Bessel zeros
and a new dependency, and the design brief defers the whole basis question to
last. A6 therefore answers "does the angular representation matter at all",
not "is DimeNet's SBF better". The latter is open work.

### Confounds and threats to validity

**Interaction range is confounded with topology smoothness in A2/A3/A4.** With
`gate_pair_messages=True` and `r_c = 5 Å`, pairs beyond 5 Å contribute exactly
zero to the atom-graph messages and to the coordinate score, whereas A0's dense
pair graph lets every atom see every other atom in the cell. So the smooth arms
differ from A0 in two ways at once: the topology is continuous *and* the
interaction range is truncated. This is what the design brief asks for
("smoothly vanishing pair interactions"), and it is defensible, but it is a
confound and should be reported as one.

It is separable without any code change. `--ablation A3 --gate-pair-messages 0`
gives smooth *triplet* topology with A0's ungated dense pair channel — the
line graph is still radius-based and `s_ijk`-weighted, but the pair channel is
untouched. If range truncation is doing the work, that arm will sit with A0;
if smoothness is, it will sit with A3. We recommend running it for A3 at
minimum before writing anything up.

**Back-tracking triplets inflate the 180° bin.** Inherited from the graph
builder and identical across arms, so it does not bias a comparison, but a
generated-versus-real angle histogram will show a large spike at 180° in both
distributions. Do not present that spike as physics.

**`angle_weight` is unswept.** It is a free hyperparameter at 1.0. A negative
result for A1 or A3 at one loss weight is weak evidence; if the primary arms
come out flat, sweep it before concluding, and report the sweep.

**Validation loss is a poor proxy here.** Already documented above in this
README: a 14.5% denoising-loss gap produced a non-significant RMSD gap and zero
change in match rate. Expect `L_ang` to fall in the angular arms *by
construction* — it is a new term being optimised — and do not report that as
evidence of anything.

**Seed spread.** Match rate across fifteen independently trained models spanned
0.437–0.524, nine structures on a 103-target split. One seed per arm can invert
any conclusion here. The runner defaults to three; more is better.

### Protocol and metrics

`scripts/atombench/run_angle_ablation.sh <data-dir> <out-dir> [seeds...]` runs
every arm with the same split, optimiser settings, epoch budget and seed list,
then generates, scores and computes the mechanism metrics.

All existing AtomBench metrics are preserved unchanged: match rate, Cartesian
RMSD, ccRMSD, lattice-parameter and lattice-angle MAE, KLD. Two mechanism
metrics are added, and **the suite was fixed before any run precisely so that a
favourable metric cannot be selected afterwards**:

**Bond-angle distributions** — generated against held-out real structures,
pooled over the split, on a common 180-bin histogram over [0°, 180°], reported
as KL, Jensen–Shannon and 1-D Wasserstein distance. The Wasserstein figure is
exact for a 1-D histogram (the integral of the absolute CDF difference) and is
in degrees, so it reads directly as "the generated angles are off by this
much"; it is calibrated to within 0.001° on a synthetic shift test. This is
FoldingDiff's own diagnostic [FoldingDiff] and is the most direct check that
the angular channel does what it claims.

**Relaxation displacement** — how far a sample must move to reach the nearest
ALIGNN-FF local minimum, as translation-corrected Cartesian RMSD, plus
fractional volume change and energy drop. MatterGen [MatterGen] evaluates
generated structures by their proximity to their relaxed counterparts; if
explicit angular denoising produces locally coherent geometry, its samples
should need less geometric repair, and this is the metric that would show it.

```bash
python scripts/atombench/angle_eval.py runs/ablation/A3_s0/bench.csv --relax
```

### Provenance summary

| design choice | source | taken from the source | ours |
|---|---|---|---|
| atom + line graph, angles→bonds→atoms | [ALIGNN] | the whole hierarchy, the cosine-RBF angular feature, the edge-gated conv | nothing — used as-is |
| angles as a denoising target | [FoldingDiff] | the conceptual move, the wrapped residual, smooth-L1 with β = 0.1π | applying it where the angle set is not persistent |
| angular target definition | — | — | **ours**: `δ = wrap(θ_t − θ_0)` on the current triplet set, with a fixed periodic-image identity |
| loss weighting | — | — | **ours**: `Σ s·ℓ / Σ s`, needed for continuity at the cutoff |
| cutoff envelope | [DimeNet] | the polynomial `u`, imported from this repo's existing `CutoffPolynomial` | using it as a *topology* weight rather than a radial-basis multiplier |
| triplet gate | [ReaxFF] | the product structure `s_ijk = s_ji·s_jk` and the fade-in/fade-out behaviour | no ReaxFF parameters or bond-order function are used |
| gate placement | — | — | **ours**: multiply `σ` before *both* sums; forced by the exact-deletion requirement |
| radius truncation | — | — | **ours**: exact sparsification (`s = 0` beyond `r_c`), not an approximation |
| one shared backbone, extra head | [MatterGen], [CrystalDiT] | joint denoising through one network; the warning against multi-stream complexity | — |
| crystal diffusion formulation | [DiffCSP], [DDPM], [iDDPM], [NCSN], [ScoreSDE], [CFG] | the inherited `(f, L)` processes, schedules and guidance | untouched by this branch |
| angle-distribution metric | [FoldingDiff] | the diagnostic | 1-D Wasserstein in degrees as the headline figure |
| relaxation-proximity metric | [MatterGen] | the evaluation | translation-corrected RMSD + volume change + energy drop |
| A4 control design | — | — | **ours**: sever coupling by zeroing the triplet aggregation weight, keeping parameters and gradient flow identical |
| A6 angular basis | [DimeNet] | the motivation for a richer basis | uses this repo's Fourier basis; DimeNet's SBF is **not** implemented |

### Bibliography

- **[ALIGNN]** K. Choudhary and B. DeCost, "Atomistic Line Graph Neural
  Network for improved materials property predictions," *npj Computational
  Materials* **7**, 185 (2021). doi:10.1038/s41524-021-00650-1
- **[FoldingDiff]** K. E. Wu, K. K. Yang, R. van den Berg, S. Alamdari,
  J. Y. Zou, A. X. Lu and A. P. Amini, "Protein structure generation via
  folding diffusion," *Nature Communications* **15** (2024).
  doi:10.1038/s41467-024-45051-2
- **[TorsDiff]** B. Jing, G. Corso, J. Chang, R. Barzilay and T. Jaakkola,
  "Torsional Diffusion for Molecular Conformer Generation," *NeurIPS* (2022).
- **[ReaxFF]** A. C. T. van Duin, S. Dasgupta, F. Lorant and W. A. Goddard III,
  "ReaxFF: A Reactive Force Field for Hydrocarbons," *J. Phys. Chem. A* **105**,
  9396–9409 (2001). doi:10.1021/jp004368u
- **[DimeNet]** J. Gasteiger (Klicpera), J. Groß and S. Günnemann,
  "Directional Message Passing for Molecular Graphs," *ICLR* (2020).
  arXiv:2003.03123
- **[MatterGen]** C. Zeni *et al.*, "A generative model for inorganic materials
  design," *Nature* (2025). doi:10.1038/s41586-025-08628-5
- **[DiffCSP]** R. Jiao, W. Huang, P. Lin, J. Han, P. Chen, Y. Lu and Y. Liu,
  "Crystal Structure Prediction by Joint Equivariant Diffusion," *NeurIPS*
  (2023).
- **[CrystalDiT]** Yi *et al.*, "CrystalDiT: Simple Diffusion Transformers for
  Crystal Generation," *AAAI* (2026). doi:10.1609/aaai.v40i2.37121
- **[DDPM]** J. Ho, A. Jain and P. Abbeel, "Denoising Diffusion Probabilistic
  Models," *NeurIPS* (2020).
- **[iDDPM]** A. Nichol and P. Dhariwal, "Improved Denoising Diffusion
  Probabilistic Models," *ICML* (2021). — source of the cosine ᾱ schedule
- **[NCSN]** Y. Song and S. Ermon, "Generative Modeling by Estimating Gradients
  of the Data Distribution," *NeurIPS* (2019).
- **[ScoreSDE]** Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon
  and B. Poole, "Score-Based Generative Modeling through Stochastic
  Differential Equations," *ICLR* (2021). — source of the predictor–corrector
  sampler
- **[CFG]** J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance,"
  (2022). arXiv:2207.12598
- **[GatedGCN]** X. Bresson and T. Laurent, "Residual Gated Graph ConvNets,"
  (2017). arXiv:1711.07553 — the edge-gated convolution ALIGNN builds on
- **[SOAP]** A. P. Bartók, R. Kondor and G. Csányi, "On representing chemical
  environments," *Phys. Rev. B* **87**, 184115 (2013).
  doi:10.1103/PhysRevB.87.184115 — background only; not used here

**Three citation details to verify against the primary sources before this
reaches a manuscript**, because they were carried over from the design brief or
recalled rather than checked: FoldingDiff's exact smooth-L1 `β` (we use
`0.1π` and attribute the *form* to FoldingDiff with confidence, the constant
with less); the CrystalDiT DOI and venue; and the author lists for
[MatterGen] and [DiffCSP], which are given here in abbreviated form.

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

Or skip the four commands above. `task_runners/` wraps each published result
in one resumable task, with the arguments pinned, seeds handled and an sbatch
script per task:

```bash
python task_runners/run_task.py tasks         # what is available
python task_runners/run_task.py bench-jarvis  # train, generate, score, x3 seeds
python task_runners/run_task.py bench-jarvis --aggregate --latex
bash    task_runners/submit.sh  bench-jarvis  # the same, through SLURM
```

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
