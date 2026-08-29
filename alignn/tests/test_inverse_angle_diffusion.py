"""Tests for the angular diffusion channel and the smooth line-graph topology.

These cover section 10 of the design brief:

* the angular corruption respects angular periodicity,
* the cutoff envelope and its first two derivatives vanish at ``r_c``,
* a pair contribution goes to zero continuously as ``r -> r_c``,
* a triplet contribution goes to zero when *either* of its bonds does,
* moving one atom across the cutoff does not make the model output jump
  merely because a sparse edge was inserted or deleted,
* periodic coordinate handling is still correct,
* the default configuration reproduces the original model exactly,
* the angle-enabled configuration trains and samples.
"""

import math

import pytest
import torch

from alignn.inverse.ablations import ABLATIONS, ablation_config
from alignn.inverse.angles import (
    CutoffPolynomial,
    angle_denoising_loss,
    angular_denoising_target,
    bond_angle,
    triplet_relevance,
    wrap_angle,
)
from alignn.inverse.data import Normalizer
from alignn.inverse.denoiser import ALIGNNCSPDenoiser, dense_pair_index
from alignn.inverse.diffusion import DiffusionSchedule, wrap_frac
from alignn.inverse.model import ALIGNNCSP
from alignn.inverse.sample import sample
from alignn.inverse.train_csp import diffusion_loss

CUTOFF = 5.0


@pytest.fixture(autouse=True)
def _float32_default():
    """Pin the default dtype for this module.

    ``test_force_reduction`` sets the global default to float64 at import
    time and does not restore it, and the diffusion denoiser's timestep
    embedding is float32 by construction, so without this the results of
    these tests would depend on collection order.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


SMALL = dict(
    hidden_features=32,
    embedding_features=16,
    alignn_layers=2,
    gcn_layers=1,
    rbf_bins=16,
    triplet_bins=8,
    score_channels=4,
    num_steps=50,
)


def _batch(seed=0, natoms=(3, 2), cell=4.0):
    """A tiny two-crystal batch."""
    torch.manual_seed(seed)
    n = torch.tensor(natoms)
    total = int(n.sum())
    return {
        "frac": torch.rand(total, 3),
        "lattice": torch.eye(3).repeat(len(natoms), 1, 1) * cell,
        "atomic_numbers": torch.randint(1, 60, (total,)),
        "natoms": n,
        "node_graph_id": torch.repeat_interleave(torch.arange(len(natoms)), n),
        "prop": torch.zeros(len(natoms)),
    }


def _forward(model, batch, t=25):
    return model(
        frac=batch["frac"],
        lattice=batch["lattice"],
        lattice_vec6=torch.zeros(len(batch["natoms"]), 6),
        atomic_numbers=batch["atomic_numbers"],
        natoms=batch["natoms"],
        node_graph_id=batch["node_graph_id"],
        t=torch.full((len(batch["natoms"]),), t, dtype=torch.long),
    )


# ── angular periodicity ──────────────────────────────────────────────────
def test_wrap_angle_is_periodic_and_in_range():
    x = torch.linspace(-20.0, 20.0, 401)
    w = wrap_angle(x)
    assert torch.all(w >= -math.pi) and torch.all(w < math.pi)
    # Adding a full turn changes nothing.
    for k in (-2, -1, 1, 2):
        assert torch.allclose(wrap_angle(x + k * 2 * math.pi), w, atol=1e-5)
    # And it is the identity where it should be.
    inner = torch.linspace(-3.0, 3.0, 61)
    assert torch.allclose(wrap_angle(inner), inner, atol=1e-6)


def test_angle_loss_is_wrapped():
    """A residual of 2*pi is no error at all."""
    pred = torch.tensor([1.0, 2.0])
    zero = angle_denoising_loss(pred, pred)
    wrapped = angle_denoising_loss(pred + 2 * math.pi, pred)
    assert float(zero) == pytest.approx(0.0, abs=1e-6)
    assert float(wrapped) == pytest.approx(0.0, abs=1e-6)


def test_angular_target_vanishes_without_corruption():
    """theta_t == theta_0 when the noised structure *is* the clean one."""
    b = _batch(seed=3)
    model = ALIGNNCSPDenoiser(**SMALL, **ablation_config("A3"))
    out = _forward(model, b)
    aux = out["angle"]
    target = angular_denoising_target(
        aux["theta_t"],
        b["frac"],
        b["lattice"],
        aux["src"],
        aux["dst"],
        aux["edge_graph_id"],
        aux["image"],
        aux["lg_src"],
        aux["lg_dst"],
    )
    assert target.numel() > 0
    assert float(target.abs().max()) < 1e-4


def test_angular_target_matches_a_hand_computed_rotation():
    """Bending one bond by a known angle shows up in the target."""
    # Two atoms placed so the triplet at atom 0 is a right angle, then the
    # third atom is swung to 60 degrees.
    lattice = torch.eye(3).unsqueeze(0) * 12.0
    clean = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]])
    moved = clean.clone()
    moved[2] = torch.tensor([0.125, 0.125 * math.sqrt(3.0), 0.0])
    src, dst, egid = dense_pair_index(torch.tensor([3]))
    model = ALIGNNCSPDenoiser(**SMALL, **ablation_config("A3"))
    b = {
        "frac": moved,
        "lattice": lattice,
        "atomic_numbers": torch.tensor([6, 6, 6]),
        "natoms": torch.tensor([3]),
        "node_graph_id": torch.zeros(3, dtype=torch.long),
    }
    aux = _forward(model, b)["angle"]
    target = angular_denoising_target(
        aux["theta_t"],
        clean,
        lattice,
        aux["src"],
        aux["dst"],
        aux["edge_graph_id"],
        aux["image"],
        aux["lg_src"],
        aux["lg_dst"],
    )
    # The 1-0-2 triplet went from 90 to 60 degrees, i.e. -30 degrees.
    is_triplet_at_0 = (
        (aux["dst"][aux["lg_src"]] == 0)
        & (aux["src"][aux["lg_src"]] == 1)
        & (aux["dst"][aux["lg_dst"]] == 2)
    )
    assert bool(is_triplet_at_0.any())
    got = math.degrees(float(target[is_triplet_at_0][0]))
    assert got == pytest.approx(-30.0, abs=0.5)


# ── smooth cutoff ────────────────────────────────────────────────────────
def test_envelope_and_two_derivatives_vanish_at_cutoff():
    env = CutoffPolynomial(cutoff=CUTOFF, coeff=5.0)
    r = torch.tensor([CUTOFF], dtype=torch.float64, requires_grad=True)
    u = env(r)
    (du,) = torch.autograd.grad(u.sum(), r, create_graph=True)
    (d2u,) = torch.autograd.grad(du.sum(), r, create_graph=True)
    assert float(u) == pytest.approx(0.0, abs=1e-12)
    assert float(du) == pytest.approx(0.0, abs=1e-10)
    assert float(d2u) == pytest.approx(0.0, abs=1e-8)
    # Unit at zero separation, monotone decreasing, never negative.
    grid = torch.linspace(0.0, CUTOFF, 501, dtype=torch.float64)
    vals = env(grid)
    assert float(env(torch.zeros(1, dtype=torch.float64))) == pytest.approx(
        1.0
    )
    assert torch.all(vals >= 0.0)
    assert torch.all(vals[1:] <= vals[:-1] + 1e-12)


def test_pair_contribution_goes_to_zero_continuously():
    env = CutoffPolynomial(cutoff=CUTOFF, coeff=5.0)
    eps = 1e-6
    inside = float(env(torch.tensor([CUTOFF - eps], dtype=torch.float64)))
    outside = float(env(torch.tensor([CUTOFF + eps], dtype=torch.float64)))
    assert inside == pytest.approx(0.0, abs=1e-14)
    assert outside == 0.0
    # No step anywhere across the boundary. Evaluated in double precision:
    # the polynomial is written as a sum of terms of order 20 that cancel to
    # ~1e-5 near r_c, so in float32 the *value* carries ~1e-6 of rounding
    # noise. That is harmless — it multiplies messages that are already being
    # driven to zero — but it would swamp a test of the exact property.
    grid = torch.linspace(
        CUTOFF - 0.05, CUTOFF + 0.05, 2001, dtype=torch.float64
    )
    vals = env(grid)
    step = float(vals.diff().abs().max())
    assert step < 5e-7
    # And the whole window sits within rounding distance of zero: there is
    # no cliff for a message to fall off.
    assert float(vals.max()) < 1e-4


def test_triplet_weight_vanishes_when_either_bond_reaches_the_cutoff():
    env = CutoffPolynomial(cutoff=CUTOFF, coeff=5.0)
    dist = torch.tensor([1.0, CUTOFF - 1e-7, 2.0])
    s = env(dist)
    lg_src = torch.tensor([0, 0, 1, 2])
    lg_dst = torch.tensor([2, 1, 2, 0])
    w = triplet_relevance(s, lg_src, lg_dst)
    # Any triplet touching edge 1 (at the cutoff) is off.
    assert float(w[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(w[2]) == pytest.approx(0.0, abs=1e-12)
    # The one made of two short bonds is not.
    assert float(w[0]) > 0.1


# ── no jump when an edge enters or leaves the sparse graph ───────────────
def test_no_finite_jump_when_a_triplet_crosses_the_cutoff():
    """Sweep one atom through the radius and check the output is smooth.

    The line graph is rebuilt from scratch at every position, so triplets are
    genuinely inserted and deleted during this sweep; the envelope is what
    makes that invisible.
    """
    torch.manual_seed(0)
    model = ALIGNNCSPDenoiser(
        **SMALL, **ablation_config("A3"), radius_cutoff=CUTOFF
    ).eval()
    cell = 20.0
    lattice = torch.eye(3).unsqueeze(0) * cell
    base = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def run(x):
        frac = base.clone()
        frac[2, 0] = x / cell
        out = model(
            frac=wrap_frac(frac),
            lattice=lattice,
            lattice_vec6=torch.zeros(1, 6),
            atomic_numbers=torch.tensor([6, 6, 6]),
            natoms=torch.tensor([3]),
            node_graph_id=torch.zeros(3, dtype=torch.long),
            t=torch.tensor([25]),
        )
        return out

    # A dense sweep straight through the cutoff radius.
    xs = torch.linspace(CUTOFF - 0.3, CUTOFF + 0.3, 121)
    scores, n_triplets = [], []
    with torch.no_grad():
        for x in xs:
            out = run(float(x))
            scores.append(out["eps_frac"].clone())
            n_triplets.append(int(out["angle"]["eps"].shape[0]))
    # The triplet count really does change across the sweep.
    assert len(set(n_triplets)) > 1
    steps = torch.stack(
        [(a - b).abs().max() for a, b in zip(scores[1:], scores[:-1])]
    )
    # Every consecutive step is small; a topology jump would show up as one
    # step far larger than its neighbours.
    assert float(steps.max()) < 20.0 * float(steps.median()) + 1e-6


def test_gated_message_ignores_a_zero_weight_edge():
    """An edge weighted to zero leaves the aggregation exactly unchanged."""
    from alignn.inverse.layers import WeightedEdgeGatedGraphConv

    torch.manual_seed(0)
    conv = WeightedEdgeGatedGraphConv(8, 8).eval()
    src = torch.tensor([0, 1, 2, 0])
    dst = torch.tensor([1, 2, 0, 2])
    x = torch.randn(3, 8)
    y = torch.randn(4, 8)
    w = torch.tensor([1.0, 1.0, 1.0, 0.0])
    with torch.no_grad():
        gated, _ = conv.forward_tensors(src, dst, 3, x, y, w)
        # Same graph with that edge physically removed.
        keep = torch.tensor([0, 1, 2])
        dropped, _ = conv.forward_tensors(src[keep], dst[keep], 3, x, y[keep])
    assert torch.allclose(gated, dropped, atol=1e-6)


# ── periodicity of the model itself ──────────────────────────────────────
@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_output_is_invariant_to_lattice_translations(name):
    """Adding whole cells to the coordinates must change nothing."""
    b = _batch(seed=1)
    model = ALIGNNCSPDenoiser(**SMALL, **ablation_config(name)).eval()
    with torch.no_grad():
        a = _forward(model, b)
        shifted = dict(b)
        shifted["frac"] = b["frac"] + torch.tensor([1.0, -2.0, 3.0])
        c = _forward(model, shifted)
    assert torch.allclose(a["eps_frac"], c["eps_frac"], atol=1e-5)
    assert torch.allclose(a["eps_lattice"], c["eps_lattice"], atol=1e-5)


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_coordinate_score_is_invariant_to_a_global_shift(name):
    """A rigid translation of the crystal is not a change of structure."""
    b = _batch(seed=2)
    model = ALIGNNCSPDenoiser(**SMALL, **ablation_config(name)).eval()
    with torch.no_grad():
        a = _forward(model, b)
        shifted = dict(b)
        shifted["frac"] = wrap_frac(b["frac"] + 0.137)
        c = _forward(model, shifted)
    assert torch.allclose(a["eps_frac"], c["eps_frac"], atol=1e-5)


# ── the baseline is untouched ────────────────────────────────────────────
def test_default_config_builds_the_original_model():
    """No new parameters, and no new outputs, unless asked for."""
    model = ALIGNNCSPDenoiser(**SMALL)
    keys = set(model.state_dict())
    assert not any(k.startswith("angle_head") for k in keys)
    assert not any("envelope" in k for k in keys)
    assert model.topology == "knn"
    assert model.angle_diffusion is False
    out = _forward(model, _batch())
    assert set(out) == {"eps_frac", "eps_lattice"}


def test_angle_head_does_not_perturb_the_structural_pathway():
    """A1 must equal A0 on eps_frac / eps_lattice, given the same weights.

    This is what makes the angular objective an addition rather than a
    change: with the topology held at the baseline's, switching the angle
    head on adds an output without moving the existing ones.
    """
    torch.manual_seed(7)
    base = ALIGNNCSPDenoiser(**SMALL, **ablation_config("A0")).eval()
    angled = ALIGNNCSPDenoiser(**SMALL, **ablation_config("A1")).eval()
    missing, unexpected = angled.load_state_dict(
        base.state_dict(), strict=False
    )
    assert not unexpected
    assert all(k.startswith("angle_head") for k in missing)
    b = _batch(seed=5)
    with torch.no_grad():
        a, c = _forward(base, b), _forward(angled, b)
    assert torch.allclose(a["eps_frac"], c["eps_frac"], atol=1e-6)
    assert torch.allclose(a["eps_lattice"], c["eps_lattice"], atol=1e-6)
    assert "angle" in c


def test_a4_cuts_the_angular_coupling():
    """A4's structural output must not depend on the angular features.

    Perturbing only the angle-embedding weights moves A3's coordinate score
    and leaves A4's alone.
    """
    b = _batch(seed=6)
    results = {}
    for name in ("A1", "A3", "A4"):
        torch.manual_seed(11)  # identical weights in every arm
        model = ALIGNNCSPDenoiser(**SMALL, **ablation_config(name)).eval()
        # The coordinate head is zero-initialised by design, which would make
        # every arm read zero; give it signal first.
        torch.nn.init.normal_(model.score_combine.weight, std=0.5)
        with torch.no_grad():
            before = _forward(model, b)["eps_frac"].clone()
            torch.manual_seed(3)
            for pname, p in model.named_parameters():
                if pname.startswith("angle_embedding"):
                    p.add_(torch.randn_like(p) * 0.5)
            after = _forward(model, b)["eps_frac"]
            results[name] = float(
                (after - before).abs().max() / before.abs().max()
            )
    assert results["A4"] == pytest.approx(0.0, abs=1e-9)
    assert results["A3"] > 1e-3
    assert results["A1"] > 1e-3


# ── end-to-end ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_train_step_and_sampling_run(name):
    """Forward, loss, backward and reverse sampling for every ablation."""
    torch.manual_seed(0)
    model = ALIGNNCSP(
        denoiser_config={**SMALL, **ablation_config(name)},
        conditioner_spec={"composition": {"type": "composition"}},
    )
    schedule = DiffusionSchedule(num_steps=SMALL["num_steps"])
    normalizer = Normalizer(
        lattice_mean=torch.zeros(6),
        lattice_std=torch.ones(6),
        prop_mean=0.0,
        prop_std=1.0,
    )
    b = _batch(seed=4)
    losses = diffusion_loss(
        model,
        schedule,
        normalizer,
        b,
        {"composition": 0.1},
        lattice_weight=1.0,
        frac_weight=10.0,
        angle_weight=1.0,
    )
    losses["loss"].backward()
    grads = [
        p.grad.abs().sum() for p in model.parameters() if p.grad is not None
    ]
    assert float(sum(grads)) > 0.0
    if ABLATIONS[name]["angle_diffusion"]:
        assert float(losses["loss_angle"]) > 0.0
    else:
        assert float(losses["loss_angle"]) == 0.0

    out = sample(
        model,
        schedule,
        normalizer,
        b,
        guidance=1.0,
        n_corrector=0,
        device=torch.device("cpu"),
    )
    assert out["frac"].shape == b["frac"].shape
    assert torch.isfinite(out["frac"]).all()
    assert torch.isfinite(out["lattice"]).all()


def test_angle_loss_gradient_reaches_the_shared_backbone():
    """The angular objective must train the trunk, not just its own head."""
    torch.manual_seed(0)
    model = ALIGNNCSPDenoiser(**SMALL, **ablation_config("A3"))
    # The angle head's last layer is zero-initialised so that training starts
    # from a silent prediction; that also zeroes the gradient through it, so
    # this test looks at the model one step into training.
    torch.nn.init.normal_(model.angle_head[-1].weight, std=0.5)
    b = _batch(seed=8)
    out = _forward(model, b)
    aux = out["angle"]
    target = angular_denoising_target(
        aux["theta_t"],
        b["frac"] + 0.05,
        b["lattice"],
        aux["src"],
        aux["dst"],
        aux["edge_graph_id"],
        aux["image"],
        aux["lg_src"],
        aux["lg_dst"],
    )
    angle_denoising_loss(aux["eps"], target, aux["weight"]).backward()
    touched = {
        name
        for name, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    }
    assert any(n.startswith("alignn_layers") for n in touched)
    assert any(n.startswith("angle_embedding") for n in touched)
    assert any(n.startswith("edge_embedding") for n in touched)


def test_bond_angle_matches_a_known_geometry():
    r_ij = torch.tensor([[1.0, 0.0, 0.0]])
    r_jk = torch.tensor([[1.0, 0.0, 0.0]])
    # i -> j -> k collinear and continuing forward is a straight 180 degrees.
    # The tolerance is the deliberate clamp inside bond_angle, which keeps
    # acos differentiable at the poles at the cost of ~0.03 degrees there.
    assert math.degrees(float(bond_angle(r_ij, r_jk))) == pytest.approx(
        180.0, abs=0.05
    )
    r_jk = torch.tensor([[0.0, 1.0, 0.0]])
    assert math.degrees(float(bond_angle(r_ij, r_jk))) == pytest.approx(
        90.0, abs=1e-2
    )
