"""High-level generation API for ALIGNN-CSP.

Mirrors how :class:`~alignn.ff.unified_calculator.AlignnUnifiedCalculator` is
used: build the object once, which loads the diffusion model and the force
field a single time, then call it repeatedly.

    from alignn.inverse.generate import ALIGNNGenerator

    gen = ALIGNNGenerator()                 # DEFAULT_MODEL
    result = gen.generate("NbN", prop=15.0)
    print(result.atoms)

Generation cost is dominated by the sequential denoising steps, not by how
many structures are drawn: on one GPU, 32 candidates take the same wall time
as one. ``num_candidates`` is therefore close to free, and it is the strongest
lever on how good the returned structure is, so it defaults to 8 rather than 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch

from alignn.inverse.data import make_generation_batch
from alignn.inverse.diffusion import DiffusionSchedule
from alignn.inverse.sample import load_model, sample, to_jarvis_atoms

CompositionLike = Union[str, Dict[str, int], Sequence[Union[int, str]]]

#: Model used when neither ``model`` nor ``checkpoint`` is given. Trained on
#: the larger of the two superconductor benchmarks (Alexandria DS-A/B, 6603
#: crystals) and fine-tuned from the dft_3d base, so it is the most broadly
#: applicable of the released set.
DEFAULT_MODEL = "csp_supercon_alex"


@dataclass
class GeneratedStructure:
    """One generated crystal and what is known about it."""

    atoms: object  # jarvis Atoms
    energy_per_atom: Optional[float] = None
    relaxed: bool = False
    converged: bool = False
    formula: str = ""
    rank: int = 0
    error: Optional[str] = None

    def __repr__(self) -> str:
        e = (
            "n/a"
            if self.energy_per_atom is None
            else f"{self.energy_per_atom:.4f} eV/atom"
        )
        return (
            f"GeneratedStructure({self.formula}, {len(self.atoms.elements)} "
            f"atoms, E={e}, relaxed={self.relaxed})"
        )

    def to_poscar(self) -> str:
        """POSCAR text for this structure."""
        from jarvis.io.vasp.inputs import Poscar

        return Poscar(self.atoms).to_string()


@dataclass
class GenerationResult:
    """Result of one :meth:`ALIGNNGenerator.generate` call.

    Behaves like the best structure for convenience (``result.atoms``) while
    keeping every candidate available in ``result.candidates``.
    """

    candidates: List[GeneratedStructure] = field(default_factory=list)

    @property
    def best(self) -> GeneratedStructure:
        return self.candidates[0]

    @property
    def atoms(self):
        return self.best.atoms

    @property
    def energy_per_atom(self) -> Optional[float]:
        return self.best.energy_per_atom

    def to_poscar(self) -> str:
        return self.best.to_poscar()

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __getitem__(self, i):
        return self.candidates[i]

    def __repr__(self) -> str:
        return (
            f"GenerationResult(best={self.best!r}, "
            f"{len(self.candidates)} candidates)"
        )


def parse_composition(comp: CompositionLike) -> List[int]:
    """Normalise a composition to an explicit list of atomic numbers.

    Accepts a formula string (``"Fe2O3"``), a counts dict
    (``{"Fe": 2, "O": 3}``), or an explicit sequence of symbols or atomic
    numbers (``["Fe", "Fe", "O", "O", "O"]``, ``[26, 26, 8, 8, 8]``). The
    atom count is what the model conditions on, so a formula is expanded to
    one atom per unit.
    """
    from jarvis.core.composition import Composition
    from jarvis.core.specie import Specie

    if isinstance(comp, str):
        counts = Composition.from_string(comp).to_dict()
    elif isinstance(comp, dict):
        counts = comp
    else:
        out = []
        for x in comp:
            out.append(int(x) if not isinstance(x, str) else Specie(x).Z)
        if not out:
            raise ValueError("composition is empty")
        return out

    numbers: List[int] = []
    for symbol, n in counts.items():
        n_int = int(round(float(n)))
        if n_int < 0:
            raise ValueError(f"negative count for {symbol}")
        numbers.extend([Specie(symbol).Z] * n_int)
    if not numbers:
        raise ValueError(f"composition {comp!r} contains no atoms")
    return numbers


class ALIGNNGenerator:
    """Generate crystal structures from a composition and a target property.

    Args:
        model: name of a released model in the ALIGNN 2.0 registry, e.g.
            ``"csp_supercon_jarvis"``; downloaded and cached on first use.
            Defaults to :data:`DEFAULT_MODEL` when no ``checkpoint`` is given.
            See ``alignn.pretrained.list_alignn2_models("generative")``.
        checkpoint: path to a local checkpoint, as an alternative to ``model``.
        relax: refine candidates with ALIGNN-FF over cell and positions.
            Leave on: the references a generator is judged against sit at
            local minima of the energy surface, and relaxation is what moves
            a sample onto them.
        rank: order candidates by ALIGNN-FF energy per atom, lowest first.
        relax_top: relax only this many of the candidates. A single-point
            energy costs ~0.3 s and a relaxation ~100x that, so all candidates
            are screened cheaply first and the relaxation budget is spent only
            on the best. Raise it to trade latency for quality.
        num_steps: denoising steps; defaults to the value the checkpoint was
            trained with. Fewer is faster and lower fidelity.
        guidance: classifier-free guidance scale. 1.0 disables guidance.
        ff_path: force-field directory; defaults to ALIGNN-FF's own default.
        relax_workers: leave ``None`` to relax in-process, reusing one
            force-field instance. Set an integer only for large batches: a
            process pool reloads the force field in every worker, which costs
            more than it saves for a handful of candidates, and it requires
            the caller to be under an ``if __name__ == "__main__":`` guard
            because the pool uses the spawn start method.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        checkpoint: Optional[str] = None,
        relax: bool = True,
        rank: bool = True,
        num_steps: Optional[int] = None,
        guidance: float = 2.0,
        device: Optional[str] = None,
        use_ema: bool = True,
        ff_path: Optional[str] = None,
        relax_top: int = 1,
        relax_fmax: float = 0.05,
        relax_steps: int = 100,
        relax_workers: Optional[int] = None,
        min_distance: float = 0.7,
    ):
        if model is not None and checkpoint is not None:
            raise ValueError("give at most one of model= or checkpoint=")
        if model is None and checkpoint is None:
            model = DEFAULT_MODEL
        if model is not None:
            checkpoint = resolve_model(model)
        self.checkpoint = checkpoint
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, schedule, self.normalizer, self.config = load_model(
            checkpoint, self.device, use_ema=use_ema
        )
        self.trained_steps = int(self.config["num_steps"])
        self.num_steps = int(num_steps or self.trained_steps)
        self.schedule = (
            schedule
            if self.num_steps == self.trained_steps
            else DiffusionSchedule(
                num_steps=self.num_steps,
                sigma_min=self.config["sigma_min"],
                sigma_max=self.config["sigma_max"],
            ).to(self.device)
        )
        self.model.denoiser.num_steps = self.num_steps

        self.guidance = guidance
        self.relax = relax
        self.rank = rank
        self.ff_path = ff_path
        self.relax_top = max(1, int(relax_top))
        self.relax_fmax = relax_fmax
        self.relax_steps = relax_steps
        self.relax_workers = relax_workers
        self.min_distance = min_distance
        self._relaxer = None

    def _get_relaxer(self):
        """Force field, loaded on first use and reused thereafter."""
        if self._relaxer is None:
            from alignn.inverse.relax_rank import AlignnFFRelaxer

            self._relaxer = AlignnFFRelaxer(
                model_path=self.ff_path,
                relax_cell=True,
                fmax=self.relax_fmax,
                steps=self.relax_steps,
            )
        return self._relaxer

    @property
    def modalities(self) -> List[str]:
        """Conditioning the loaded checkpoint understands."""
        return self.model.modalities

    def generate(
        self,
        composition: CompositionLike,
        prop: Optional[float] = None,
        num_candidates: int = 8,
        formula_units: int = 1,
        active_modalities: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **conditioning,
    ) -> GenerationResult:
        """Generate structures for one composition.

        Args:
            composition: formula, counts dict, or explicit atom list.
            prop: target value for the checkpoint's scalar property (the Tc
                the model was trained on, for the supercon checkpoints).
                ``None`` leaves that modality unconditioned.
            num_candidates: how many to draw. Nearly free, since cost scales
                with denoising steps rather than batch size.
            formula_units: repeat the composition this many times, to ask for
                a larger cell.
            active_modalities: restrict guidance to a subset of what the
                checkpoint knows, e.g. ``["composition"]``.
            **conditioning: extra modality values by name, for checkpoints
                trained with them (``xrd=...``, ``stem=...``).
        """
        if seed is not None:
            torch.manual_seed(seed)
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if formula_units < 1:
            raise ValueError("formula_units must be >= 1")

        numbers = parse_composition(composition) * int(formula_units)
        formula = _formula_of(numbers)

        item = {
            "material_id": formula,
            "formula": formula,
            "atomic_numbers": numbers,
            "prop": 0.0 if prop is None else float(prop),
            "target_poscar": "",
        }
        batch = make_generation_batch([item] * num_candidates, self.device)
        for name, value in conditioning.items():
            batch[name] = _as_batched_tensor(
                value, num_candidates, self.device
            )

        active = active_modalities
        if active is None and prop is None:
            # Unspecified property: guide on everything except that.
            active = [m for m in self.modalities if m != "prop"]

        out = sample(
            self.model,
            self.schedule,
            self.normalizer,
            batch,
            guidance=self.guidance,
            active_modalities=active,
        )
        atoms = to_jarvis_atoms(
            out["frac"],
            out["lattice"],
            batch["atomic_numbers"],
            batch["natoms"],
        )

        if not (self.relax or self.rank):
            return GenerationResult(
                [
                    GeneratedStructure(atoms=a, formula=formula, rank=i)
                    for i, a in enumerate(atoms)
                ]
            )

        if self.relax_workers and self.relax_workers > 1:
            from alignn.inverse.relax_rank import parallel_rank

            ranked = parallel_rank(
                [atoms],
                model_path=self.ff_path,
                relax=self.relax,
                relax_cell=True,
                fmax=self.relax_fmax,
                steps=self.relax_steps,
                min_distance=self.min_distance,
                n_workers=self.relax_workers,
                progress_every=0,
            )[0]
        else:
            from alignn.inverse.relax_rank import rank_candidates

            relaxer = self._get_relaxer()
            if not self.relax:
                ranked = rank_candidates(
                    atoms,
                    relaxer,
                    relax=False,
                    min_distance=self.min_distance,
                )
            else:
                # Cheap screen over everything, then relax only the best few.
                screened = rank_candidates(
                    atoms,
                    relaxer,
                    relax=False,
                    min_distance=self.min_distance,
                )
                keep = [r.atoms for r in screened[: self.relax_top]]
                ranked = rank_candidates(
                    keep,
                    relaxer,
                    relax=True,
                    min_distance=self.min_distance,
                )
                # Unrelaxed remainder stays available, after the relaxed ones.
                ranked = ranked + list(screened[self.relax_top :])
        return GenerationResult(
            [
                GeneratedStructure(
                    atoms=r.atoms,
                    energy_per_atom=(
                        None
                        if r.energy_per_atom == float("inf")
                        else r.energy_per_atom
                    ),
                    relaxed=(
                        self.relax and r.error is None and i < self.relax_top
                    ),
                    converged=r.converged,
                    formula=formula,
                    rank=i,
                    error=r.error,
                )
                for i, r in enumerate(ranked)
            ]
        )

    def generate_many(
        self, compositions: Sequence[CompositionLike], **kwargs
    ) -> List[GenerationResult]:
        """Generate for several compositions, one result each."""
        return [self.generate(c, **kwargs) for c in compositions]

    def __repr__(self) -> str:
        return (
            f"ALIGNNGenerator(steps={self.num_steps}, "
            f"guidance={self.guidance}, relax={self.relax}, "
            f"modalities={self.modalities}, device={self.device.type})"
        )


def resolve_model(name: str) -> str:
    """Path to a released model's checkpoint, downloading it on first use.

    Names come from the generative section of the ALIGNN 2.0 registry in
    :mod:`alignn.pretrained`.
    """
    from alignn.pretrained import ALIGNN2_MODELS, get_alignn2_model

    meta = ALIGNN2_MODELS.get(name)
    if meta is None or meta.get("category") != "generative":
        available = sorted(
            k
            for k, v in ALIGNN2_MODELS.items()
            if v.get("category") == "generative"
        )
        raise KeyError(
            f"unknown generative model {name!r}; available: {available}"
        )
    return get_alignn2_model(name)["best_model.pt"]


def _formula_of(numbers: Sequence[int]) -> str:
    from collections import Counter

    from jarvis.core.specie import atomic_numbers_to_symbols

    symbols = list(atomic_numbers_to_symbols([int(z) for z in numbers]))
    counts = Counter(symbols)
    return "".join(
        f"{s}{n if n > 1 else ''}" for s, n in sorted(counts.items())
    )


def _as_batched_tensor(value, n: int, device) -> torch.Tensor:
    """Broadcast one conditioning value across the candidate batch."""
    t = torch.as_tensor(value, dtype=torch.float32, device=device)
    if t.dim() == 0:
        return t.view(1).expand(n).contiguous()
    if t.shape[0] != n:
        return t.unsqueeze(0).expand(n, *t.shape).contiguous()
    return t


def generate(
    composition: CompositionLike,
    model: Optional[str] = None,
    checkpoint: Optional[str] = None,
    prop: Optional[float] = None,
    **kwargs,
) -> GenerationResult:
    """One-shot convenience wrapper.

    Loads the model on every call, so use :class:`ALIGNNGenerator` directly
    for more than a single structure. With neither ``model`` nor
    ``checkpoint``, uses :data:`DEFAULT_MODEL`.
    """
    gen_kwargs = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k
        in {
            "relax",
            "rank",
            "num_steps",
            "guidance",
            "device",
            "use_ema",
            "ff_path",
            "relax_fmax",
            "relax_steps",
            "relax_workers",
            "min_distance",
        }
    }
    return ALIGNNGenerator(
        model=model, checkpoint=checkpoint, **gen_kwargs
    ).generate(composition, prop=prop, **kwargs)
