"""ASE reuses ALIGNN calculator results for an unchanged structure."""

import torch

import os
import tempfile

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.db import connect
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from alignn.ff.calculators import (
    AlignnAtomwiseCalculator,
    iAlignnAtomwiseCalculator,
)
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)

CONFIG = {
    "neighbor_strategy": "pure_torch",
    "cutoff": 4.0,
    "max_neighbors": 12,
    "atom_features": "cgcnn",
    "use_canonize": True,
    "three_body_cutoff": 3.5,
    "batch_size": 1,
}
MODEL_CONFIG = {
    "name": "alignn_atomwise_pure",
    "alignn_layers": 1,
    "gcn_layers": 1,
    "atom_input_features": 92,
    "edge_input_features": 16,
    "triplet_input_features": 8,
    "embedding_features": 16,
    "hidden_features": 16,
    "output_features": 1,
    "calculate_gradient": True,
    "gradwise_weight": 1.0,
    "graphwise_weight": 1.0,
    "stresswise_weight": 0.1,
    "atomwise_weight": 0.0,
    "batch_stress": True,
}


def _model():
    """Build a small untrained force-field model."""
    torch.manual_seed(0)
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**MODEL_CONFIG))
    model.eval()
    return model


def _calculator():
    """Return a calculator and a list counting its model passes."""
    model = _model()
    passes = []
    model.register_forward_hook(lambda *args: passes.append(1))
    calculator = AlignnAtomwiseCalculator(
        model=model,
        config=dict(CONFIG, model=dict(MODEL_CONFIG)),
        device=torch.device("cpu"),
    )
    return calculator, passes


class _PropertyStub(torch.nn.Module):
    """Stand in for the per-atom property model of the other calculator."""

    def __init__(self, n_atoms, n_props):
        """Store the output sizes."""
        super().__init__()
        self.n_atoms = n_atoms
        self.n_props = n_props

    def forward(self, inputs):
        """Return constant per-atom and graph-level predictions."""
        return {
            "atomwise_pred": torch.ones((self.n_atoms, 2)),
            "additional": torch.ones((1, self.n_props)),
        }


def _property_calculator(n_atoms):
    """Build the other calculator without downloading its two models."""
    calculator = iAlignnAtomwiseCalculator.__new__(iAlignnAtomwiseCalculator)
    Calculator.__init__(calculator)
    calculator.device = torch.device("cpu")
    calculator.stress_wt = 1.0
    calculator.props = ["gap"]
    calculator.implemented_properties = [
        "energy",
        "forces",
        "stress",
        "charges",
        "magmoms",
        "gap",
    ]
    calculator.ff_config = dict(CONFIG, model=dict(MODEL_CONFIG))
    calculator.prop_config = calculator.ff_config
    calculator.ff_model = _model()
    calculator.prop_model = _PropertyStub(n_atoms, 1)
    passes = []
    calculator.ff_model.register_forward_hook(
        lambda *args: passes.append(1)
    )
    return calculator, passes


def _rattled_sige():
    """Return a rattled SiGe cell: untrained models barely see pure Si."""
    atoms = bulk("Si", "diamond", a=5.5, cubic=True)
    atoms.numbers[::2] = 32
    atoms.rattle(0.05, seed=1)
    return atoms


def test_unchanged_atoms_run_the_model_once():
    """Energy, forces and stress of one structure share one model pass."""
    calculator, passes = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert len(passes) == 1
    assert atoms.get_potential_energy() == energy
    assert np.array_equal(atoms.get_forces(), forces)
    assert np.array_equal(atoms.get_stress(), stress)
    assert len(passes) == 1


def test_any_change_runs_the_model_again():
    """Position, cell and species changes each give a new calculation."""
    calculator, passes = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    atoms.positions[0, 0] += 0.05
    moved_energy = atoms.get_potential_energy()
    moved_forces = atoms.get_forces()
    assert len(passes) == 2
    assert moved_energy != energy
    assert not np.array_equal(moved_forces, forces)

    atoms.set_cell(atoms.cell.array * 1.01, scale_atoms=False)
    atoms.get_forces()
    assert len(passes) == 3

    atoms.numbers[1] = 32
    atoms.get_forces()
    assert len(passes) == 4


def test_cached_results_match_a_fresh_calculator():
    """Results after a change equal those of a calculator with no history."""
    calculator, _ = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    atoms.get_potential_energy()
    atoms.positions[3, 1] -= 0.04
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    fresh, _ = _calculator()
    other = atoms.copy()
    other.calc = fresh
    assert other.get_potential_energy() == energy
    assert np.array_equal(other.get_forces(), forces)
    assert np.array_equal(other.get_stress(), stress)


def test_one_calculator_across_structures():
    """A calculator shared by two structures gives each its own results."""
    calculator, _ = _calculator()
    first = _rattled_sige()
    second = _rattled_sige()
    second.positions[0, 0] += 0.05
    first.calc = calculator
    second.calc = calculator
    first_forces = first.get_forces()
    second_forces = second.get_forces()
    assert not np.array_equal(first_forces, second_forces)
    assert np.array_equal(first.get_forces(), first_forces)
    assert np.array_equal(second.get_forces(), second_forces)


def test_relaxation_runs_the_model_once_per_step():
    """A relaxation asking for forces and energy needs one pass per step."""
    calculator, passes = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=1e-6, steps=5)
    assert optimizer.nsteps == 5
    assert len(passes) <= optimizer.nsteps + 1


def test_cell_relaxation_runs_the_model_once_per_step():
    """A cell filter also asks for stress, without any extra model pass."""
    calculator, passes = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    optimizer = FIRE(FrechetCellFilter(atoms), logfile=None)
    optimizer.run(fmax=1e-6, steps=5)
    assert optimizer.nsteps == 5
    assert len(passes) <= optimizer.nsteps + 1


def test_returned_arrays_do_not_change_the_cache():
    """Editing a returned array leaves the stored results alone."""
    calculator, _ = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    expected_forces = forces.copy()
    expected_stress = stress.copy()
    forces[0, 0] = 999.0
    stress[0] = 999.0
    assert np.array_equal(atoms.get_forces(), expected_forces)
    assert np.array_equal(atoms.get_stress(), expected_stress)


def test_failed_calculation_leaves_no_old_results():
    """A failed direct call must not pair old results with the new atoms."""
    calculator, _ = _calculator()
    atoms = _rattled_sige()
    calculator.calculate(atoms)

    def fail(*args):
        raise RuntimeError("model failed")

    hook = calculator.model.register_forward_pre_hook(fail)
    other = atoms.copy()
    other.positions[0, 0] += 0.05
    try:
        calculator.calculate(other)
    except RuntimeError:
        pass
    hook.remove()
    assert calculator.results == {}

    other.calc = calculator
    fresh, _ = _calculator()
    expected = other.copy()
    expected.calc = fresh
    assert np.array_equal(other.get_forces(), expected.get_forces())


def test_results_survive_an_ase_database():
    """Forces and stress keep their shape and values through ase.db."""
    calculator, _ = _calculator()
    atoms = _rattled_sige()
    atoms.calc = calculator
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert atoms.calc.results["forces"].dtype == np.float64
    assert atoms.calc.results["stress"].dtype == np.float64
    with tempfile.TemporaryDirectory() as folder:
        database = connect(os.path.join(folder, "results.db"))
        database.write(atoms)
        row = database.get(1)
    assert np.array_equal(row.forces, forces)
    assert np.array_equal(row.stress, stress)


def test_property_calculator_uses_the_cache():
    """The calculator with per-atom properties also caches, and safely."""
    atoms = _rattled_sige()
    calculator, passes = _property_calculator(len(atoms))
    atoms.calc = calculator
    atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.get_stress()
    charges = atoms.calc.get_charges(atoms)
    assert len(passes) == 1
    assert forces.dtype == np.float64

    charges[0] = 999.0
    assert atoms.calc.get_charges(atoms)[0] == 1.0

    atoms.positions[0, 0] += 0.05
    atoms.get_forces()
    assert len(passes) == 2


def test_property_calculator_failed_calculation_leaves_no_old_results():
    """The other calculator also clears old results when a call fails."""
    atoms = _rattled_sige()
    calculator, _ = _property_calculator(len(atoms))
    calculator.calculate(atoms)

    def fail(*args):
        raise RuntimeError("model failed")

    calculator.ff_model.register_forward_pre_hook(fail)
    try:
        calculator.calculate(atoms)
    except RuntimeError:
        pass
    assert calculator.results == {}
