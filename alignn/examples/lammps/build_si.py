"""Build a Si supercell, relax it with the ALIGNN-FF `mps` model
(so the LAMMPS run doesn't start at MBar-scale residual stress),
and write `si.data`.
"""
from ase.io import write
from ase.optimize import FIRE
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms

from alignn.ff.ff import default_path
from alignn.ff.calculators import AlignnAtomwiseCalculator

# ExpCellFilter moved to ase.filters (and is deprecated in favor of
# FrechetCellFilter) in recent ASE; fall back across versions.
try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter


a = Atoms.from_dict(get_jid_data(jid="JVASP-1002", dataset="dft_3d")["atoms"])
# JVASP-1002 is the 2-atom primitive FCC cell; a supercell of it is
# triclinic. Use the conventional (cubic) cell so the box stays orthogonal
# -> a 2x2x2 supercell is a clean 64-atom cubic system for the MD examples.
ase_atoms = a.get_conventional_atoms.make_supercell([2, 2, 2]).ase_converter()
print(f"initial: {len(ase_atoms)} atoms, V = {ase_atoms.get_volume():.2f} Å³")

ase_atoms.calc = AlignnAtomwiseCalculator(path=default_path())
ecf = CellFilter(ase_atoms)
FIRE(ecf, logfile="-").run(fmax=0.05, steps=200)
print(f"relaxed: V = {ase_atoms.get_volume():.2f} Å³, "
      f"E = {ase_atoms.get_potential_energy():.3f} eV")

write("si.data", ase_atoms, format="lammps-data", specorder=["Si"])
print(f"wrote si.data: {len(ase_atoms)} atoms")
