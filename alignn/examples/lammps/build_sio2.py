"""Build an α-quartz SiO2 supercell, relax it with the ALIGNN-FF `mps`
model, and write `sio2.data`.

JVASP-41 is α-quartz. A 3x3x2 supercell gives ~162 atoms.
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


a = Atoms.from_dict(get_jid_data(jid="JVASP-41", dataset="dft_3d")["atoms"])
ase_atoms = a.make_supercell([3, 3, 2]).ase_converter()
print(f"initial: {len(ase_atoms)} atoms, V = {ase_atoms.get_volume():.2f} Å³")

ase_atoms.calc = AlignnAtomwiseCalculator(path=default_path())
ecf = CellFilter(ase_atoms)
FIRE(ecf, logfile="-").run(fmax=0.1, steps=200)
print(f"relaxed: V = {ase_atoms.get_volume():.2f} Å³, "
      f"E = {ase_atoms.get_potential_energy():.3f} eV")

# specorder fixes type-id mapping: type 1 = Si, type 2 = O.
write("sio2.data", ase_atoms, format="lammps-data", specorder=["Si", "O"])
print(f"wrote sio2.data: {len(ase_atoms)} atoms")
