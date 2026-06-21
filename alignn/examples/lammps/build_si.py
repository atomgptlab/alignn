"""Build a Si supercell, relax it with the ALIGNN-FF `mps` model
(so the LAMMPS run doesn't start at MBar-scale residual stress),
and write `si.data`.
"""
from ase.io import write
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms

from alignn.ff.ff import default_path
from alignn.ff.calculators import AlignnAtomwiseCalculator


a = Atoms.from_dict(get_jid_data(jid="JVASP-1002", dataset="dft_3d")["atoms"])
ase_atoms = a.make_supercell([3, 3, 3]).ase_converter()
print(f"initial: {len(ase_atoms)} atoms, V = {ase_atoms.get_volume():.2f} Å³")

ase_atoms.calc = AlignnAtomwiseCalculator(path=default_path())
ecf = ExpCellFilter(ase_atoms)
FIRE(ecf, logfile="-").run(fmax=0.05, steps=200)
print(f"relaxed: V = {ase_atoms.get_volume():.2f} Å³, "
      f"E = {ase_atoms.get_potential_energy():.3f} eV")

write("si.data", ase_atoms, format="lammps-data", specorder=["Si"])
print(f"wrote si.data: {len(ase_atoms)} atoms")
