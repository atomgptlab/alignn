# ALIGNN-FF ASE Calculator

The [ASE calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)
interface lets you drop ALIGNN-FF into any ASE-based workflow — optimization, phonons,
MD, equation-of-state scans, interfaces, etc.

![ALIGNN-FF animation](https://github.com/atomgptlab/alignn/blob/develop/alignn/tex/animation.gif?raw=true)

## Minimal usage

```python
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

calc = AlignnAtomwiseCalculator(path=default_path())
ase_atoms.calc = calc
energy = ase_atoms.get_potential_energy()
forces = ase_atoms.get_forces()
```

## Full example — relax + EV curve on Silicon

```python
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import ase_to_atoms
from ase.constraints import ExpCellFilter
from ase.optimize.fire import FIRE
import numpy as np
import matplotlib.pyplot as plt

poscar = """Si2
1.0
3.3641499856336465 -2.5027128e-09 1.94229273881412
1.121382991333525 3.1717517190189715 1.9422927388141193
-2.5909987e-09 -1.8321133e-09 3.884586486670313
Si
2
Cartesian
3.92483875 2.77528125 6.7980237500000005
0.56069125 0.39646875 0.9711462500000001
"""

calc = AlignnAtomwiseCalculator(path=default_path())

def general_relaxer(atoms, calculator, fmax=0.05, steps=150, relax=True):
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator
    if not relax:
        return ase_atoms.get_potential_energy()
    ase_atoms = ExpCellFilter(ase_atoms)
    FIRE(ase_atoms).run(fmax=fmax, steps=steps)
    return ase_to_atoms(ase_atoms.atoms)

atoms = Poscar.from_string(poscar).atoms
atoms = general_relaxer(atoms, calc)

energies, volumes = [], []
for strain in np.arange(-0.1, 0.1, 0.01):
    s = atoms.strain_atoms(strain)
    a = s.ase_converter()
    a.calc = calc
    energies.append(a.get_potential_energy())
    volumes.append(s.volume)

plt.plot(volumes, energies, "-o")
plt.xlabel(r"Volume ($\AA^3$)")
plt.ylabel("Total energy (eV)")
```

## Choosing a model

`default_path()` returns the currently recommended checkpoint
(`alignn_ff_mptrj`). To pin a specific one, load it by name or pass a local path:

```python
from alignn.ff.ff import get_figshare_model_ff

calc = AlignnAtomwiseCalculator(
    path=get_figshare_model_ff(model_name="alignn_ff_matpes_pbe")
)
```

See the [pretrained ALIGNN-FF page](../pretrained/alignn-ff.md) for available
checkpoints.

## What you can do from here

Anything ASE supports — e.g.:

- `ase.optimize.fire.FIRE`, `BFGS` for structure relaxation
- `ase.phonons.Phonons` for finite-difference phonons (or use `phonopy`)
- `ase.md.VelocityVerlet`, `Langevin`, `NPT` for molecular dynamics
- `ase.neb.NEB` for minimum energy paths

## See also

- [Pretrained ALIGNN-FF checkpoints](../pretrained/alignn-ff.md)
- [CLI `run_alignn_ff.py`](../pretrained/alignn-ff.md#cli)
- [Colab: relaxation + phonons + interfaces](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb)
