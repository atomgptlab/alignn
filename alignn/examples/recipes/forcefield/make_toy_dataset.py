"""Generate a TOY force-field (energy/forces/stress) dataset -> id_prop.json (inline jarvis Atoms dicts).

CAUTION: this is a tiny synthetic dataset for a smoke-test only. Real training
needs thousands+ of DFT-labeled structures and 100-300 epochs. Bump N below,
supply real targets, and raise `epochs`/`batch_size` in config_example.json.
"""
import json, random
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_jid_data

random.seed(0)
N = 40  # toy size -- increase to thousands for a real run

def rattle(a, amp=0.05):
    d = a.to_dict()
    d["coords"] = [[c + random.uniform(-amp, amp) for c in xyz] for xyz in d["coords"]]
    return Atoms.from_dict(d)

# base crystal (Si); swap for your own structures
base = Atoms.from_dict(get_jid_data(jid="JVASP-1002", dataset="dft_3d")["atoms"])

data = []
for i in range(N):
    a = rattle(base)
    nat = a.num_atoms
    # TOY labels: replace with DFT energy_per_atom (eV/atom), forces (Nx3 eV/A),
    # stresses (Voigt-6). Energy MUST be per atom.
    energy_per_atom = -5.0 + 0.01 * i
    forces = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(nat)]
    stresses = [random.uniform(-0.5, 0.5) for _ in range(6)]
    data.append({"jid": f"toy-{i}", "atoms": a.to_dict(),
                 "energy_per_atom": energy_per_atom, "forces": forces, "stresses": stresses})

json.dump(data, open("id_prop.json", "w"))
print(f"wrote id_prop.json with {len(data)} entries (energy_per_atom/forces/stresses)")
