"""Generate a TOY scalar-property (radius graph) dataset -> id_prop.json (inline jarvis Atoms dicts).

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
    target = -5.0 + 0.1 * i          # TOY scalar target
    data.append({"jid": f"toy-{i}", "atoms": a.to_dict(), "target": target})

json.dump(data, open("id_prop.json", "w"))
print(f"wrote id_prop.json with {len(data)} entries")
