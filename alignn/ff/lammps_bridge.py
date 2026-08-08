"""Drive LAMMPS MD with ALIGNN-FF via `fix external pf/callback`.

Runs LAMMPS from Python. Each step, LAMMPS hands positions to the callback;
we build an ASE Atoms, run AlignnAtomwiseCalculator, and return forces,
energy, and the global virial.

Usage:
    python -m alignn.ff.lammps_bridge \\
        --data system.data \\
        --types Si,O \\
        --steps 1000 --timestep 0.001 --temp 300
    (uses the default ALIGNN-FF model, default_path(); pass --model-path to
     override with your own model directory.)

Notes:
    - Requires the LAMMPS Python module (`pip install lammps` or build with
      -DPKG_PYTHON and -DBUILD_SHARED_LIBS=on).
    - `--types` maps LAMMPS atom type (1,2,...) to element symbols in order.
    - Single-rank only. For MPI runs, positions/forces must be gathered to
      rank 0 via lmp.gather_atoms / scatter_atoms inside the callback.
    - Expect ~10-100x slower than a compiled pair style.
"""

import argparse
import numpy as np
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress

from alignn.ff.calculators import AlignnAtomwiseCalculator, default_path

# eV/A^3 -> bar (LAMMPS `metal` pressure unit)
EV_PER_A3_TO_BAR = 1.602176634e6


def make_callback(calc, symbols_by_type):
    """Build the LAMMPS fix-external callback closure."""

    def callback(lmp, ntimestep, nlocal, tag, x, f):
        boxlo, boxhi, xy, yz, xz, *_ = lmp.extract_box()
        cell = np.array(
            [
                [boxhi[0] - boxlo[0], 0.0, 0.0],
                [xy, boxhi[1] - boxlo[1], 0.0],
                [xz, yz, boxhi[2] - boxlo[2]],
            ]
        )

        types = np.ctypeslib.as_array(
            lmp.extract_atom("type"), shape=(nlocal,)
        )
        symbols = [symbols_by_type[t] for t in types]

        atoms = Atoms(
            symbols=symbols,
            positions=np.array(x),
            cell=cell,
            pbc=True,
        )
        atoms.calc = calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        f[:] = forces

        try:
            stress_voigt = atoms.get_stress(voigt=True)  # eV/A^3
            stress_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)
            volume = atoms.get_volume()
            # LAMMPS wants the global virial tensor in pressure*volume units,
            # ordered xx, yy, zz, xy, xz, yz. Virial = -stress * V.
            virial = -stress_3x3 * volume * EV_PER_A3_TO_BAR
            v6 = [
                virial[0, 0],
                virial[1, 1],
                virial[2, 2],
                virial[0, 1],
                virial[0, 2],
                virial[1, 2],
            ]
            lmp.fix_external_set_virial_global("alignn", v6)
        except Exception:
            pass  # stress may be disabled on the calculator

        lmp.fix_external_set_energy_global("alignn", energy)

    return callback


def run(args):
    from lammps import lammps

    calc = AlignnAtomwiseCalculator(path=args.model_path or default_path())
    symbols_by_type = {
        i + 1: sym for i, sym in enumerate(args.types.split(","))
    }

    lmp = lammps()

    if args.input:
        # User-provided LAMMPS script. It must create the `alignn` fix via
        # `fix alignn all external pf/callback N 1` before any `run`.
        # We register the callback after setup commands but before `run`
        # by sourcing the file and letting the script drive execution.
        cb = make_callback(calc, symbols_by_type)
        # Execute commands up to (but not past) the first `run`, register
        # callback, then continue. Simpler: require the script to call
        # `python register_alignn_cb` — instead we just read line-by-line.
        with open(args.input) as fh:
            lines = fh.readlines()
        registered = False
        for line in lines:
            stripped = line.strip()
            if (
                not registered
                and stripped.startswith("run ")
                and "alignn" in "".join(lines[: lines.index(line)])
            ):
                lmp.set_fix_external_callback("alignn", cb, lmp)
                registered = True
            lmp.command(line.rstrip("\n"))
    else:
        lmp.commands_string(f"""
            units           metal
            atom_style      atomic
            boundary        p p p
            read_data       {args.data}
            pair_style      zero {args.cutoff}
            pair_coeff      * *
            neighbor        2.0 bin
            neigh_modify    every 1 delay 0 check yes
            fix             alignn all external pf/callback 1 1
            timestep        {args.timestep}
            velocity        all create {args.temp} {args.seed} mom yes rot yes
            fix             nve all nve
            fix             tfix all langevin {args.temp} {args.temp} 0.1 {args.seed}
            thermo          10
            thermo_style    custom step temp pe ke etotal press
            """)
        cb = make_callback(calc, symbols_by_type)
        lmp.set_fix_external_callback("alignn", cb, lmp)
        lmp.command(f"run {args.steps}")

    lmp.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="LAMMPS data file (default mode only)")
    p.add_argument(
        "--input",
        help="Path to a LAMMPS input script. Must define `fix alignn ... "
        "external pf/callback` before any `run` command. Overrides --data.",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="ALIGNN-FF model dir (default: the bundled default_path() model, "
        "i.e. matpes_pbe)",
    )
    p.add_argument(
        "--types",
        required=True,
        help="Comma-separated element symbols, ordered by LAMMPS atom type",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--timestep", type=float, default=0.001, help="ps")
    p.add_argument("--temp", type=float, default=300.0, help="K")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
