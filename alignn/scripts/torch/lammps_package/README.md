# ML-ALIGNN — LAMMPS package scaffold

This directory is a **drop-in LAMMPS package** for the native `pair_style alignn`
(ALIGNN-FF via libtorch). The layout mirrors the LAMMPS source tree so the files
can be copied straight into a LAMMPS checkout, and it is structured to become an
upstream pull request to [lammps/lammps](https://github.com/lammps/lammps).

It packages the same `pair_alignn.{cpp,h}` that `build_lammps_alignn.sh` drops
into `src/`, but as a *proper optional package* (`make yes-ml-alignn` /
`-D PKG_ML-ALIGNN=yes`) instead of an unconditional edit to the main build.

## Layout

```
lammps_package/
├── src/ML-ALIGNN/
│   ├── pair_alignn.cpp              # pair style implementation
│   ├── pair_alignn.h               # pair style header
│   ├── Install.sh                  # legacy make-build install hook
│   └── README                      # package description (LAMMPS convention)
├── cmake/Modules/Packages/
│   └── ML-ALIGNN.cmake             # finds + links libtorch when pkg enabled
├── doc/src/
│   └── pair_alignn.rst             # user documentation
└── examples/PACKAGES/alignn/
    ├── in.alignn.si                # runnable NVE example
    └── README                      # how to generate inputs + run
```

Each path under here maps 1:1 onto the same path in a LAMMPS checkout.

## Installing into a LAMMPS checkout

From this directory, with `$LAMMPS` pointing at your LAMMPS clone:

```bash
LAMMPS=~/lammps

cp -r src/ML-ALIGNN                         "$LAMMPS/src/"
cp    cmake/Modules/Packages/ML-ALIGNN.cmake "$LAMMPS/cmake/Modules/Packages/"
cp    doc/src/pair_alignn.rst                "$LAMMPS/doc/src/"
cp -r examples/PACKAGES/alignn               "$LAMMPS/examples/PACKAGES/"
```

Then register the package name in the CMake package list
(`$LAMMPS/cmake/CMakeLists.txt`) — add `ML-ALIGNN` to the `STANDARD_PACKAGES`
set so `-D PKG_ML-ALIGNN=yes` is recognized and the `.cmake` module above is
auto-included.

### Build (CMake, recommended)

```bash
cd "$LAMMPS"
mkdir build && cd build
TORCH_CMAKE=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/share/cmake
cmake ../cmake \
  -D PKG_ML-ALIGNN=yes \
  -D CMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -D CMAKE_BUILD_TYPE=Release
cmake --build . -j
```

If your libtorch uses the old C++ ABI, also export
`TORCH_CXX11_ABI=$(python -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")`
before configuring (the CMake module reads it).

### Build (legacy make)

```bash
cd "$LAMMPS/src"
make yes-ml-alignn
# add libtorch include/link flags to your Makefile.<machine>, then:
make <machine>
```

## Note on the fully-automated path

For local development you usually do **not** need this package layout — the
script `../build_lammps_alignn.sh` clones LAMMPS, drops the pair style into
`src/`, links libtorch, builds, and installs the Python module in one shot. This
directory exists specifically to package the same code for an **upstream LAMMPS
contribution**.

## Checklist before opening an upstream PR

The files here are complete enough to build, but mainline LAMMPS acceptance
needs more. Open items:

- [ ] **MPI / domain decomposition.** The current pair style is single-rank: it
      builds the model graph from local atoms only and does not include ghost
      atoms as graph nodes (the model's energy is a global pool, so ghosts would
      inflate it). Multi-rank support is required for upstream acceptance.
- [ ] Register `ML-ALIGNN` in `cmake/CMakeLists.txt` (`STANDARD_PACKAGES`) and
      add it to `src/Makefile`'s package lists / `src/.gitignore`.
- [ ] Add the package + pair style to the doc indices: `doc/src/Packages_*.rst`,
      `doc/src/Commands_pair.rst`, and the `pair_style.rst` overview table.
- [ ] Add a regression/unit test under `unittest/` or a YAML test for the pair
      style, and wire the example into the example test harness.
- [ ] Make the libtorch dependency robust/optional in CMake the way `ML-IAP`'s
      PyTorch backend does, and document supported libtorch versions/ABI.
- [ ] Confirm licensing/attribution headers match LAMMPS conventions (GPL-2.0,
      contributor credit) on `pair_alignn.{cpp,h}`.
- [ ] Validate `pair_alignn` against the Python reference with
      `../validate_pair_alignn.py` and include the numbers in the PR.

See the upstream LAMMPS guide for contributing packages:
https://docs.lammps.org/Modify_contribute.html
