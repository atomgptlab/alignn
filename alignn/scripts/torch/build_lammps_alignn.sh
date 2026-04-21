#!/bin/bash
# Build LAMMPS with pair_alignn, linking against the conda-env libtorch.
# Self-contained: clones LAMMPS, drops in USER-ALIGNN, patches cmake, builds,
# installs the Python module into the active conda env.

set -euo pipefail

LAMMPS_TAG="stable_29Aug2024_update1"
LAMMPS_DIR="$HOME/lammps-alignn"
ALIGNN_REPO="/home/kamalch/Software/ollama311/alignn"
PAIR_SRC="$ALIGNN_REPO/scripts/torch/pair_alignn"

# ── discover conda env's libtorch ─────────────────────────────────────────
TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_CMAKE="$TORCH_DIR/share/cmake/Torch"
TORCH_CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

echo "── environment ──────────────────────────"
echo "TORCH_DIR       = $TORCH_DIR"
echo "TORCH_CXX11_ABI = $TORCH_CXX11_ABI"
echo "python          = $(which python)"
echo "cmake           = $(cmake --version | head -1)"
echo

# ── clone LAMMPS ─────────────────────────────────────────────────────────
if [ ! -d "$LAMMPS_DIR" ]; then
    echo "[1/5] cloning LAMMPS $LAMMPS_TAG..."
    git clone --depth=1 --branch "$LAMMPS_TAG" \
        https://github.com/lammps/lammps.git "$LAMMPS_DIR"
else
    echo "[1/5] LAMMPS dir exists at $LAMMPS_DIR, reusing"
fi

# ── install pair_alignn as USER-ALIGNN package ───────────────────────────
echo "[2/5] installing pair_alignn into USER-ALIGNN..."
PKG_DIR="$LAMMPS_DIR/src/USER-ALIGNN"
mkdir -p "$PKG_DIR"
cp "$PAIR_SRC/pair_alignn.cpp" "$PKG_DIR/"
cp "$PAIR_SRC/pair_alignn.h"   "$PKG_DIR/"

cat > "$PKG_DIR/Install.sh" <<'SHEOF'
#!/bin/sh
action() {
    if test -e ../$1 && test ! -e ../$1.bak ; then
        mv ../$1 ../$1.bak
    fi
    if test ! -e $1 ; then return ; fi
    cp $1 ../$1
}
if (test $1 = 1) ; then
    action pair_alignn.cpp
    action pair_alignn.h
elif (test $1 = 0) ; then
    for f in pair_alignn.cpp pair_alignn.h ; do
        if test -e ../$f ; then rm -f ../$f ; fi
        if test -e ../${f}.bak ; then mv ../${f}.bak ../$f ; fi
    done
fi
SHEOF
chmod +x "$PKG_DIR/Install.sh"

# ── patch LAMMPS cmake to register USER-ALIGNN ───────────────────────────
CMAKE_EXTRA="$LAMMPS_DIR/cmake/Modules/Packages/USER-ALIGNN.cmake"
mkdir -p "$(dirname "$CMAKE_EXTRA")"
cat > "$CMAKE_EXTRA" <<'CMEOF'
find_package(Torch REQUIRED)

file(GLOB ALIGNN_SOURCES
     ${LAMMPS_SOURCE_DIR}/USER-ALIGNN/pair_alignn.cpp)
target_sources(lammps PRIVATE ${ALIGNN_SOURCES})
target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/USER-ALIGNN)
target_link_libraries(lammps PRIVATE ${TORCH_LIBRARIES})
target_compile_features(lammps PRIVATE cxx_std_17)
CMEOF

# Append USER-ALIGNN to STANDARD_PACKAGES if not already there
MAIN_CMAKE="$LAMMPS_DIR/cmake/CMakeLists.txt"
if ! grep -q "USER-ALIGNN" "$MAIN_CMAKE"; then
    # Find the line with STANDARD_PACKAGES and append our package.
    python3 - <<PYEOF
import re
p = "$MAIN_CMAKE"
s = open(p).read()
# Add USER-ALIGNN to STANDARD_PACKAGES list.
s = re.sub(r"(STANDARD_PACKAGES\s+[A-Z_0-9\s]+)",
           r"\1\n  USER-ALIGNN", s, count=1)
open(p, "w").write(s)
print(f"patched {p}")
PYEOF
fi

# ── configure + build ────────────────────────────────────────────────────
BUILD_DIR="$LAMMPS_DIR/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[3/5] cmake configure..."
cmake ../cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLAMMPS_EXCEPTIONS=ON \
    -DPKG_USER-ALIGNN=ON \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}" \
    -DPython_EXECUTABLE="$(which python)" \
    2>&1 | tail -40

echo
echo "[4/5] building (this takes a while)..."
cmake --build . -j"$(nproc)" 2>&1 | tail -20

# ── install the Python module ────────────────────────────────────────────
echo "[5/5] installing lammps Python module..."
cmake --build . --target install-python 2>&1 | tail -10

# ── verify ───────────────────────────────────────────────────────────────
echo
echo "── verification ─────────────────────────"
python - <<PYEOF
from lammps import lammps
l = lammps()
styles = l.available_styles("pair")
has = "alignn" in styles
print(f"LAMMPS version: {l.version()}")
print(f"pair_alignn available: {has}")
if not has:
    print("available pair styles (excerpt):", [s for s in styles if s.startswith(('a','b'))])
l.close()
PYEOF
