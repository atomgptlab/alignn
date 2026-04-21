#!/bin/bash
# ---------------------------------------------------------------------------
# build_lammps_alignn.sh
#
# One-shot build of LAMMPS stable_29Aug2024_update1 with the native
# pair_alignn style linked against the conda-env libtorch.
#
# Does:
#   1. Discovers PyTorch/CUDA/ABI in the active conda env
#   2. (Optional) installs matching cuda-toolkit + MKL if missing
#   3. Clones LAMMPS to $LAMMPS_DIR (default ~/lammps-alignn)
#   4. Copies pair_alignn.{cpp,h} from this repo into LAMMPS src/
#   5. Appends find_package(Torch) + target_link_libraries to main CMakeLists.txt
#   6. Configures + builds with Ninja
#   7. Installs the lammps Python module into the active env
#   8. Overwrites any pre-existing liblammps.so.0 in the Python prefix with the
#      pair_alignn build (fixes the standalone `lmp` binary's RPATH picking up
#      a stale .so).
#
# Works in a conda env (uses mamba for CUDA/MKL) OR in system Python like
# Colab (falls back to `pip install nvidia-cuda-*-cu12 mkl mkl-devel mkl-include`).
#   9. Verifies "alignn" appears in both Python module and the standalone binary
#
# Env vars:
#   LAMMPS_DIR       (default: $HOME/lammps-alignn)
#   LAMMPS_TAG       (default: stable_29Aug2024_update1)
#   SKIP_CUDA_MKL    set to 1 to skip the mamba install step
#   JOBS             (default: $(nproc))
# ---------------------------------------------------------------------------
set -euo pipefail

LAMMPS_TAG="${LAMMPS_TAG:-stable_29Aug2024_update1}"
LAMMPS_DIR="${LAMMPS_DIR:-$HOME/lammps-alignn}"
JOBS="${JOBS:-$(nproc)}"

# Locate the alignn repo root from *this* script's path (so it's portable).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_SRC="$SCRIPT_DIR/pair_alignn"
ALIGNN_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "── paths ────────────────────────────────"
echo "script dir   : $SCRIPT_DIR"
echo "alignn repo  : $ALIGNN_REPO"
echo "pair sources : $PAIR_SRC"
echo "LAMMPS target: $LAMMPS_DIR (tag $LAMMPS_TAG)"
echo

# ── detect Python install prefix (conda env OR system python, e.g. Colab) ──
PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
PREFIX="${CONDA_PREFIX:-$PY_PREFIX}"
HAS_MAMBA=0
command -v mamba >/dev/null 2>&1 && HAS_MAMBA=1
command -v cmake >/dev/null || {
    echo "ERROR: cmake not found. Install it (apt / conda / pip) and rerun." >&2
    exit 1
}
command -v ninja >/dev/null || pip install -q ninja

# ── discover torch + CUDA ABI ────────────────────────────────────────────
TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_CMAKE="$TORCH_DIR/share/cmake/Torch"
TORCH_CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or 'cpu')")

echo "── environment ──────────────────────────"
echo "python           = $(which python)"
echo "torch            = $(python -c 'import torch; print(torch.__version__)')"
echo "torch CUDA       = $TORCH_CUDA"
echo "torch CXX11 ABI  = $TORCH_CXX11_ABI"
echo "cmake            = $(cmake --version | head -1)"
echo "prefix           = $PREFIX"
echo "has mamba        = $HAS_MAMBA"
echo

# ── ensure CUDA toolkit + MKL match libtorch ─────────────────────────────
if [ "${SKIP_CUDA_MKL:-0}" != "1" ] && [ "$TORCH_CUDA" != "cpu" ]; then
    # CUDA toolkit check: nvcc matches torch CUDA?
    NVCC_VER="none"
    if command -v nvcc >/dev/null; then
        NVCC_VER=$(nvcc --version | awk '/release/{gsub(",","",$5); print $5}')
    fi
    NEED_CUDA=0
    [ "$NVCC_VER" != "$TORCH_CUDA" ] && NEED_CUDA=1

    # MKL headers check — look in common places
    MKL_H=""
    for d in "$PREFIX/include" "$PY_PREFIX/include" /usr/include /usr/local/include; do
        [ -f "$d/mkl.h" ] && MKL_H="$d/mkl.h" && break
    done
    NEED_MKL=0
    [ -z "$MKL_H" ] && NEED_MKL=1

    if [ $HAS_MAMBA -eq 1 ]; then
        if [ $NEED_CUDA -eq 1 ] || [ $NEED_MKL -eq 1 ]; then
            echo "── installing cuda-toolkit=$TORCH_CUDA + mkl-devel via mamba ──"
            mamba install -y -c nvidia -c conda-forge \
                "cuda-toolkit=$TORCH_CUDA" mkl-devel mkl-include
        fi
    else
        # Pip fallback (Colab / system python). Grab CUDA + MKL from pip wheels.
        PIP_ARGS=()
        if [ $NEED_CUDA -eq 1 ]; then
            cu_short=$(echo "$TORCH_CUDA" | tr -d '.')
            echo "── installing CUDA ${TORCH_CUDA} dev headers via pip (cu${cu_short}) ──"
            PIP_ARGS+=(\
                "nvidia-cuda-nvcc-cu${cu_short%??}==${TORCH_CUDA}.*" \
                "nvidia-cuda-runtime-cu${cu_short%??}==${TORCH_CUDA}.*" \
                "nvidia-cuda-cupti-cu${cu_short%??}==${TORCH_CUDA}.*" \
            ) || true
        fi
        if [ $NEED_MKL -eq 1 ]; then
            echo "── installing mkl via pip ──"
            PIP_ARGS+=(mkl mkl-devel mkl-include)
        fi
        [ ${#PIP_ARGS[@]} -gt 0 ] && pip install -q "${PIP_ARGS[@]}" || true

        # Re-locate MKL headers after install
        for d in "$PREFIX/include" "$PY_PREFIX/include" /usr/include /usr/local/include; do
            [ -f "$d/mkl.h" ] && MKL_H="$d/mkl.h" && break
        done

        # Colab ships nvcc in /usr/local/cuda; export for CMake's detection
        if [ -x /usr/local/cuda/bin/nvcc ] && ! command -v nvcc >/dev/null; then
            export PATH="/usr/local/cuda/bin:$PATH"
            export CUDA_HOME=/usr/local/cuda
        fi
    fi
fi

# ── clone LAMMPS ─────────────────────────────────────────────────────────
if [ ! -d "$LAMMPS_DIR" ]; then
    echo "[1/5] cloning LAMMPS $LAMMPS_TAG..."
    git clone --depth=1 --branch "$LAMMPS_TAG" \
        https://github.com/lammps/lammps.git "$LAMMPS_DIR"
else
    echo "[1/5] LAMMPS dir exists at $LAMMPS_DIR, reusing"
fi

# ── drop pair_alignn into src/ (no USER- package machinery) ──────────────
echo "[2/5] installing pair_alignn.{cpp,h} into LAMMPS src/..."
cp "$PAIR_SRC/pair_alignn.cpp" "$LAMMPS_DIR/src/pair_alignn.cpp"
cp "$PAIR_SRC/pair_alignn.h"   "$LAMMPS_DIR/src/pair_alignn.h"

# ── append unconditional libtorch link to the main CMakeLists.txt ────────
MAIN_CMAKE="$LAMMPS_DIR/cmake/CMakeLists.txt"
HOOK_MARKER="# ALIGNN-FF libtorch hook"
if ! grep -q "$HOOK_MARKER" "$MAIN_CMAKE"; then
    cat >> "$MAIN_CMAKE" <<EOF

# ── ALIGNN-FF (libtorch) ──────────────────────────────────────────────────
$HOOK_MARKER
# pair_alignn.cpp lives directly in src/ and is picked up by the main glob.
# We only need to supply libtorch includes + link.
find_package(Torch REQUIRED)
target_link_libraries(lammps PRIVATE \${TORCH_LIBRARIES})
target_compile_features(lammps PRIVATE cxx_std_17)
EOF
    echo "  appended libtorch hook to cmake/CMakeLists.txt"
else
    echo "  libtorch hook already present"
fi

# ── configure + build ────────────────────────────────────────────────────
BUILD_DIR="$LAMMPS_DIR/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CUDA headers are in a few possible places; collect any that exist.
CUDA_FLAGS=""
for d in \
    "$PREFIX/targets/x86_64-linux/include" \
    "$PY_PREFIX/targets/x86_64-linux/include" \
    /usr/local/cuda/include \
    /usr/local/cuda/targets/x86_64-linux/include; do
    [ -d "$d" ] && CUDA_FLAGS="$CUDA_FLAGS -I$d"
done
# MKL headers
MKL_INC=""
for d in "$PREFIX/include" "$PY_PREFIX/include" /usr/include /usr/local/include; do
    [ -f "$d/mkl.h" ] && MKL_INC="$d" && break
done

echo "[3/5] cmake configure..."
cmake ../cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLAMMPS_EXCEPTIONS=ON \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;$PREFIX;$PY_PREFIX" \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ${CUDA_FLAGS}" \
    ${MKL_INC:+-DMKL_INCLUDE_DIR=$MKL_INC} \
    -DPython_EXECUTABLE="$(which python)" \
    -DCMAKE_CUDA_COMPILER=$(command -v nvcc 2>/dev/null || echo "") \
    2>&1 | tail -8

echo
echo "[4/5] compiling (this takes 5–15 min)..."
cmake --build . -j"$JOBS" 2>&1 | tail -8

# ── install the lammps Python module ─────────────────────────────────────
echo "[5/5] installing lammps Python module into $PREFIX..."
install_ok=1
if cmake --build . --target install-python 2>&1 | tail -5; then
    # `install-python` may report success but actually fail silently on Colab
    # when its internal venv bootstrap breaks. Verify by importing.
    python -c "import lammps" 2>/dev/null || install_ok=0
else
    install_ok=0
fi

if [ $install_ok -eq 0 ]; then
    echo
    echo "  install-python target failed (common on Colab / system Python)."
    echo "  Falling back to manual copy of python/lammps → site-packages..."
    SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
    # Uninstall any stale lammps wheel first
    pip uninstall -y lammps 2>/dev/null || true
    rm -rf "$SITE_PKG/lammps"
    cp -r "$LAMMPS_DIR/python/lammps" "$SITE_PKG/lammps"
    # The Python module locates liblammps.so via rpath / LD_LIBRARY_PATH.
    # Put the .so next to the package so ctypes.CDLL finds it.
    cp "$BUILD_DIR/liblammps.so.0" "$SITE_PKG/lammps/liblammps.so"
    # Also place a copy in PY_PREFIX/lib so the standalone lmp binary's RPATH works
    mkdir -p "$PY_PREFIX/lib"
    cp "$BUILD_DIR/liblammps.so.0" "$PY_PREFIX/lib/liblammps.so.0"
    echo "  installed to $SITE_PKG/lammps"
fi

# ── fix standalone binary's RPATH by overwriting any pre-existing liblammps ──
for lib_dir in "$PREFIX/lib" "$PY_PREFIX/lib" /usr/lib /usr/local/lib; do
    if [ -f "$lib_dir/liblammps.so.0" ]; then
        echo "  syncing liblammps.so.0 → $lib_dir/ (for standalone lmp binary)"
        cp "$BUILD_DIR/liblammps.so.0" "$lib_dir/liblammps.so.0"
    fi
done

# ── verify ───────────────────────────────────────────────────────────────
echo
echo "── verification ─────────────────────────"

PY_OK=$(python - <<'PYEOF'
from lammps import lammps
l = lammps()
ok = "alignn" in l.available_styles("pair")
print("YES" if ok else "NO")
l.close()
PYEOF
)
if [ "$PY_OK" = "YES" ]; then
    echo "✓ Python module has pair_alignn"
else
    echo "✗ Python module does NOT have pair_alignn"
fi

if "$BUILD_DIR/lmp" -help 2>&1 | grep -q alignn; then
    echo "✓ Standalone binary has pair_alignn:  $BUILD_DIR/lmp"
else
    echo "✗ Standalone binary does NOT have pair_alignn"
fi

echo
echo "──────────────────────────────────────────"
echo "Done. Try:"
echo "  export_torchscript.py --model-dir <OutputDir> --out alignn_ff.pt"
echo "  $BUILD_DIR/lmp -in <your_input.in>"
echo "──────────────────────────────────────────"
