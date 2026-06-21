# ML-ALIGNN package: native `pair_style alignn` backed by libtorch.
#
# This file is auto-included by the LAMMPS CMake build when the package is
# enabled with -D PKG_ML-ALIGNN=yes. The pair_alignn.{cpp,h} sources in
# src/ML-ALIGNN/ are picked up by the standard per-package source glob, so all
# this module has to do is locate libtorch and link it.
#
# Point CMake at libtorch via CMAKE_PREFIX_PATH, e.g.
#   -D CMAKE_PREFIX_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/share/cmake
# or download the standalone libtorch and pass its root.

find_package(Torch REQUIRED)

target_link_libraries(lammps PRIVATE ${TORCH_LIBRARIES})
target_compile_features(lammps PRIVATE cxx_std_17)

# The libtorch ABI flag must match how libtorch itself was compiled. The conda
# / pip PyTorch wheels are built with the new C++11 ABI for recent versions but
# this has varied; expose it as an env var so users can override.
if(DEFINED ENV{TORCH_CXX11_ABI})
  target_compile_definitions(lammps PRIVATE
                             _GLIBCXX_USE_CXX11_ABI=$ENV{TORCH_CXX11_ABI})
endif()
