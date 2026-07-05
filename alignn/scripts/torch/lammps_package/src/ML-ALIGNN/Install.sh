# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update
#
# Standard LAMMPS package install script (legacy `make` build system). The
# CMake build does not use this file. It copies pair_alignn.{cpp,h} into the
# parent src/ directory on `make yes-ml-alignn` and removes them on
# `make no-ml-alignn`.

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

# arg1 = file, arg2 = file it depends on

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# all package files

for file in *.cpp *.h; do
  action $file
done

# NOTE: the legacy make build will also need libtorch on the include/link path.
# Add libtorch include dirs to the LMP_INC line and -ltorch -ltorch_cpu -lc10
# (plus CUDA torch libs if applicable) to the LIB line of your Makefile, or use
# the CMake build (recommended) which handles this via ML-ALIGNN.cmake.
