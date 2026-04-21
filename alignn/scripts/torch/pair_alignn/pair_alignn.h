// ALIGNN-FF native LAMMPS pair style.
// Drop this header + pair_alignn.cpp into LAMMPS src/USER-ALIGNN/ and rebuild.
//
// Input script usage:
//   pair_style  alignn
//   pair_coeff  * * alignn_ff.pt  Si  # TorchScript model path, then element
//                                     # symbols ordered by LAMMPS atom type.

#ifdef PAIR_CLASS
// clang-format off
PairStyle(alignn,PairAlignn)
// clang-format on
#else

#ifndef LMP_PAIR_ALIGNN_H
#define LMP_PAIR_ALIGNN_H

#include "pair.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>

namespace LAMMPS_NS {

class PairAlignn : public Pair {
 public:
  PairAlignn(class LAMMPS *);
  ~PairAlignn() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void allocate();

 protected:
  torch::jit::script::Module model_;
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kFloat32;

  double cutoff_ = 8.0;                      // must match trained cutoff
  int    max_neighbors_ = 12;                // k-nearest cap per atom
  std::vector<int> type_to_Z_;               // LAMMPS type (1-indexed) -> atomic number

  bool model_loaded_ = false;
  bool debug_ = false;
};

}  // namespace LAMMPS_NS

#endif
#endif
