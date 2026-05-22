// Native LAMMPS pair style for ALIGNN-FF (via libtorch).
// Single-rank MVP: uses only LOCAL atoms + shift-based edges. The model's
// energy readout is a global pool over all graph nodes, so ghost atoms
// must NOT be passed as graph nodes — they'd inflate the sum.

#include "pair_alignn.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace LAMMPS_NS;

static int element_to_Z(const std::string &sym) {
  static const std::vector<std::string> periodic = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P",
    "S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu",
    "Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc",
    "Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba",
  };
  for (size_t i = 0; i < periodic.size(); ++i)
    if (periodic[i] == sym) return static_cast<int>(i + 1);
  return -1;
}

PairAlignn::PairAlignn(LAMMPS *lmp) : Pair(lmp) {
  single_enable = 0;
  restartinfo   = 0;
  one_coeff     = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
}

PairAlignn::~PairAlignn() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairAlignn::allocate() {
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq,   n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
}

void PairAlignn::settings(int narg, char **arg) {
  if (narg < 1 || narg > 2)
    error->all(FLERR, "pair_style alignn: cutoff (Å) [max_neighbors]");
  cutoff_ = std::stod(arg[0]);
  if (cutoff_ <= 0.0)
    error->all(FLERR, "pair_style alignn: cutoff must be positive");
  if (narg == 2) max_neighbors_ = std::stoi(arg[1]);
}

void PairAlignn::coeff(int narg, char **arg) {
  if (!allocated) allocate();
  if (narg != 3 + atom->ntypes)
    error->all(FLERR,
               "pair_coeff * * model.pt <Sym for type 1> [<Sym for type 2> ...]");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff alignn requires * *");

  const std::string model_path = arg[2];
  type_to_Z_.assign(atom->ntypes + 1, 0);
  for (int t = 1; t <= atom->ntypes; ++t) {
    std::string sym = arg[2 + t];
    int Z = element_to_Z(sym);
    if (Z < 0) error->all(FLERR,
                          (std::string("Unknown element: ") + sym).c_str());
    type_to_Z_[t] = Z;
  }

  device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  try {
    model_ = torch::jit::load(model_path, device_);
    model_.eval();
  } catch (const c10::Error &e) {
    error->all(FLERR,
               (std::string("Failed to load TorchScript model: ") + e.what()).c_str());
  }
  model_loaded_ = true;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) setflag[i][j] = 1;
}

void PairAlignn::init_style() {
  if (!model_loaded_)
    error->all(FLERR, "pair_style alignn requires pair_coeff * * <model.pt> ...");
  if (atom->tag_enable == 0)
    error->all(FLERR, "pair_style alignn requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "pair_style alignn requires newton on");
  if (atom->map_style == Atom::MAP_NONE) atom->map_init();
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

double PairAlignn::init_one(int /*i*/, int /*j*/) { return cutoff_; }

// ---------------------------------------------------------------------------
void PairAlignn::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int *type  = atom->type;
  tagint *tag = atom->tag;

  int *ilist       = list->ilist;
  int *numneigh    = list->numneigh;
  int **firstneigh = list->firstneigh;
  const int inum   = list->inum;

  const double hx = domain->xprd;
  const double hy = domain->yprd;
  const double hz = domain->zprd;
  const double xy = domain->xy;
  const double xz = domain->xz;
  const double yz = domain->yz;
  const bool triclinic = (xy != 0.0) || (xz != 0.0) || (yz != 0.0);

  // --- positions / atomic numbers: LOCAL atoms only ---
  auto t_opts = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);
  auto l_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  torch::Tensor pos = torch::empty({nlocal, 3}, t_opts);
  torch::Tensor Z   = torch::empty({nlocal},    l_opts);
  {
    auto pa = pos.accessor<float, 2>();
    auto Za = Z.accessor<long, 1>();
    for (int i = 0; i < nlocal; ++i) {
      pa[i][0] = static_cast<float>(x[i][0]);
      pa[i][1] = static_cast<float>(x[i][1]);
      pa[i][2] = static_cast<float>(x[i][2]);
      Za[i]    = static_cast<long>(type_to_Z_[type[i]]);
    }
  }

  // --- edges: src/dst both LOCAL; shift carries PBC image offset ---
  std::vector<long>  src_vec, dst_vec;
  std::vector<float> shift_flat;
  src_vec.reserve(nlocal * 16);
  dst_vec.reserve(nlocal * 16);
  shift_flat.reserve(nlocal * 16 * 3);
  const double cut2 = cutoff_ * cutoff_;

  // k-nearest (same convention as jarvis `neighbor_strategy="k-nearest"`):
  // for each local i, keep up to max_neighbors_ closest j's within cutoff.
  struct Cand { double r2; int dst_local; float shx, shy, shz; };
  std::vector<Cand> buf;
  buf.reserve(256);

  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (i >= nlocal) continue;
    const double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    const int jnum  = numneigh[i];
    const int *jlist = firstneigh[i];
    buf.clear();
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      const double dxr = x[j][0] - xi;
      const double dyr = x[j][1] - yi;
      const double dzr = x[j][2] - zi;
      const double r2  = dxr*dxr + dyr*dyr + dzr*dzr;
      if (r2 >= cut2) continue;

      int dst_local;
      double shx = 0.0, shy = 0.0, shz = 0.0;
      if (j < nlocal) {
        dst_local = j;
      } else {
        tagint jtag = tag[j];
        int owner = atom->map(jtag);
        if (owner < 0 || owner >= nlocal) continue;
        dst_local = owner;
        const double dox = x[j][0] - x[owner][0];
        const double doy = x[j][1] - x[owner][1];
        const double doz = x[j][2] - x[owner][2];
        (void)triclinic;
        shx = std::round(dox / hx);
        shy = std::round(doy / hy);
        shz = std::round(doz / hz);
      }
      buf.push_back({r2, dst_local,
                     static_cast<float>(shx),
                     static_cast<float>(shy),
                     static_cast<float>(shz)});
    }
    // Sort by r2 ascending, keep first max_neighbors_.
    if (static_cast<int>(buf.size()) > max_neighbors_) {
      std::partial_sort(buf.begin(),
                        buf.begin() + max_neighbors_,
                        buf.end(),
                        [](const Cand &a, const Cand &b) { return a.r2 < b.r2; });
      buf.resize(max_neighbors_);
    }
    for (const auto &c : buf) {
      src_vec.push_back(i);
      dst_vec.push_back(c.dst_local);
      shift_flat.push_back(c.shx);
      shift_flat.push_back(c.shy);
      shift_flat.push_back(c.shz);
    }
  }
  const long nedges = static_cast<long>(src_vec.size());
  torch::Tensor src   = torch::from_blob(src_vec.data(),  {nedges},    torch::kLong).clone();
  torch::Tensor dst   = torch::from_blob(dst_vec.data(),  {nedges},    torch::kLong).clone();
  torch::Tensor shift = torch::from_blob(shift_flat.data(),{nedges, 3}, torch::kFloat32)
                            .to(dtype_).clone();

  torch::Tensor lat = torch::zeros({3, 3}, t_opts);
  {
    auto la = lat.accessor<float, 2>();
    la[0][0] = static_cast<float>(hx);
    la[1][1] = static_cast<float>(hy);
    la[2][2] = static_cast<float>(hz);
    la[1][0] = static_cast<float>(xy);
    la[2][0] = static_cast<float>(xz);
    la[2][1] = static_cast<float>(yz);
  }

  pos   = pos.to(device_).set_requires_grad(true);
  lat   = lat.to(device_);
  Z     = Z.to(device_);
  src   = src.to(device_);
  dst   = dst.to(device_);
  shift = shift.to(device_);

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(pos);
  inputs.emplace_back(lat);
  inputs.emplace_back(Z);
  inputs.emplace_back(src);
  inputs.emplace_back(dst);
  inputs.emplace_back(shift);
  inputs.emplace_back(static_cast<bool>(vflag_global));

  c10::IValue result;
  {
    torch::AutoGradMode grad(true);
    result = model_.get_method("forward_tensors_z")(inputs);
  }
  auto dict = result.toGenericDict();
  torch::Tensor energy_t = dict.at("energy").toTensor().to(torch::kCPU).to(torch::kDouble);
  torch::Tensor forces_t = dict.at("forces").toTensor().to(torch::kCPU).to(torch::kDouble);

  auto fa = forces_t.accessor<double, 2>();
  for (int i = 0; i < nlocal; ++i) {
    f[i][0] += fa[i][0];
    f[i][1] += fa[i][1];
    f[i][2] += fa[i][2];
  }
  if (eflag_global) eng_vdwl += energy_t.item<double>();

  if (vflag_global && dict.contains("stress")) {
    torch::Tensor s = dict.at("stress").toTensor().to(torch::kCPU).to(torch::kDouble);
    auto sa = s.accessor<double, 2>();
    const double V = hx * hy * hz;
    virial[0] += -sa[0][0] * V;
    virial[1] += -sa[1][1] * V;
    virial[2] += -sa[2][2] * V;
    virial[3] += -sa[0][1] * V;
    virial[4] += -sa[0][2] * V;
    virial[5] += -sa[1][2] * V;
  }
}
