.. index:: pair_style alignn

pair_style alignn command
=========================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style alignn cutoff keyword

* alignn = name of this pair style
* cutoff = cutoff distance for the model graph (distance units, Angstroms)
* keyword = optional maximum number of neighbors per atom

Examples
""""""""

.. code-block:: LAMMPS

   pair_style alignn 5.0 12
   pair_coeff * * alignn_ff.pt Si

   pair_style alignn 5.0 12
   pair_coeff * * alignn_ff.pt Si O

Description
"""""""""""

Style *alignn* computes interatomic interactions using the ALIGNN-FF
(Atomistic Line Graph Neural Network Force-Field) machine-learning potential
:ref:`(Choudhary2023) <Choudhary2023>`. The potential is evaluated by calling a
TorchScript-exported ALIGNN-FF model through the libtorch (PyTorch C++) API. At
each timestep the pair style builds the atom graph from the LAMMPS neighbor
list, passes atomic numbers, positions, the cell, and periodic edge shifts to
the model, and reads back the total energy, per-atom forces, and (optionally)
the virial stress.

Because evaluation happens entirely in compiled C++/libtorch there is no Python
interpreter in the MD loop. If a CUDA-capable GPU and a CUDA build of libtorch
are available the model runs on the GPU automatically; otherwise it runs on the
CPU.

The mandatory *cutoff* argument and the optional max-neighbors keyword define
the graph the model sees and **must match the values the model was trained
with**. These are stored as the top-level ``cutoff`` and ``max_neighbors``
fields of the model's ``config.json``. For the default Materials-Project
``mps`` ALIGNN-FF model these are ``5.0`` and ``12``. Using values that differ
from training puts the model out of distribution and typically destabilizes the
simulation.

Only a single :doc:`pair_coeff <pair_coeff>` command is used with this style,
with the form:

.. code-block:: LAMMPS

   pair_coeff * * model.pt Sym1 Sym2 ...

where:

* the two asterisks map to all LAMMPS atom types,
* ``model.pt`` is the path to the TorchScript model file, and
* ``Sym1 Sym2 ...`` are chemical element symbols, one per LAMMPS atom type, in
  type order. These map each numeric atom type to the atomic number the model
  expects.

The TorchScript model file is produced from a trained ALIGNN-FF checkpoint with
the ``export_torchscript.py`` utility shipped with the ALIGNN package. The
default ``mps`` model can be downloaded and exported with the ``get_model.py``
helper in the ALIGNN ``examples/lammps`` directory.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This pair style does not support mixing. It is a manybody potential defined by
a single ``pair_coeff`` command as described above.

This pair style does not support the :doc:`pair_modify <pair_modify>` shift,
table, and tail options.

This pair style does not write its information to :doc:`binary restart files
<restart>`, since it is stored in the external model file. Thus, you need to
re-specify the pair_style and pair_coeff commands in an input script that reads
a restart file.

This pair style can only be used via the *pair* keyword of the :doc:`run_style
respa <run_style>` command. It does not support the *inner*, *middle*, *outer*
keywords.

Restrictions
""""""""""""

This pair style is part of the ML-ALIGNN package. It is only enabled if LAMMPS
was built with that package. See the :doc:`Build package <Build_package>` page
for more info.

This pair style requires libtorch (the PyTorch C++ distribution) to build and
run. The libtorch version's C++ ABI must be compatible with the compiler used
to build LAMMPS.

The current implementation runs on a single MPI rank (it builds the model graph
from local atoms only). It requires ``newton on`` and atom IDs.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

The optional max-neighbors keyword defaults to 12.

----------

.. _Choudhary2023:

**(Choudhary2023)** Choudhary, DeCost, Major, Butler, Thiyagalingam, Tavazza,
"Unified graph neural network force-field for the periodic table: solid state
applications", Digital Discovery, 2, 346-355 (2023).
