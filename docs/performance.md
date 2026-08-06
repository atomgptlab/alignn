# Performance

ALIGNN has been benchmarked across many public materials datasets. For the most
up-to-date numbers see [JARVIS-Leaderboard](https://atomgptlab.github.io/jarvis_leaderboard/).

## JARVIS-DFT 2021 — classification

| Classifier | Threshold | ALIGNN AUC |
|---|---|---|
| Metal / non-metal (OPT) | 0.01 eV | 0.92 |
| Metal / non-metal (MBJ) | 0.01 eV | 0.92 |
| Magnetic / non-magnetic | 0.05 µB | 0.91 |
| High / low SLME | 10 % | 0.83 |
| High / low spillage | 0.1 | 0.80 |
| Stable / unstable (ehull) | 0.1 eV | 0.94 |
| High / low n-Seebeck | -100 µV K⁻¹ | 0.88 |
| High / low p-Seebeck | 100 µV K⁻¹ | 0.92 |
| High / low n-PF | 1000 µW (mK²)⁻¹ | 0.74 |
| High / low p-PF | 1000 µW (mK²)⁻¹ | 0.74 |

## JARVIS-DFT 2021 — regression (MAE)

| Property | Units | MAD | CFID | CGCNN | ALIGNN | MAD:MAE |
|---|---|---|---|---|---|---|
| Formation energy | eV/atom | 0.86 | 0.14 | 0.063 | **0.033** | 26.06 |
| Bandgap (OPT) | eV | 0.99 | 0.30 | 0.20 | **0.14** | 7.07 |
| Total energy | eV/atom | 1.78 | 0.24 | 0.078 | **0.037** | 48.11 |
| Ehull | eV | 1.14 | 0.22 | 0.17 | **0.076** | 15.00 |
| Bandgap (MBJ) | eV | 1.79 | 0.53 | 0.41 | **0.31** | 5.77 |
| Bulk modulus K_v | GPa | 52.80 | 14.12 | 14.47 | **10.40** | 5.08 |
| Shear modulus G_v | GPa | 27.16 | 11.98 | 11.75 | **9.48** | 2.86 |
| Magnetic moment | µB | 1.27 | 0.45 | 0.37 | **0.26** | 4.88 |
| SLME (%) | — | 10.93 | 6.22 | 5.66 | **4.52** | 2.42 |
| Spillage | — | 0.52 | 0.39 | 0.40 | **0.35** | 1.49 |
| ε (DFPT: elec + ionic) | — | 45.81 | 43.71 | 38.78 | **28.15** | 1.63 |
| Max. piezo dij | C N⁻¹ | 24.57 | 36.41 | 34.71 | **20.57** | 1.19 |
| Exfoliation energy | meV/atom | 62.63 | 63.31 | 50.0 | **51.42** | 1.22 |

(Full table in the repository README — trimmed here for readability.)

## Materials Project 2018

| Property | Unit | MAD | CGCNN | MEGNet | SchNet | ALIGNN |
|---|---|---|---|---|---|---|
| Formation energy | eV/atom | 0.93 | 0.039 | 0.028 | 0.035 | **0.022** |
| Bandgap | eV | 1.35 | 0.388 | 0.33 | — | **0.218** |

## QM9 — MAE

| Target | Units | SchNet | MEGNet | DimeNet++ | ALIGNN |
|---|---|---|---|---|---|
| HOMO | eV | 0.041 | 0.043 | 0.0246 | **0.0214** |
| LUMO | eV | 0.034 | 0.044 | 0.0195 | **0.0195** |
| Gap | eV | 0.063 | 0.066 | **0.0326** | 0.0381 |
| µ | Debye | 0.033 | 0.050 | 0.0297 | **0.0146** |
| ZPVE | eV | 0.0017 | 0.00143 | **0.00121** | 0.0031 |

!!! info "QM9 caveat"
    See [issue #54](https://github.com/atomgptlab/alignn/issues/54) for known
    discrepancies related to the QM9 split.

## hMOF — regression

| Property | Unit | MAE | R² |
|---|---|---|---|
| Gravimetric surface area | m² g⁻¹ | 91.15 | 0.99 |
| Volumetric surface area | m² cm⁻³ | 107.81 | 0.91 |
| Void fraction | — | 0.017 | 0.98 |
| LCD | Å | 0.75 | 0.83 |
| PLD | Å | 0.92 | 0.78 |
| CO₂ adsorption (all) | mol kg⁻¹ | 0.18 | 0.95 |

## qMOF

MAE on electronic bandgap: **0.20 eV**.

## Open Catalyst — IS2RE 10k

| Model | CGCNN | DimeNet | SchNet | DimeNet++ | ALIGNN |
|---|---|---|---|---|---|
| 10k | 0.988 | 1.0117 | 1.059 | 0.8837 | **0.61** |

## Coming soon

OMDB, HOPV, QETB.

---

Claims of *best* performance should be verified on the latest
[JARVIS-Leaderboard](https://atomgptlab.github.io/jarvis_leaderboard/). Numbers from models
other than ALIGNN are reported as-published by the original authors and are not
necessarily reproduced in-house.
