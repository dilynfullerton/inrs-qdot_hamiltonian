from RamanQD import RamanQD
import numpy as np
import itertools as it
from matplotlib import pyplot as plt


LATTICE_CONST_GaAs = .05635  # [nm]
REFIDX_GaAs = 3.8

ELECTRON_CHARGE = 1239.84193  # [nm]
FREE_ELECTRON_MASS = 0.00243  # [nm]

GAMMA_A1 = ELECTRON_CHARGE * 100  # [nm]


qd_GaAs = RamanQD(
    nmax=4, lmax=1,
    unit_cell=LATTICE_CONST_GaAs * np.eye(3, 3),
    refidx=lambda w: REFIDX_GaAs,
    radius=5.0,  # [nm]
    V1=1280.82844,  # [nm]
    V2=1921.34190,  # [nm]
    E_g=816.92161,  # [nm]
    mu_i1=.0665,  # [mu_0]
    mu_i2=.45,  # [mu_0]
    mu_01=1,  # [mu_0]
    mu_02=1,  # [mu_0]
    electron_charge=ELECTRON_CHARGE,
    free_electron_mass=FREE_ELECTRON_MASS,
    Gamma_f=GAMMA_A1*3,
    Gamma_a1=GAMMA_A1,
    Gamma_a2=GAMMA_A1,
    expected_roots_x_lj={
        (0, 1): [.001, 3.0, 5.86, 8.9, 11.92809],
        (0, 2): [.001, 3.1, 6.2462, 9.4, 12.47],
        (1, 1): [.001, 4.2, 7.3, 10.3, 13.369],
        (1, 2): [.001, 4.4, 7.67016736, 10.9, 13.9659],
    },
    verbose=True,
)

# Plot x roots
XDAT = np.linspace(1e-6, 15, 10000)
qd_GaAs.plot_root_fn_x(l=0, j=1, xdat=XDAT)
qd_GaAs.plot_root_fn_x(l=0, j=2, xdat=XDAT)
qd_GaAs.plot_root_fn_x(l=1, j=1, xdat=XDAT)
qd_GaAs.plot_root_fn_x(l=1, j=2, xdat=XDAT)

# Plot cross section
e_l = np.array([0, 0, 1])  # incident polarization
omega_l = 123.98419  # incident energy [nm]
e_s = np.array([1, 0, 0])  # secondary polarization

xdat = np.linspace(1, 10, 101)
ydatr = np.empty_like(xdat)
ydati = np.empty_like(xdat)
for x, i in zip(xdat, it.count()):
    omega_s = qd_GaAs.E_0 * x
    cross_sec = qd_GaAs.cross_section(omega_l=omega_l, e_l=e_l,
                                      omega_s=omega_s, e_s=e_s)
    ydatr[i] = cross_sec.real
    ydati[i] = cross_sec.imag

fig, ax = plt.subplots(1, 1)
ax.plot(xdat, ydatr, '-', color='red')
ax.plot(xdat, ydati, '-', color='blue')
plt.show()
