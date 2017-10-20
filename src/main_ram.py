from RamanQD import RamanQD
from HamiltonianQD import HamiltonianQD
import numpy as np
import itertools as it
from matplotlib import pyplot as plt


LATTICE_CONST_GaAs = .05635  # [nm]
REFIDX_GaAs = 3.8

ELECTRON_CHARGE = 1  # [eV]
FREE_ELECTRON_MASS = 1

GAMMA_A1 = ELECTRON_CHARGE / 100


QD_GaAs_HAM = HamiltonianQD(
    n_max=6,
    l_max=3,
    r_0=2.0,  # [nm]
    omega_L=305,  # [cm-1]
    omega_T=238,  # [cm-1]
    beta_L=5.04e3,
    beta_T=1.58e3,
    electron_charge=ELECTRON_CHARGE,
    epsilon_inf_qdot=5.3,
    epsilon_inf_env=4.64,
    expected_roots_mu_l={
        0: [.0001, 1.35, 2.38, 3.39, 4.39, 4.49, 5.38],
        1: [.0001, 1.62, 2.22, 2.90, 3.87, 4.86, 5.81],
        2: [.0001, 2.06, 3.03, 3.61, 4.37, 5.33, 6.31],
        3: [.0001, 2.50, 3.50, 4.50, 5.00, 5.80, 6.80],
        4: [.0001, 2.75, 4.00, 4.90, 5.75, 6.30, 7.25],
    }
)

XDAT_MU = np.linspace(1e-6, 10, 1000)
# QD_GaAs_HAM.plot_root_function_mu(l=0, xdat=XDAT_MU, show=True)
# QD_GaAs_HAM.plot_root_function_mu(l=1, xdat=XDAT_MU, show=True)
# QD_GaAs_HAM.plot_root_function_mu(l=2, xdat=XDAT_MU, show=True)
# QD_GaAs_HAM.plot_root_function_mu(l=3, xdat=XDAT_MU, show=True)
# QD_GaAs_HAM.plot_root_function_mu(l=4, xdat=XDAT_MU, show=True)


QD_GaAs_RAMAN = RamanQD(
    hamiltonian=QD_GaAs_HAM,
    unit_cell=LATTICE_CONST_GaAs * np.eye(3, 3),
    refidx=lambda w: REFIDX_GaAs,
    V1=2.5,  # [eV]
    V2=1.9,  # [eV]
    E_g=2.6,  # [eV]
    mu_i1=.18 * FREE_ELECTRON_MASS,
    mu_i2=.51 * FREE_ELECTRON_MASS,
    mu_01=FREE_ELECTRON_MASS,
    mu_02=FREE_ELECTRON_MASS,
    free_electron_mass=FREE_ELECTRON_MASS,
    Gamma_f=GAMMA_A1*3,
    Gamma_a1=GAMMA_A1,
    Gamma_a2=GAMMA_A1,
    Gamma_b1=GAMMA_A1,
    Gamma_b2=GAMMA_A1,
    expected_roots_x_lj={
        (0, 1): [.001, 1.59],
        (0, 2): [.001, 2.06],
        (1, 1): [.001],
        (1, 2): [.001],
        (2, 1): [.001],
        (2, 2): [.001],
        (3, 1): [.001],
        (3, 2): [.001],
        (4, 1): [.001],
        (4, 2): [.001],
    },
    verbose=True,
)

# Plot x roots
XDAT_X = np.linspace(1e-6, 2.1, 10000)
# QD_GaAs_RAMAN.plot_root_fn_x(l=0, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=0, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=1, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=1, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=2, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=2, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=3, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=3, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=4, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=4, j=2, xdat=XDAT_X, show=True)

# Plot cross section
e_l = np.array([0, 0, 1])  # incident polarization
omega_l = 123.98419  # incident energy [nm]
e_s = np.array([1, 0, 0])  # secondary polarization

xdat = np.linspace(1, 10, 101)
ydatr = np.empty_like(xdat)
ydati = np.empty_like(xdat)
for x, i in zip(xdat, it.count()):
    omega_s = QD_GaAs_RAMAN.E_0 * x
    cross_sec = QD_GaAs_RAMAN.cross_section(
        omega_l=omega_l, e_l=e_l, omega_s=omega_s, e_s=e_s)
    ydatr[i] = cross_sec.real
    ydati[i] = cross_sec.imag

fig, ax = plt.subplots(1, 1)
ax.plot(xdat, ydatr, '-', color='red')
ax.plot(xdat, ydati, '-', color='blue')
plt.show()
