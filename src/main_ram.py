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
    r_0=2.0,  # [nm]
    omega_L=0.03782,  # [eV]
    omega_T=0.02951,  # [eV]
    beta_L=0.62488,
    beta_T=0.19590,
    electron_charge=ELECTRON_CHARGE,
    epsilon_inf_qdot=5.3,
    epsilon_inf_env=4.64,
    expected_roots_mu_l={
        0: [1.30, 2.30, 3.30, 4.31, 4.42],
        1: [.0001, 1.55, 2.15, 2.83, 3.80, 4.79],
        2: [.0001, 1.99, 2.95, 3.54, 4.30, 5.26],
        3: [.0000001, 2.39, 3.45, 4.30, 4.90, 5.75],
        4: [.0000001, 2.75, 3.88, 4.85, 5.65, 6.30],
    },
    num_n=2,
    num_l=2,
)

# Plot roots
XDAT_MU = np.linspace(1e-6, 10, 10000)
QD_GaAs_HAM.plot_root_function_mu(l=0, xdat=XDAT_MU, show=True)
QD_GaAs_HAM.plot_root_function_mu(l=1, xdat=XDAT_MU, show=True)

# assert False

# Plot Phi(r)
XDAT_R = np.linspace(0, QD_GaAs_HAM.r_0, 1001)
ydats_Phi_n0 = [np.empty_like(XDAT_R) for l in range(QD_GaAs_HAM.num_l)]
ydats_Phi_n1 = [np.empty_like(XDAT_R) for l in range(QD_GaAs_HAM.num_l)]
ydats_Phi_n2 = [np.empty_like(XDAT_R) for l in range(QD_GaAs_HAM.num_l)]
ydats_Phi_n3 = [np.empty_like(XDAT_R) for l in range(QD_GaAs_HAM.num_l)]
for l in range(QD_GaAs_HAM.num_l):
    for r, i in zip(XDAT_R, it.count()):
        ydats_Phi_n0[l][i] = QD_GaAs_HAM.Phi_ln(l=l, n=0)(r)
        ydats_Phi_n1[l][i] = QD_GaAs_HAM.Phi_ln(l=l, n=1)(r)
        ydats_Phi_n2[l][i] = QD_GaAs_HAM.Phi_ln(l=l, n=2)(r)
        ydats_Phi_n3[l][i] = QD_GaAs_HAM.Phi_ln(l=l, n=3)(r)
fig, axes = plt.subplots(4, 1, sharex=True)
for l in range(QD_GaAs_HAM.num_l):
    axes[0].plot(XDAT_R, ydats_Phi_n0[l], label='l={}'.format(l))
    axes[1].plot(XDAT_R, ydats_Phi_n1[l], label='l={}'.format(l))
    axes[2].plot(XDAT_R, ydats_Phi_n2[l], label='l={}'.format(l))
    axes[3].plot(XDAT_R, ydats_Phi_n3[l], label='l={}'.format(l))
axes[0].set_title('Phi(r)')
axes[0].legend()
axes[0].set_ylabel('n=0')
axes[1].set_ylabel('n=1')
axes[2].set_ylabel('n=2')
axes[3].set_ylabel('n=3')
plt.show()

# assert False

QD_GaAs_RAMAN = RamanQD(
    hamiltonian=QD_GaAs_HAM,
    V_v=2.5,  # [eV]
    V_c=1.9,  # [eV]
    E_gap=2.6,  # [eV]
    # mu_i1=.18 * FREE_ELECTRON_MASS,
    # mu_i2=.51 * FREE_ELECTRON_MASS,
    m_eff_e=1.8 * FREE_ELECTRON_MASS,
    m_eff_h=5.1 * FREE_ELECTRON_MASS,
    m_e=FREE_ELECTRON_MASS,
    m_h=FREE_ELECTRON_MASS,
    Gamma_f=GAMMA_A1*3,
    Gamma_a_v=GAMMA_A1,
    Gamma_a_c=GAMMA_A1,
    Gamma_b_v=GAMMA_A1,
    Gamma_b_c=GAMMA_A1,
    expected_roots_x_lj={
        # (0, 1): [.001, 2.79, 5.36],
        # (0, 2): [.001, 3.00, 5.94, 8.34],
        # (1, 1): [.001, 3.96],
        # (1, 2): [.001, 4.29, 7.27],
        (0, 1): [-5.36, -2.76, .001, 2.79, 5.36],
        (0, 2): [-8.34, -5.94, -3.00, .001, 3.00, 5.94, 8.34],
        (1, 1): [-3.96, .001, 3.96],
        (1, 2): [-7.27, -4.29, .001, 4.29, 7.27],
        (2, 1): [.001],
        (2, 2): [.001],
        (3, 1): [.001],
        (3, 2): [.001],
        (4, 1): [.001],
        (4, 2): [.001],
    },
    num_n=7,
    num_l=2,
    verbose=True,
)

# Plot x roots
# XDAT_X = np.linspace(1e-6, 10, 10000)
XDAT_X = np.linspace(-10, 10, 10000)
QD_GaAs_RAMAN.plot_root_fn_x(l=0, j=1, xdat=XDAT_X, show=True)
QD_GaAs_RAMAN.plot_root_fn_x(l=0, j=2, xdat=XDAT_X, show=True)
QD_GaAs_RAMAN.plot_root_fn_x(l=1, j=1, xdat=XDAT_X, show=True)
QD_GaAs_RAMAN.plot_root_fn_x(l=1, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=2, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=2, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=3, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=3, j=2, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=4, j=1, xdat=XDAT_X, show=True)
# QD_GaAs_RAMAN.plot_root_fn_x(l=4, j=2, xdat=XDAT_X, show=True)

# assert False

# Plot cross section
e_l = np.array([1, 0, 0])  # incident polarization
# omega_l = 10  # incident energy [eV]
omega_l = 2.4  # incident energy [eV]
# omega_l = .03782
e_s = np.array([0, 0, 1])  # secondary polarization

xdat = np.linspace(1e-4, 1, 101)
ydatr = np.empty_like(xdat)
ydati = np.empty_like(xdat)
for x, i in zip(xdat, it.count()):
    # omega_s = omega_l - x
    # omega_s = QD_GaAs_RAMAN.E_0 * x
    omega_s = x
    cross_sec = QD_GaAs_RAMAN.differential_raman_efficiency(
        omega_l=omega_l, e_l=e_l, omega_s=omega_s,
        # e_s=e_s,
        e_s=None,
        phonon_assist=True,
    )
    ydatr[i] = cross_sec.real
    ydati[i] = cross_sec.imag

fig, ax = plt.subplots(1, 1)

ax.plot(xdat, ydatr, '-', color='red')
ax.plot(xdat, ydati, '-', color='blue')

# ax.axvline(QD_GaAs_HAM.omega_L / QD_GaAs_RAMAN.E_0, ls='--', color='green', alpha=.7)
# ax.axvline(QD_GaAs_HAM.omega_T / QD_GaAs_RAMAN.E_0, ls='--', color='orange', alpha=.7)
ax.axvline(QD_GaAs_HAM.omega_L, ls='--', color='green', alpha=.7)
ax.axvline(QD_GaAs_HAM.omega_T, ls='--', color='orange', alpha=.7)
plt.show()
