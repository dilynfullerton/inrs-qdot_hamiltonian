import numpy as np
from PhononsModelSpace import PhononModelSpace
from ExitonModelSpace import ExitonModelSpace
from CavityModelSpace import CavityModelSpace
from RamanQD import RamanQD
from matplotlib import pyplot as plt
from os import path
import time
import pickle
from collections import deque
import itertools as it
from scipy import integrate as integ
# import warnings
# warnings.filterwarnings('error')

RADIUS = 2  # [nm]

# Phonon parameters
NMAX_PHONON = 9
LMAX_PHONON = 1
OMEGA_L = 305  # [cm-1]
OMEGA_T = 238  # [cm-1]
BETA_L = 5.04e1  # [nm/cm]
BETA_T = 1.58e1  # [nm/cm]
EPS_INF_IN = 5.72
EPS_INF_OUT = 4.64
XDAT_MU = np.linspace(1e-6, 10, 10000)
EXPECTED_ROOTS_PHONON = {
    0: (0, [1.65, 4.04, 4.50, 5.29, 6.14, 6.74, 7.16, 7.42, 7.55]),
    1: (0, [3.18, 4.80, 5.26, 5.78, 6.48, 6.63, 7.01, 7.35, 7.57]),
}

# Exiton parameters
NMAX_EXITON = 5
LMAX_EXITON = 1
V0_VAL = 1.5324e4 * 1e2  # [cm-1]
V0_COND = 2.0164e4 * 1e2  # [cm-1]
MASS_E_IN = .18
MASS_H_IN = .51
MASS_E_OUT = 1
MASS_H_OUT = 1
FREE_ELECTRON_MASS = 412.149e-7  # [cm/nm^2]
XDAT_X = np.linspace(1e-10, 15, 20000)
EXPECTED_ROOTS_ELECTRONS = {
    0: (0, [2.53943, 5.26517, 8.1282, 10.9340]),
    1: (0, [3.67152, 6.55451, 9.478265])
}
EXPECTED_ROOTS_HOLES = {
    0: (0, [2.88065, 5.7688, 8.6669894, 11.5682, 14.44034]),
    1: (0, [4.12112, 7.09644, 10.0319, 12.94567])
}

# Cavity parameters
OMEGA_CAV = (OMEGA_L - OMEGA_T) / 16
ANTENNA_L = 1.e2  # [nm]
ANTENNA_W = 1.e3  # [nm]
ANTENNA_H = 1.e3  # [nm]
COUPLING_E_CAV = 1.0e-4

# Raman parameters
E_GAP = 2.097e4 * 1e2  # [cm-1]
GAMMA_PH = 2
GAMMA_E = 8.e-4
GAMMA_H = 8.e-4
GAMMA_CAV = 4.e-2  # [nm^2]
# E_GAP = 0
# GAMMA_A = 8.06554  # [cm-1]
# GAMMA_B = GAMMA_A
# GAMMA_F = 3 * GAMMA_A
# GAMMA_A = 8e-4
# GAMMA_B = 8e-4
# GAMMA_F = 2.4

OMEGA_LASER = 1000  # [cm-1]
OMEGA_SEC = 250  # [cm-1]
POLAR_LASER = [1., 0., 0.]
POLAR_SEC = [1., 0., 0.]
KAPPA_LASER = [0., 0., 1]
KAPPA_SEC = [0., 0., 1]
REFRACTION_IDX_LASER = 1
REFRACTION_IDX_SEC = 1


# Get phonon space
phonon_space = PhononModelSpace(
    nmax=NMAX_PHONON, lmax=LMAX_PHONON,
    radius=RADIUS, beta_L=BETA_L, beta_T=BETA_T,
    omega_L=OMEGA_L, omega_T=OMEGA_T,
    epsilon_inf_qdot=EPS_INF_IN, epsilon_inf_env=EPS_INF_OUT,
    expected_roots_mu_l=EXPECTED_ROOTS_PHONON,
)

for s in phonon_space.states():
    print(s, end='\t')
    print(phonon_space.get_omega(s))

# fig, ax = plt.subplots(1, 1)
# for s in phonon_space.states():
#     func = phonon_space.Phi_ln(s)
#     xdat = np.linspace(0, 2 * phonon_space.R, 1000)
#     ydat = np.array([func(x) for x in xdat])
#     ax.plot(xdat, ydat, '-', label=phonon_space.get_nums(s))
# plt.legend()
# plt.show()

# Plot roots for phonon potential
# phonon_space.plot_root_function_mu(l=0, xdat=XDAT_MU, show=True)
# phonon_space.plot_root_function_mu(l=1, xdat=XDAT_MU, show=True)

# Get exiton space
exiton_space = ExitonModelSpace(
    nmax=NMAX_EXITON,
    lmax=LMAX_EXITON,
    radius=RADIUS,
    V_v=V0_VAL,
    V_c=V0_COND,
    E_gap=E_GAP,
    me_eff_in=MASS_E_IN,
    me_eff_out=MASS_E_OUT,
    mh_eff_in=MASS_H_IN,
    mh_eff_out=MASS_H_OUT,
    free_electron_mass=FREE_ELECTRON_MASS,
    expected_roots_x_elec=EXPECTED_ROOTS_ELECTRONS,
    expected_roots_x_hole=EXPECTED_ROOTS_HOLES,
)

# for s1, s2 in it.combinations_with_replacement(exiton_space.states(), r=2):
#     if (s1.l, s1.m) != (s2.l, s2.m):
#         continue
#
#     wfr1 = exiton_space.wavefunction_envelope_radial(s1)
#     wfr2 = exiton_space.wavefunction_envelope_radial(s2)
#     wfa1 = exiton_space.wavefunction_envelope_angular(s1)
#     wfa2 = exiton_space.wavefunction_envelope_angular(s2)
#
#     def intfn(r, theta, phi):
#         dV = np.sin(theta) * r**2
#         return np.conj(wfr1(r) * wfa1(theta, phi)) * (wfr2(r) * wfa2(theta, phi)) * dV
#     real_part = integ.nquad(lambda a, b, c: intfn(a, b, c).real,
#                             ranges=[(0, np.inf), (0, np.pi), (0, 2 * np.pi)])[0]
#     imag_part = integ.nquad(lambda a, b, c: intfn(a, b, c).imag,
#                             ranges=[(0, np.inf), (0, np.pi), (0, 2 * np.pi)])[0]
#     print('{}, \t{}'.format(s1, s2))
#     print('  {}'.format(real_part + 1j * imag_part))



# fig, ax = plt.subplots(1, 1)
# for s in exiton_space.states():
#     func = exiton_space.wavefunction_envelope_radial(state=s)
#     xdat = np.linspace(.00001 * phonon_space.R, 2 * phonon_space.R, 1000)
#     ydat = np.array([func(x) for x in xdat])
#     ax.plot(xdat, ydat, '-', label=exiton_space.get_nums(s))
# plt.legend()
# plt.show()
# fig, ax = plt.subplots(1, 1)
# for s in exiton_space.states():
#     func = exiton_space.wavefunction_envelope_radial(state=s)
#     xdat = np.linspace(.00001 * phonon_space.R, 2 * phonon_space.R, 1000)
#     ydat2 = np.array([func(x, 1) for x in xdat])
#     ax.plot(xdat, ydat2, '-', label=exiton_space.get_nums(s))
# plt.legend()
# plt.show()

# Plots roots: Electrons
# exiton_space.plot_root_fn_electrons(l=0, xdat=XDAT_X, show=True)
# exiton_space.plot_root_fn_electrons(l=1, xdat=XDAT_X, show=True)

# Plot roots: Holes
# exiton_space.plot_root_fn_holes(l=0, xdat=XDAT_X, show=True)
# exiton_space.plot_root_fn_holes(l=1, xdat=XDAT_X, show=True)


# Get cavity space
cavity_space = CavityModelSpace(
    l=ANTENNA_L, w=ANTENNA_W, h=ANTENNA_H,
    omega_c=OMEGA_CAV,
    g_e=COUPLING_E_CAV,
)

# Make system Hamiltonian
SAVENAME = 'savedham'
if not path.exists(SAVENAME):
    ham = RamanQD(
        phonon_space=phonon_space,
        exiton_space=exiton_space,
        cavity_space=cavity_space,
        phonon_lifetime=GAMMA_PH,
        electron_lifetime=GAMMA_E,
        hole_lifetime=GAMMA_H,
        cavity_lifetime=GAMMA_CAV,
    )
    print('Computing non-cavity matrix elements')
    ham.differential_raman_cross_section(
        omega_l=OMEGA_LASER, e_l=POLAR_LASER, n_l=REFRACTION_IDX_LASER,
        omega_s=OMEGA_SEC, e_s=POLAR_SEC, n_s=REFRACTION_IDX_SEC,
        include_cavity=False,
    )
    print('Computing cavity matrix elements')
    ham.differential_raman_cross_section(
        omega_l=OMEGA_LASER, e_l=POLAR_LASER, n_l=REFRACTION_IDX_LASER,
        omega_s=OMEGA_SEC, e_s=POLAR_SEC, n_s=REFRACTION_IDX_SEC,
        include_cavity=True,
    )
    print('Saving matrix elements locally')
    with open(SAVENAME, 'wb') as fwb:
        pickle.dump(ham, fwb)
else:
    print('Retrieving saved Hamiltonian')
    with open(SAVENAME, 'rb') as frb:
        ham = pickle.load(frb)

domega = OMEGA_L - OMEGA_T
# xdat = np.linspace(OMEGA_T-domega/10, OMEGA_L+domega/10, 100)
# xdat = np.linspace(0, OMEGA_LASER, 100)
# ydatr = np.zeros_like(xdat)
# ydati = np.zeros_like(xdat)

plt.ion()
fig, ax = plt.subplots(1, 1)
xdat = []
ydatr = []
# ydati = []
liner, = ax.plot(xdat, ydatr, '-', color='red')
# linei, = ax.plot(xdat, ydati, '-', color='blue')

# for i, x in enumerate(xdat):

todo = deque([(OMEGA_T-.5*domega, OMEGA_L+.5*domega)])
while True:
    lo, hi = todo.popleft()
    omega_ph = (lo+hi)/2
    todo.extend([(lo, omega_ph), (omega_ph, hi)])
    omega_s = OMEGA_LASER - omega_ph
    # print('Step {} of {}'.format(i+1, len(xdat)))
    print('  omega_s={}'.format(omega_s))
    eff = ham.differential_raman_cross_section(
        omega_l=OMEGA_LASER, e_l=POLAR_LASER, n_l=REFRACTION_IDX_LASER,
        omega_s=omega_s, e_s=POLAR_SEC, n_s=REFRACTION_IDX_SEC,
        include_cavity=True,
    )

    xdat.append(omega_ph)
    ydatr.append(eff.real)
    ydatr = [y for x, y in sorted(zip(xdat, ydatr))]
    xdat = sorted(xdat)

    print('  eff={}'.format(eff))
    liner.set_xdata(xdat)
    liner.set_ydata(ydatr)
    # linei.set_ydata(ydati)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1e-6)
    time.sleep(1e-6)
plt.ioff()
plt.show()
