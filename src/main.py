from collections import namedtuple
from HamiltonianQD import HamiltonianEPI
import numpy as np
import itertools as it
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from scipy import interpolate as interp


ModelConstants = namedtuple('ModelConstants', ['epsilon_0', 'e', 'hbar'])

R0 = 1.
R1 = 10 * R0
OMEGA_LO = 305
OMEGA_TO = 238
# EPS_INF_QD = 5.3
# EPS_INF_ENV = 4.64
EPS_INF_QD = .53
EPS_INF_ENV = .464
EPS_0_QD = OMEGA_LO**2 / OMEGA_TO**2 * EPS_INF_QD
BETA_L = 5.04e+1
BETA_T = 1.58e+1
KNOWN_MU = {
    # 0: [0.],
    # 1: [0.],
}

EXP_MU_NORMAL = {
    0: [
        +1.40, +2.40, +3.40, +4.39, +4.49, +5.40, +6.40, +7.40, +7.70, +8.40,
        +9.40,
    ],
    1: [
        +1.60, +2.25, +2.85, +3.90, +4.85, +5.81, +5.99, +6.90, +7.90, +8.90,
        +9.20,
    ],
}
EXP_MU_LARGE_R = {
    0: [4.5, 7.7, 10.9, 14.1, 17.2],
    1: [5.9, 9.2, 12.4, 15.6],
}
EXP_PERIODIC_START_LARGE_R = {
    0: 4.5,
    1: 5.9,
}
EXP_ROOT_PERIOD_LARGE_R = {
    0: 3.25,
    1: 3.25,
}
MU_ROUND = 6

LMAX = 1
NMODES = 3
NMAX = NMODES - 1
LARGE_R = True
if LARGE_R:
    EXP_MU = EXP_MU_LARGE_R
else:
    EXP_MU = EXP_MU_NORMAL

ham = HamiltonianEPI(
    r_0=R0,
    omega_LO=OMEGA_LO,
    l_max=LMAX,
    max_n_modes=NMODES,
    epsilon_inf_qdot=EPS_INF_QD,
    epsilon_inf_env=EPS_INF_ENV,
    epsilon_0_qdot=EPS_0_QD,
    constants=ModelConstants(epsilon_0=1, e=1, hbar=1),
    beta_L=BETA_L,
    beta_T=BETA_T,
    expected_mu_dict=EXP_MU,
    known_mu_dict=KNOWN_MU,
    mu_round=MU_ROUND,
    large_R_approximation=LARGE_R,
    verbose_roots=True,
    periodic_roots=True,
    periodic_roots_start_dict=EXP_PERIODIC_START_LARGE_R,
    expected_root_period_dict=EXP_ROOT_PERIOD_LARGE_R,
)

XDAT = np.linspace(-5, 20, 1000)
ham.plot_root_function_mu(l=0, xdat=XDAT)
ham.plot_root_function_mu(l=1, xdat=XDAT)

expect_mu = dict(EXP_MU)
expect_period = dict(EXP_ROOT_PERIOD_LARGE_R)
rdat = np.linspace(R0, R1, 101)
wdats_l0 = [[] for i in range(NMAX + 1)]
wdats_l1 = [[] for i in range(NMAX + 1)]
for R, i in zip(rdat, it.count()):
    print('R = {:8.4e}'.format(R))
    ham = HamiltonianEPI(
        r_0=R,
        omega_LO=OMEGA_LO,
        l_max=LMAX,
        max_n_modes=NMODES,
        epsilon_inf_qdot=EPS_INF_QD,
        epsilon_inf_env=EPS_INF_ENV,
        epsilon_0_qdot=EPS_0_QD,
        constants=ModelConstants(epsilon_0=1, e=1, hbar=1),
        beta_L=BETA_L,
        beta_T=BETA_T,
        expected_mu_dict=expect_mu,
        known_mu_dict=KNOWN_MU,
        mu_round=MU_ROUND,
        large_R_approximation=LARGE_R,
        verbose_roots=True,
        periodic_roots=True,
        periodic_roots_start_dict=EXP_PERIODIC_START_LARGE_R,
        expected_root_period_dict=expect_period,
    )
    print()
    # if i % 200 == 0:
        # ham.plot_root_function_mu(l=0, xdat=XDAT)
        # ham.plot_root_function_mu(l=1, xdat=XDAT)
    for omega_n, n in ham.iter_omega_n(l=0):
        wdats_l0[n].append(omega_n)
    for omega_n, n in ham.iter_omega_n(l=1):
        wdats_l1[n].append(omega_n)
    # expect_period = {
    #     0: ham.mu_nl(n=1, l=0) - ham.mu_nl(n=0, l=0),
    #     1: ham.mu_nl(n=1, l=1) - ham.mu_nl(n=1, l=1),
    # }
    # expect_mu = {
    #     0: [nu for nu, n in ham.iter_mu_n(l=0)],
    #     1: [nu for nu, n in ham.iter_mu_n(l=1)],
    # }

print('Making plots...')

norm = colors.Normalize(vmin=0, vmax=len(wdats_l0)-1)
sm = cm.ScalarMappable(norm=norm)

fig, axes = plt.subplots(2, 1)
axes[0].set_ylabel('l = 0')
axes[1].set_ylabel('l = 1')
axes[0].axhline(ham.omega_T, ls='--', color='gray', alpha=.5)
axes[1].axhline(ham.omega_T, ls='--', color='gray', alpha=.5)
axes[0].axhline(ham.omega_L, ls='--', color='gray', alpha=.5)
axes[1].axhline(ham.omega_L, ls='--', color='gray', alpha=.5)
axes[0].axhline(ham.omega_F, ls='--', color='gray', alpha=.5)
axes[1].axhline(ham.omega_F, ls='--', color='gray', alpha=.5)

for wdat, mode in zip(wdats_l0, it.count()):
    axes[0].plot(rdat, np.real(wdat), '-',
                 label='n={:.2e}'.format(mode),
                 color=sm.to_rgba(mode))
    # axes[0].plot(rdat, np.imag(wdat), '-.',
    #              label='mu={}, Im'.format(ham.mu_nl(n=mode, l=0)),
    #              color=sm.to_rgba(mode))

for wdat, mode in zip(wdats_l1, it.count()):
    axes[1].plot(rdat, np.real(wdat), '-',
                 label='={:.2e}'.format(mode),
                 color=sm.to_rgba(mode))
    # axes[1].plot(rdat, np.imag(wdat), '-.',
    #              label='mu={}, Im'.format(ham.mu_nl(n=mode, l=1)),
    #              color=sm.to_rgba(mode))

plt.legend()

plt.show()
