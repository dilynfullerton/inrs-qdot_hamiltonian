from collections import namedtuple
from HamiltonianQD import HamiltonianRoca
import numpy as np
import itertools as it
from matplotlib import pyplot as plt


ModelConstants = namedtuple('ModelConstants', ['epsilon_0', 'e', 'h'])

R0 = 1
OMEGA_TO = 4
EPS_INF = 1/2
BETA_L = 1
BETA_T = 9
EXP_NU = {
    0: [.5, 4., 7., 10.],
    1: [-0., 2., 5.5, 8.5, 12.],
}
EXP_NU = {}

ham = HamiltonianRoca(
    R=R0,
    omega_TO=OMEGA_TO,
    l_max=1,
    n_max=3,
    epsilon_inf_qdot=EPS_INF,
    epsilon_inf_env=1,
    constants=ModelConstants(epsilon_0=1, e=1, h=1),
    beta_L=BETA_L,
    beta_T=BETA_T,
    expected_nu_dict=EXP_NU,
    large_R_approximation=True,
)

ham.plot_dispersion_nu(l=0, xdat=np.linspace(0, 5 * R0, 1001))
ham.plot_dispersion_nu(l=1, xdat=np.linspace(0, 5 * R0, 1001))

print(ham.h())

nmax = 3
expect_nu = dict(EXP_NU)
rdat = np.linspace(1, 6, 101)
wdats_l0 = [np.empty_like(rdat)] * (nmax + 1)
wdats_l1 = [np.empty_like(rdat)] * (nmax + 1)
for R, i in zip(rdat, it.count()):
    print('R = {:8.4e}'.format(R))
    ham = HamiltonianRoca(
        R=R,
        omega_TO=OMEGA_TO,
        l_max=1,
        n_max=nmax,
        epsilon_inf_qdot=EPS_INF,
        epsilon_inf_env=1,
        constants=ModelConstants(epsilon_0=1, e=1, h=1),
        beta_L=BETA_L,
        beta_T=BETA_T,
        expected_nu_dict=expect_nu,
        large_R_approximation=True,
    )
    for omega_n, n in zip(ham.eigenfrequencies(l=0), range(nmax+1)):
        wdats_l0[n][i] = omega_n
    for omega_n, n in zip(ham.eigenfrequencies(l=1), range(nmax+1)):
        wdats_l1[n][i] = omega_n
    expect_nu = {
        0: [nu for nu, n in zip(ham._iter_nu_n(l=0), range(nmax+1))],
        1: [nu for nu, n in zip(ham._iter_nu_n(l=1), range(nmax+1))],
    }

print('Making plots...')

fig, axes = plt.subplots(2, 1)
for wdat, mode in zip(wdats_l0, it.count()):
    axes[0].plot(rdat, wdat, '-', label='n={}'.format(mode))
axes[0].set_ylabel('l = 0')
axes[0].legend()

for wdat, mode in zip(wdats_l1, it.count()):
    axes[1].plot(rdat, wdat, '-', label='n={}'.format(mode))
axes[1].set_ylabel('l = 1')
axes[0].legend()

plt.show()
