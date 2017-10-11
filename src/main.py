from collections import namedtuple
from HamiltonianQD import HamiltonianEPI
import numpy as np
import itertools as it
from matplotlib import pyplot as plt
from matplotlib import cm, colors


from HamiltonianQD import _j, _i
L = 1
xdat = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(xdat, xdat)
Z = np.empty(shape=X.shape, dtype=complex)
for i, j in it.product(range(len(xdat)), repeat=2):
    Z[i, j] = _j(l=L, d=1)(X[i, j] + 1j * Y[i, j])
fig, ax = plt.subplots(1, 1)
ax.contourf(X, Y, np.real(Z), len(xdat))
plt.show()
fig, ax = plt.subplots(1, 1)
ax.contourf(X, Y, np.imag(Z), len(xdat))
plt.show()


jdat0 = _j(l=L, d=0)(xdat)
jdat1 = _j(l=L, d=1)(xdat)
idat0 = _i(l=L, d=0)(1j*xdat)
idat1 = _i(l=L, d=1)(1j*xdat)
fig, ax = plt.subplots(1, 1)
ax.plot(xdat, jdat0, '-', color='red')
ax.plot(xdat, jdat1, '--', color='red')
# ax.plot(xdat, idat0, '-', color='blue')
# ax.plot(xdat, idat1, '--', color='blue')
plt.show()
assert False


ModelConstants = namedtuple('ModelConstants', ['epsilon_0', 'e', 'hbar'])

R0 = 1e-7
R1 = 6e-7
OMEGA_LO = 305
OMEGA_TO = 238
EPS_INF_QD = 5.3
EPS_INF_ENV = 4.64
EPS_0_QD = OMEGA_LO**2 / OMEGA_TO**2 * EPS_INF_QD
BETA_L = 5.04e3
BETA_T = 1.58e3
KNOWN_MU = None
EXP_MU = {
    0: [
        1.4, 2.4, 3.4, 4.375, 4.5],
    1: [
        1.6, 2.2, 2.8, 3.9],
}
LMAX = 1
NMAX = 3

ham = HamiltonianEPI(
    r_0=R0,
    omega_LO=OMEGA_LO,
    l_max=LMAX,
    n_max=NMAX,
    epsilon_inf_qdot=EPS_INF_QD,
    epsilon_inf_env=EPS_INF_ENV,
    epsilon_0_qdot=EPS_0_QD,
    constants=ModelConstants(epsilon_0=1, e=1, hbar=1),
    beta_L=BETA_L,
    beta_T=BETA_T,
    expected_mu_dict=EXP_MU,
    known_mu_dict=KNOWN_MU,
    large_R_approximation=True,
)

XDAT = np.linspace(-10, 10, 1000)
ham.plot_root_function_mu(l=0, xdat=XDAT)
ham.plot_root_function_mu(l=1, xdat=XDAT)

expect_mu = dict(EXP_MU)
rdat = np.linspace(R0, R1, 101)
wdats_l0 = [np.empty(shape=rdat.shape, dtype=complex) for i in range(NMAX + 1)]
wdats_l1 = [np.empty(shape=rdat.shape, dtype=complex) for i in range(NMAX + 1)]
for R, i in zip(rdat, it.count()):
    print('R = {:8.4e}'.format(R))
    ham = HamiltonianEPI(
        r_0=R,
        omega_LO=OMEGA_LO,
        l_max=LMAX,
        n_max=NMAX,
        epsilon_inf_qdot=EPS_INF_QD,
        epsilon_inf_env=EPS_INF_ENV,
        epsilon_0_qdot=EPS_0_QD,
        constants=ModelConstants(epsilon_0=1, e=1, hbar=1),
        beta_L=BETA_L,
        beta_T=BETA_T,
        expected_mu_dict=expect_mu,
        known_mu_dict=KNOWN_MU,
        large_R_approximation=True,
    )
    print()
    for omega_n, n in ham.iter_omega_n(l=0):
        wdats_l0[n][i] = omega_n
    for omega_n, n in ham.iter_omega_n(l=1):
        wdats_l1[n][i] = omega_n
    expect_mu = {
        0: [nu for nu, n in ham.iter_mu_n(l=0)],
        1: [nu for nu, n in ham.iter_mu_n(l=1)],
    }

print('Making plots...')

norm = colors.Normalize(vmin=0, vmax=len(wdats_l0)-1)
sm = cm.ScalarMappable(norm=norm)

fig, axes = plt.subplots(2, 1)
for wdat, mode in zip(wdats_l0, it.count()):
    axes[0].plot(rdat, wdat.real, '-',
                 label='n={}, Re'.format(mode), color=sm.to_rgba(mode))
    axes[0].plot(rdat, wdat.imag, '-.',
                 label='n={}, Im'.format(mode), color=sm.to_rgba(mode))
axes[0].set_ylabel('l = 0')
axes[0].legend()

for wdat, mode in zip(wdats_l1, it.count()):
    axes[1].plot(rdat, wdat.real, '-',
                 label='n={}, Re'.format(mode), color=sm.to_rgba(mode))
    axes[1].plot(rdat, wdat.imag, '-.',
                 label='n={}, Im'.format(mode), color=sm.to_rgba(mode))
axes[1].set_ylabel('l = 1')
axes[0].legend()

plt.show()
