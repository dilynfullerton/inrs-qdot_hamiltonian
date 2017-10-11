from collections import namedtuple
from HamiltonianQD import HamiltonianRoca
import numpy as np
import itertools as it
from matplotlib import pyplot as plt


ModelConstants = namedtuple('ModelConstants', ['epsilon_0', 'e', 'h'])


ham = HamiltonianRoca(
    R=99999,
    omega_TO=1/999999,
    l_max=1,
    epsilon_inf_qdot=1/9,
    epsilon_inf_env=1,
    constants=ModelConstants(epsilon_0=1, e=1, h=1),
    beta_L=1,
    beta_T=9,
)

r = ham.R

xdat = np.linspace(1e-7, 12e-7, 101)
ydatsl0 = dict()
ydatsl1 = dict()
for R, idx in zip(xdat, it.count()):
    ham = HamiltonianRoca(
        R=R,
        omega_TO=200,
        l_max=1,
        epsilon_inf_qdot=.3,
        epsilon_inf_env=1,
        constants=ModelConstants(epsilon_0=1, e=1, h=1),
        beta_T=1, beta_L=100000, large_R_approximation=True,
    )
    for omeg0, mode in zip(ham.eigenfrequencies(l=0, r=0), it.count()):
        if mode not in ydatsl0:
            ydatsl0[mode] = np.empty_like(xdat)
        ydatsl0[mode][idx] = omeg0
    for omeg1, mode in zip(ham.eigenfrequencies(l=1, r=0), it.count()):
        if mode not in ydatsl1:
            ydatsl1[mode] = np.empty_like(xdat)
        ydatsl1[mode][idx] = omeg1

fig, ax = plt.subplots(2, 1)
for mode, ydat in ydatsl0.items():
    ax[0].plot(xdat, ydat, '-', label='n={}'.format(mode))
for mode, ydat in ydatsl1.items():
    ax[1].plot(xdat, ydat, '-', label='n={}'.format(mode))
ax[0].legend()
ax[1].legend()
plt.show()
