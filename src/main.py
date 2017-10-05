from collections import namedtuple
from HamiltonianQD import HamiltonianRoca


ModelConstants = namedtuple('ModelConstants', ['epsilon_0', 'e', 'h'])


ham = HamiltonianRoca(
    R=10,
    omega_TO=36,
    l_max=0,
    epsilon_inf_qdot=8.9,
    epsilon_inf_env=1,
    constants=ModelConstants(epsilon_0=1, e=1, h=1),
    beta_L=1,
    beta_T=1,
)

r = ham.R/2

for l in range(ham.l_max+1):
    print('l = {}'.format(l))
    for nu in ham._gen_nu(l)(r=r, theta=0, phi=0):
        print('  nu = {:8.4}'.format(nu), end='')
        print('  omega = {:16.8}'.format(ham.omega(r=r, nu=nu)))
