from collections import namedtuple
from HamiltonianQD import HamiltonianRoca


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

for l in range(ham.l_max+1):
    nulist = []
    omlist = []
    for nu in ham._gen_nu(l)(r=r):
        nulist.append(nu)
        omlist.append(ham.omega(r=r, nu=nu))
    print('l = {}'.format(l))
    for nu, om in zip(nulist, omlist):
        print('  nu = {:8.4}'.format(nu), end='')
        print('  omega = {:16.8}'.format(om))
