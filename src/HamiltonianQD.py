"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definitions used in the theoretical model are based on
Sec. III of "Polar optical vibrations in quantum dots" by Roca et. al.
"""
from math import pi
import qutip as qt
import itertools as it
import numpy as np
from scipy import optimize as opt
from scipy import special as sp
from scipy import integrate as integ
from matplotlib import pyplot as plt


def _get_roots(rootfn, expected_roots):
    def fn(x):
        return rootfn(x[0])
    roots = []
    for x0 in expected_roots:
        result = opt.root(fun=fn, x0=x0)
        if not result.success:
            print('  FAIL: {}'.format(result.message))
        else:
            print('  SUCCESS: {}'.format(result.message))
            print('  root : x={}'.format(result.x[0]))
            roots.append(result.x[0])
    if len(roots) < len(expected_roots):
        print('  ERR: Found {} of {} expected roots'
              ''.format(len(roots), len(expected_roots)))
    return sorted(roots)


def _plot_dispersion_function(xdat, rootfn, iterfn):
    fig, ax = plt.subplots(1, 1)

    xdat = np.array(xdat)
    ydat = np.array([rootfn(nu=nu) for nu in xdat])
    ax.plot(xdat, ydat, '-', color='red')

    zxdat = np.array(iterfn)
    zydat = np.zeros_like(zxdat)
    ax.plot(zxdat, zydat, 'o', color='green')

    ax.axhline(0., ls='--', lw=1., color='gray', alpha=.5)
    ax.axvline(0., ls='--', lw=1., color='gray', alpha=.5)
    plt.show()
    return fig, ax


# Integrating over angles
def _Y(l, m):
    def fn_y(theta, phi):
        return sp.sph_harm(m, l, phi, theta)

    def intfn(theta, phi):
        return np.sin(theta) * fn_y(theta, phi)

    def ifnr(theta, phi):
        return np.real(intfn(theta, phi))
    re = integ.dblquad(func=ifnr, a=0, b=2*np.pi,
                       gfun=lambda x: 0, hfun=lambda x: np.pi)[0]

    def ifni(theta, phi):
        return np.imag(intfn(theta, phi))
    im = integ.dblquad(func=ifni, a=0, b=2*np.pi,
                       gfun=lambda x: 0, hfun=lambda x: np.pi)[0]

    return re + 1j * im


def _J(l, d=0):
    def fn_j(z):
        return sp.jvp(v=l, z=z, n=d)
    return fn_j


def _spherical_bessel1(l, func, d=0):
    def fn_spb(z):
        if d == 0:
            return func(n=l, z=z, derivative=False)
        elif d == 1:
            return func(n=l, z=z, derivative=True)
        elif l == 0:
            return -_spherical_bessel1(l=1, func=func, d=d - 1)(z)
        else:
            return (
                _spherical_bessel1(l=l-1, func=func, d=d - 1)(z) -
                (l+1) / 2 * _spherical_bessel1(l=l, func=func, d=d - 1)(z)
            )
    return fn_spb


def _j(l, d=0):
    return _spherical_bessel1(l=l, func=sp.spherical_jn, d=d)


def _i(l, d=0):
    return _spherical_bessel1(l=l, func=sp.spherical_in, d=d)


def _g(l, d=0):
    def fn_g(z):
        if z**2 >= 0:
            j = _j(l, d)(z)
            return j
        else:
            i = _i(l, d)(z)
            return i
    return fn_g


class ModelSpaceRoca:
    def __init__(self, nmax, nfock=2):
        self.nmax = nmax
        self.dim = 2 * self.nmax + 1
        self.nfock = nfock

    def vacuum(self):
        return qt.tensor([qt.fock_dm(self.nfock, 0)] * self.dim)

    def zero(self):
        return qt.qzero([self.nfock] * self.dim)

    def one(self):
        return qt.qeye([self.nfock] * self.dim)

    def b(self, n):
        assert -self.nmax <= n <= self.nmax
        if n < 0:
            n = self.dim + n
        ops = [qt.qeye(self.nfock)] * self.dim
        ops.insert(n, qt.destroy(self.nfock))
        ops.pop(n+1)
        return qt.tensor(ops)


class HamiltonianRoca:
    def __init__(self, R, omega_TO, l_max, n_max, epsilon_inf_qdot,
                 epsilon_inf_env, constants, beta_T, beta_L,
                 large_R_approximation=True, expected_nu_dict=None):
        self.ms = ModelSpaceRoca(nmax=n_max, nfock=2)
        self.R = R
        self.V_0 = 4/3 * pi * R**3
        self.l_max = l_max
        self.n_max = n_max
        self.epsilon_inf1 = epsilon_inf_qdot
        self.epsilon_inf2 = epsilon_inf_env
        self.constants = constants
        self._large_R_approximation = large_R_approximation
        self._epsdiv = self.epsilon_inf2 / self.epsilon_inf1
        self.beta_T = beta_T
        self.beta2_T = self.beta_T**2
        self.beta_L = beta_L
        self.beta2_L = self.beta_L**2

        # Frequencies of QD
        self.omega_TO = omega_TO
        self.omega2_TO = self.omega_TO**2
        self.omega2_LO = (self.omega2_TO * self.constants.epsilon_0 /
                          self.epsilon_inf1)
        self.omega_LO = np.sqrt(self.omega2_LO)

        self._nu_n_dict = dict()  # l -> [nu_0, nu_1, ..., nu_n]
        self._expected_nu_n_dict = dict()

        if expected_nu_dict is not None:
            for l, nulist in expected_nu_dict.items():
                self._expected_nu_n_dict[l] = sorted(list(nulist))

    # Making the assumption that spatial coordinates can be integrated over
    def h(self):
        """Electron-phonon interaction Hamiltonian at spherical coordinates
        (r, theta, phi). The definition is based on (Roca, Eq. 42).
        """
        h = self._C_F() * self.R * np.sqrt(2*pi)
        hi = 0
        for l in range(self.l_max+1):
            h_l = (2*l + 1) * 1j**l
            h_li = 0
            for m in range(-l, l+1):
                h_m = _Y(l, m)
                h_mi = 0
                for nu_nl, n in zip(self._iter_nu_n(l=l), range(self.n_max+1)):
                    h_mi += (
                        1 / nu_nl * self._Phibar(nu_nl, l) *
                        (self._b(n) + self._b(-n).dag())
                    )
                h_li += h_m * h_mi
            hi += h_l * h_li
        h *= hi
        return h + h.dag()

    def eigenfrequencies(self, l):
        nulist = self._iter_nu_n(l=l)
        omeglist = [self.omega(nu=nu) for nu in nulist]
        return sorted(omeglist)

    def _iter_nu_n(self, l):
        if l not in self._expected_nu_n_dict:
            return []
        elif l in self._nu_n_dict:
            return self._nu_n_dict[l]
        elif self._large_R_approximation:
            rootfn = self._root_function_nu_large_R(l=l)
        else:
            rootfn = self._root_function_nu_full(l=l)
        roots = _get_roots(
            rootfn=rootfn, expected_roots=self._expected_nu_n_dict[l])
        self._nu_n_dict[l] = sorted(roots)
        return roots

    def plot_dispersion_nu(self, l, xdat):
        if self._large_R_approximation:
            return _plot_dispersion_function(
                xdat=xdat, rootfn=self._root_function_nu_large_R(l=l),
                iterfn=self._iter_nu_n(l=l)
            )
        else:
            return _plot_dispersion_function(
                xdat=xdat, rootfn=self._root_function_nu_large_R(l=l),
                iterfn=self._iter_nu_n(l=l)
            )

    def _root_function_nu_large_R(self, l):
        def rootfn(nu):
            Q0 = self._Q(nu=nu)

            c1 = _J(l, d=1)(self._q(nu=nu) * self.R)
            c2 = _g(l, d=1)(Q0 * self.R)
            c3 = (
                (self.omega2_LO - self.omega2(nu=nu)) /
                self.beta2_T * l / Q0 + Q0 * self._lbar2(l)
            )
            return c1 * c2 * c3
        return rootfn

    def _root_function_nu_full(self, l):
        def rootfn(nu):
            return 0  # TODO

        return rootfn

    def _lbar2(self, l):
        return l + (l + 1) * self._epsdiv

    def _gamma_0(self, dnu=0):
        if dnu > 0:
            return 0
        else:
            return (self.omega2_LO - self.omega2_TO) / self.beta2_T

    def _nu(self, omega2):
        return self.R * np.sqrt(
            (self.omega2_LO - omega2) / self.beta2_L
        )

    def _mu(self, nu, dnu=0):
        return self._Q(nu=nu, dnu=dnu) * self.R

    def _Q(self, nu, dnu=0):
        bfrac = self.beta2_L / self.beta2_T
        q = self._q(nu=nu, dnu=dnu)
        gam = self._gamma_0(dnu=dnu)
        ans = bfrac * q - gam
        print(q)
        assert ans >= 0
        return ans

    def _q(self, nu, dnu=0):
        if dnu == 0:
            return nu / self.R
        elif dnu == 1:
            return 1 / self.R
        else:
            return 0

    def omega(self, nu, dnu=0):
        if dnu == 0:
            return np.sqrt(self.omega2(nu=nu))
        elif dnu == 1:
            return self.omega2(nu=nu, dnu=1) / 2 / self.omega(nu=nu)
        else:
            print('Why am I here?')  # TODO define this behavior
            raise RuntimeError

    def omega2(self, nu, dnu=0):
        if dnu == 0:
            om2 = self.omega2_LO - self.beta2_L * (nu / self.R)**2
            if not om2 >= 0:
                print('omega2_LO = {}'.format(self.omega2_LO))
                print('beta2_L = {}'.format(self.beta2_L))
                print('nu = {}'.format(nu))
                print('R = {}'.format(self.R))
                print('omega2 = {}'.format(om2))
            assert om2 >= 0
        elif dnu == 1:
            om2 = -2 * self.beta2_L * nu / self.R**2
        elif dnu == 2:
            om2 = -2 * self.beta2_L / self.R**2
        else:
            om2 = 0
        return om2

    def _C_F(self):
        e = self.constants.e
        h = self.constants.h
        eps0 = self.constants.epsilon_0
        return (
            e * np.sqrt(h / self.V_0 * self.omega_LO) *
            np.sqrt(1 / self.epsilon_inf1 - 1/eps0)
        )

    def _Phibar(self, nu, l):
        def _fn_phibar(r):
            rr = r / self.R
            if r <= self.R:
                return (
                    (nu * _j(l, d=1)(nu) + (l + 1) * self._epsdiv * _j(l)(nu)) *
                    rr**l - self._lbar2(l) * _j(l, d=1)(nu * rr)
                )
            else:
                return (
                    (nu * _j(l, d=1)(nu) - l * _j(l)(nu)) * rr**(-l-1)
                )
        return integ.quad(func=_fn_phibar, a=0, b=self.R)[0]

    def _b(self, n):
        return self.ms.b(n=n)
