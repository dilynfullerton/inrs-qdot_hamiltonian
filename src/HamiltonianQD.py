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
from scipy import fftpack as ft
from scipy import linalg as lin
from scipy import interpolate as interp
import warnings
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import time as tm

warnings.filterwarnings('error')


def _Y(l, m):
    def fn_y(theta, phi):
        return sp.sph_harm(m, l, phi, theta)
    return fn_y


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


class HamiltonianRoca:
    def __init__(self, R, omega_TO, l_max, epsilon_inf_qdot, epsilon_inf_env,
                 constants, beta_T, beta_L, large_R_approximation=True,
                 omega_min=None, omega_max=None, omega_resolution=1001):
        self.R = R
        self.V_0 = 4/3 * pi * R**3
        self.l_max = l_max
        self.epsilon_inf1 = epsilon_inf_qdot
        self.epsilon_inf2 = epsilon_inf_env
        self.constants = constants
        self._large_R_approximation = large_R_approximation
        self._epsdiv = self.epsilon_inf2 / self.epsilon_inf1
        self._beta_T_fn = beta_T
        self._beta_L_fn = beta_L
        self._omega_TO_fn = omega_TO
        self._omega_res = omega_resolution
        self._omega_min_fn = omega_min
        self._omega_max_fn = omega_max

    def h(self, r, theta, phi):
        """Electron-phonon interaction Hamiltonian at spherical coordinates
        (r, theta, phi). The definition is based on (Roca, Eq. 42).
        """
        h = self._C_F(r) * self.R * np.sqrt(2*pi)
        hi = 0
        for l in range(self.l_max+1):
            h_l = (2*l + 1) * 1j**l
            h_li = 0
            for m in range(-l, l+1):
                h_m = _Y(l, m)(theta, phi)
                h_mi = 0
                for nu_nl, n in zip(self._gen_nu(l)(r), it.count()):
                    h_mi += (
                        1 / nu_nl *
                        self._Phibar(nu_nl, l)(r) *
                        (self._b(n) + self._b(-n).dag())
                    )
                h_li += h_m * h_mi
            hi += h_l * h_li
        return h

    def eigenfrequencies(self, l, r):
        nulist = self._gen_nu(l=l)(r=r)
        # TODO: should probably not have to cast to real
        omeglist = [abs(self.omega(r=r, nu=nu)) for nu in nulist]
        return sorted(omeglist)

    def _gen_nu(self, l):
        if self._large_R_approximation:
            return self._gen_nu_large_R(l)
        else:
            return self._gen_nu_full(l)

    def _nu_min(self, r):
        return self._nu(omega2=self._omega2_max(r=r), r=r)

    def _nu_max(self, r):
        return self._nu(omega2=self._omega2_min(r=r), r=r)

    def _nu_res(self):
        return self._omega_res

    def _gen_nu_large_R(self, l):
        def fn_gen_nu_large_r(r):
            R = self.R
            q = self._q
            Q = self._Q

            def f1(nu):
                return _J(l, d=1)(q(nu=nu) * R)

            def f2(nu):
                Q0 = Q(r=r, nu=nu)
                return _g(l, d=1)(Q0 * R)

            def f3(nu):
                Q0 = Q(r=r, nu=nu)
                return (
                    (self.omega2_LO(r=r) - self.omega2(r=r, nu=nu)) /
                    self._beta2_T(r=r) * l / Q0 + Q0 * self._lbar2(l)
                )

            def fprod(nu):
                c1 = f1(nu)
                c2 = f2(nu)
                c3 = f3(nu)
                return c1 * c2 * c3

            return 0  # TODO
        return fn_gen_nu_large_r

    def _gen_nu_full(self, l):
        yield 0  # TODO

    def _lbar2(self, l):
        return l + (l + 1) * self._epsdiv

    def _gamma_0(self, r, dnu=0):
        if dnu > 0:
            return 0
        else:
            return (self.omega2_LO(r) - self.omega2_TO(r)) / self._beta2_T(r)

    def omega2_LO(self, r):
        return (self.omega2_TO(r) *
                self.constants.epsilon_0 / self._epsilon_inf(r=r))

    def omega2_TO(self, r):
        if callable(self._omega_TO_fn):
            return self._omega_TO_fn(r)**2
        else:
            return self._omega_TO_fn**2

    def _omega2_min(self, r):
        if self._omega_min_fn is None:
            omega2to = self.omega2_TO(r=r)
            return omega2to + (self.omega2_LO(r=r)-omega2to)/self._omega_res
        elif callable(self._omega_min_fn):
            return self._omega_min_fn(r=r)**2
        else:
            return self._omega_min_fn**2

    def _omega2_max(self, r):
        if self._omega_max_fn is None:
            omega2lo = self.omega2_LO(r=r)
            return omega2lo - (omega2lo-self.omega2_TO(r=r))/self._omega_res
        elif callable(self._omega_max_fn):
            return self._omega_max_fn(r=r)**2
        else:
            return self._omega_max_fn**2

    def _nu(self, omega2, r):
        return self.R * np.sqrt(
            (self.omega2_LO(r=r) - omega2) / self._beta2_L(r=r)
        )

    def _mu(self, r, nu, dnu=0):
        return self._Q(r=r, nu=nu, dnu=dnu) * self.R

    def _Q(self, r, nu, dnu=0):
        return (
            self._beta2_L(r=r)/self._beta2_T(r=r) *
            self._q(nu=nu, dnu=dnu) - self._gamma_0(r=r, dnu=dnu)
        )

    def _q(self, nu, dnu=0):
        if dnu == 0:
            return nu / self.R
        elif dnu == 1:
            return 1 / self.R
        else:
            return 0

    def omega(self, r, nu, dnu=0):
        if dnu == 0:
            return np.sqrt(complex(self.omega2(r=r, nu=nu)))
        elif dnu == 1:
            return self.omega2(r=r, nu=nu, dnu=1) / 2 / self.omega(r=r, nu=nu)
        else:
            print('Why am I here?')  # TODO define this behavior
            raise RuntimeError

    def omega2(self, r, nu, dnu=0):
        if dnu == 0:
            return self.omega2_LO(r=r) - self._beta2_L(r=r) * (nu / self.R)**2
        elif dnu == 1:
            return -2 * self._beta2_L(r=r) * nu / self.R**2
        elif dnu == 2:
            return -2 * self._beta2_L(r=r) / self.R**2
        else:
            return 0

    def _beta2_T(self, r):
        return self._beta_T_fn**2

    def _beta2_L(self, r):
        return self._beta_L_fn**2

    def _C_F(self, r):
        e = self.constants.e
        h = self.constants.h
        eps0 = self.constants.epsilon_0
        return e * np.sqrt(
            h * np.sqrt(self.omega2_LO(r=r)) / self.V_0 *
            (1/self._epsilon_inf(r=r) - 1/eps0)
        )

    def _epsilon_inf(self, r):
        if r <= self.R:
            return self.epsilon_inf1
        else:
            return self.epsilon_inf2

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
        return _fn_phibar

    def _b(self, n):
        return qt.qzero(2)  # TODO
