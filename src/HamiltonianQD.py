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
            return _j(l, d)(z)
        else:
            return _i(l, d)(z)
    return fn_g


class HamiltonianRoca:
    def __init__(self, R, omega_TO, l_max, epsilon_inf_qdot, epsilon_inf_env,
                 constants, beta_T, beta_L, large_R_approximation=True):
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
                for nu_nl, n in zip(self._gen_nu(l)(r, theta, phi), it.count()):
                    h_mi += (
                        1 / nu_nl *
                        self._Phibar(nu_nl, l)(r) *
                        (self._b(n) + self._b(-n).dag())
                    )
                h_li += h_m * h_mi
            hi += h_l * h_li
        return h

    def _gen_nu(self, l):
        if self._large_R_approximation:
            return self._gen_nu_large_R(l)
        else:
            return self._gen_nu_full(l)

    def _gen_nu_abstract(self, fun0, jac0=None, ztr=None):
        def p(nu, zeros):
            p0 = 1
            for z in zeros:
                p0 *= (nu - z)
            return p0

        def dp(nu, zeros):
            zeros = list(zeros)
            if len(zeros) == 0:
                return 0
            else:
                z = zeros.pop()
                return dp(nu, zeros) * (nu - z) + p(nu, zeros)

        def fun(nu, *args):
            zeros = args[0]
            # nu = nu[0]
            fun_nu = fun0(nu) / p(nu, zeros=zeros)
            if ztr is not None:
                fun_nu = ztr(fun_nu)
            # fun_nu = complex(fun_nu)
            # return np.array([fun_nu.real, fun_nu.imag])
            return abs(fun_nu)

        def jac(nu, *args):
            zeros = args[0]
            # nu = nu[0]
            pk = p(nu, zeros=zeros)
            dpk = dp(nu, zeros=zeros)
            jac_nu = complex(jac0(nu) / pk - fun0(nu) * dpk / pk**2)
            return np.array([[jac_nu.real, jac_nu.imag],
                             [-jac_nu.imag, jac_nu.real]])

        all_zeros = list()
        x0 = np.array([1.])
        while True:
            # if jac0 is None:
            #     result = opt.root(fun=fun, jac=None, args=(all_zeros,), x0=x0)
            # else:
            #     result = opt.root(fun=fun, jac=jac, args=(all_zeros,), x0=x0)
            result = opt.minimize_scalar(
                fun=fun, bounds=(0., 20.), args=(all_zeros,)
            )
            # print(result.message)
            print('success = {}'.format(result.success))
            print('x = {}'.format(result.x))
            print('f(x) = {}'.format(result.fun))
            if not result.success:
                break
            # solution_nu = result.x[0]
            solution_nu = result.x
            yield solution_nu
            all_zeros.append(solution_nu)

    def _gen_nu_large_R(self, l):
        def fn_gen_nu_large_r(r, theta, phi):
            R = self.R
            q = self._q
            Q = self._Q

            def f1(nu):
                return _J(l, d=1)(q(nu=nu) * R)

            def f2(nu):
                return _g(l, d=1)(Q(r=r, nu=nu) * R)

            def f3(nu):
                Q0 = Q(r=r, nu=nu)
                return (
                    (self.omega2_L0(r=r) - self.omega2(r=r, nu=nu)) /
                    self._beta2_T(r=r) * l / Q0 + Q0 * self._lbar2(l)
                )

            def fprod(nu):
                return f1(nu) * f2(nu) * f3(nu)

            def ztr(z):
                return abs(z)

            # xdat = np.linspace(-10, 10, 101)
            # ydat = np.linspace(-1e-6, 1e-6, 101)
            # xdat, ydat = np.meshgrid(xdat, ydat)

            # zdat = np.empty_like(xdat)
            # for i, j in it.product(range(len(xdat)), repeat=2):
            #     zdat[i, j] = ztr(f1(xdat[i, j] + 1j*ydat[i, j]))
            # n = len(zdat)
            # fig, ax = plt.subplots(1, 1)
            # cs = ax.contourf(xdat, ydat, zdat, n)
            # fig.colorbar(cs)
            # plt.show()
            #
            # zdat = np.empty_like(xdat)
            # for i, j in it.product(range(len(xdat)), repeat=2):
            #     zdat[i, j] = ztr(f2(xdat[i, j] + 1j*ydat[i, j]))
            # n = len(zdat)
            # fig, ax = plt.subplots(1, 1)
            # cs = ax.contourf(xdat, ydat, zdat, n)
            # fig.colorbar(cs)
            # plt.show()
            #
            # zdat = np.empty_like(xdat)
            # for i, j in it.product(range(len(xdat)), repeat=2):
            #     zdat[i, j] = ztr(f3(xdat[i, j] + 1j*ydat[i, j]))
            # n = len(zdat)
            # fig, ax = plt.subplots(1, 1)
            # cs = ax.contourf(xdat, ydat, zdat, n)
            # fig.colorbar(cs)
            # plt.show()

            # zdat = np.empty_like(xdat)
            # for i, j in it.product(range(len(xdat)), repeat=2):
            #     zdat[i, j] = ztr(fprod(xdat[i, j] + 1j*ydat[i, j]))
            # n = len(zdat)
            # fig, ax = plt.subplots(1, 1)
            # cs = ax.contourf(xdat, ydat, zdat, n)
            # fig.colorbar(cs)
            # plt.show()

            xlen = 20000
            xmin = 0
            xmax = xlen
            xpts = 1000001

            xdat0 = np.linspace(xmin, xmax, xpts)
            ydat0 = np.array([f1(x) * f2(x) * f3(x) for x in xdat0])

            xdat = xdat0[~np.isnan(ydat0)]
            ydat = ydat0[~np.isnan(ydat0)]

            kdat = np.arange(xpts)[~np.isnan(ydat0)]
            ykdat_cplx = ft.fft(ydat)
            ykdat_real = np.real(ykdat_cplx)
            ykdat_imag = np.imag(ykdat_cplx)

            ykdat_real_fn = interp.interp1d(kdat, ykdat_real, kind='cubic')
            ykdat_imag_fn = interp.interp1d(kdat, ykdat_imag, kind='cubic')

            omegkdat = np.linspace(2*pi*kdat[0]/xlen, 2*pi*kdat[50]/xlen, xpts)
            yomegdat_real = ykdat_real_fn(omegkdat*xlen/2/pi)
            yomegdat_imag = ykdat_imag_fn(omegkdat*xlen/2/pi)

            fig, ax = plt.subplots(2, 1)
            ax[0].plot(xdat, ydat, '-', color='black')
            ax[1].plot(omegkdat, yomegdat_real, '-', color='red')
            ax[1].plot(omegkdat, yomegdat_imag, '-', color='blue')

            xmaxr, ymaxr = max(zip(omegkdat, yomegdat_real),
                               key=lambda x: abs(x[1]))
            xmaxi, ymaxi = max(zip(omegkdat, yomegdat_imag),
                               key=lambda x: abs(x[1]))

            lammaxr = 2 * pi / xmaxr
            lammaxi = 2 * pi / xmaxi
            print(lammaxr)
            print(lammaxi)
            for i in it.count():
                if lammaxi * i < xdat[-1]:
                    ax[0].axvline(lammaxi * i, color='blue', alpha=.5, lw=.5)
                if lammaxr * i < xdat[-1]:
                    ax[0].axvline(lammaxr * i, color='red', alpha=.5, lw=.5)
                if lammaxi*i > xdat[-1] and lammaxr*i > xdat[-1]:
                    break
            plt.show()

            omeg2 = self.omega2_TO(r=r) * (
                (self.constants.epsilon_0 * l + self.epsilon_inf2 * (l + 1)) /
                (self.epsilon_inf1 * l + self.epsilon_inf2 * (l + 1))
            )
            yield R * np.sqrt(
                (self.omega2_L0(r=r) - omeg2) / self._beta2_L(r=r)
            )

            # return map(lambda nu: nu.real, it.chain(
            #     self._gen_nu_abstract(fun0=f1, jac0=None, ztr=ztr),
            #     self._gen_nu_abstract(fun0=f2, jac0=None, ztr=ztr),
            #     self._gen_nu_abstract(fun0=f3, jac0=None, ztr=ztr),
            # ))
        return fn_gen_nu_large_r

    def _gen_nu_full(self, l):
        yield 0  # TODO

    def _lbar2(self, l):
        return l + (l + 1) * self._epsdiv

    def _gamma_0(self, r, dnu=0):
        if dnu > 0:
            return 0
        else:
            return (self.omega2_L0(r) - self.omega2_TO(r)) / self._beta2_T(r)

    def omega2_L0(self, r):
        return (self.omega2_TO(r) *
                self.constants.epsilon_0 / self._epsilon_inf(r=r))

    def omega2_TO(self, r):
        if callable(self._omega_TO_fn):
            return self._omega_TO_fn(r)**2
        else:
            return self._omega_TO_fn**2

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
            return self.omega2_L0(r=r) - self._beta2_L(r=r) * (nu / self.R)**2
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
            h * np.sqrt(self.omega2_L0(r=r)) / self.V_0 *
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
