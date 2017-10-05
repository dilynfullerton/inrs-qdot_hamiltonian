"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definitions used in the theoretical model are based on
Sec. III of "Polar optical vibrations in quantum dots" by Roca et. al.
"""
from math import sqrt, pi
import qutip as qt
import itertools as it
import numpy as np
import scipy.optimize as opt
import scipy.special as sp


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
        h = self._C_F(r) * self.R * sqrt(2*pi)
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

    def _gen_nu_abstract(self, fun0, jac0):
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
            nu = nu[0] + 1j*nu[1]
            fun_nu = complex(fun0(nu) / p(nu, zeros=zeros))
            return np.array([fun_nu.real, fun_nu.imag])

        def jac(nu, *args):
            zeros = args[0]
            nu = nu[0] + 1j*nu[1]
            pk = p(nu, zeros=zeros)
            dpk = dp(nu, zeros=zeros)
            jac_nu = complex(jac0(nu) / pk - fun0(nu) * dpk / pk**2)
            return np.array([[jac_nu.real, jac_nu.imag],
                             [-jac_nu.imag, jac_nu.real]])

        zsf = []
        x0 = np.array([1., 1.])
        print(opt.check_grad(fun, jac, x0, []))
        assert False
        while True:
            if jac0 is None:
                result = opt.root(fun=fun, jac=None, args=(zsf,), x0=x0)
            else:
                result = opt.root(fun=fun, jac=jac, args=(zsf,), x0=x0)
            if not result.success:
                break
            solution_nu = result.x[0] + 1j*result.x[1]
            yield solution_nu
            zsf.append(solution_nu)

    def _gen_nu_large_R(self, l):
        def fn_gen_nu_large_r(r, theta, phi):
            def fun0(nu):
                Q = self._Q(r=r, nu=nu)
                return _J(l, d=1)(nu) * _g(l, d=1)(self._mu(r=r, nu=nu)) * (
                    self._gamma_0(r) * l / Q + Q * self._lbar2(l)
                )

            def jac0(nu):
                mu = self._mu(r=r, nu=nu)
                dmu = self._mu(r=r, nu=nu, dnu=1)
                Q = self._Q(r=r, nu=nu)
                dQ = self._Q(r=r, nu=nu, dnu=1)
                gam0 = self._gamma_0(r)
                dgam0 = self._gamma_0(r=r, d=1)
                ll = self._lbar2(l)
                dj = _J(l=l, d=1)(nu)
                ddj = _J(l=l, d=2)(nu)
                dg = _g(l=l, d=1)(mu)
                ddg = dmu * _g(l=l, d=2)(mu)
                gam = (gam0 * l/Q + Q * ll)
                dgam = l * (dgam0/Q-gam0*dQ/Q**2) + dQ * ll
                return ddj * dg * gam + dj * ddg * gam + dj * dg * dgam

            return self._gen_nu_abstract(fun0=fun0, jac0=jac0)
            # return self._gen_nu_abstract(fun0=fun0, jac0=None)
        return fn_gen_nu_large_r

    def _gen_nu_full(self, l):
        yield 0  # TODO

    def _lbar2(self, l):
        return l + (l + 1) * self._epsdiv

    def _gamma_0(self, r, d=0):
        if d > 0:
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

    def _Q(self, r, nu, dnu=0, domega=0):
        if domega == 0 and dnu == 0:
            return sqrt(
                (self.omega2_TO(r=r) - self.omega2(r=r, nu=nu)) /
                self._beta2_T(r=r)
            )
        elif dnu == 0 and domega == 1:
            return -self.omega(r=r, nu=nu) / self._Q(r=r, nu=nu)
        elif dnu == 1 and domega == 0:
            return self._Q(r=r, nu=nu, domega=1) * self.omega(r=r, nu=nu, dnu=1)
        else:
            print('Why am I here?')  # TODO define this behavior
            raise RuntimeError

    def omega(self, r, nu, dnu=0):
        if dnu == 0:
            return sqrt(self.omega2(r=r, nu=nu))
        elif dnu == 1:
            return self.omega2(r=r, nu=nu, dnu=1) / 2 / self.omega(r=r, nu=nu)
        else:
            print('Why am I here?')  # TODO define this behavior
            raise RuntimeError

    def omega2(self, r, nu, dnu=0):
        if dnu == 0:
            return self.omega2_L0(r=r) - self._beta2_L(r=r) * nu**2
        elif dnu == 1:
            return -2 * self._beta2_L(r=r) * nu
        elif dnu == 2:
            return -2 * self._beta2_L(r=r)
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
        return e * sqrt(
            h * sqrt(self.omega2_L0(r=r)) / self.V_0 *
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
