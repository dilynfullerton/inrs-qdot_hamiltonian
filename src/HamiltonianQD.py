"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definitions used in the theoretical model are based on

Electron Raman Scattering in Nanostructures
R. Betancourt-Riera, R. Riera, J. L. Marin
Centro de Investigacion en Fisica, Universidad de Sonora, 83190 Hermosillo, Sonora, Mexico
R. Rosas
Departamento de Fisica, Universidad de Sonora, 83000 Hermosillo, Sonora, Mexico
"""
import itertools as it
from math import pi

import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from scipy import integrate as integ
from scipy import linalg as lin
from helper_functions import Y_lm
from helper_functions import j_sph as _j
from helper_functions import g_sph as _g
from root_solver_2d import RootSolverComplex2d


class HamiltonianQD(RootSolverComplex2d):
    def __init__(self, r_0, omega_L, omega_T, num_n, num_l,
                 epsilon_inf_qdot, epsilon_inf_env, beta_T, beta_L,
                 expected_roots_mu_l, electron_charge=1,
                 large_R_approximation=False, verbose=False):
        # Model constants
        self.num_n = num_n
        self.num_l = num_l
        self.n_max = self.num_n - 1
        self.l_max = self.num_l - 1

        # Physical constants
        self.r_0 = r_0
        self.eps_a_inf = epsilon_inf_qdot
        self.eps_b_inf = epsilon_inf_env
        self.beta_L2 = beta_L**2
        self.beta_T2 = beta_T**2
        self.omega_L = omega_L
        self.omega_T = omega_T

        # Derived physical constants
        self.V = 4/3 * pi * self.r_0**3
        self._eps_div = self.eps_b_inf / self.eps_a_inf
        self._beta_div2 = self.beta_L2 / self.beta_T2
        self.omega_L2 = self.omega_L**2
        self.omega_T2 = self.omega_T**2
        self.eps_a_0 = self.eps_a_inf * self.omega_L2 / self.omega_T2
        self.omega_F2 = (
            self.omega_T2 * (self.eps_a_0 + 2 * self.eps_b_inf) /
            (self.eps_a_inf + 2 * self.eps_b_inf)
        )
        self.omega_F = np.sqrt(self.omega_F2)
        self.electron_charge = abs(electron_charge)
        # NOTE: The following definition for gamma is different from gamma_0
        # of Riera
        self.gamma = (self.omega_L2 - self.omega_T2) / self.beta_T2
        self.C_F = -self.electron_charge * np.sqrt(  # Riera (45)
            2 * pi * self.omega_L /
            self.V * (1/self.eps_a_inf - 1/self.eps_a_0)
        )
        self.low_mu = self.r_0 / np.sqrt(self._beta_div2) * np.sqrt(self.gamma)

        # Root finding
        self._roots_mu = dict()  # l, n -> mu
        self._roots_nu = dict()  # l, n -> nu
        self._expected_roots_mu = expected_roots_mu_l  # l -> mu0
        self._large_R = large_R_approximation
        self._fill_roots_mu_nu()

        # Other storage dictionaries
        self._u2 = dict()  # l, n -> abs(u)**2

        self._verbose = verbose

    # -- Hamiltonian --
    def H_epi(self, r, theta, phi):
        """Electron-phonon interaction Hamiltonian at spherical coordinates
        (r, theta, phi). The definition is based on Roca (42) and
        Riera (60).
        """
        h = 0
        for l, m, n in self.iter_lmn():
            h += (
                self._H_ep_rad_ln(l=l, n=n)(r) * Y_lm(l=l, m=m)(theta, phi) *
                self._b(l=l, m=m, n=n)
            )
        return 1/2 * (h + h.dag())

    def _H_ep_rad_ln(self, l, n):
        def hrfn(r):
            return (
                self.r_0 * self.C_F * np.sqrt(4*np.pi/3) *
                self.Phi_ln(l=l, n=n)(r)
            )
        return hrfn

    def Phi_ln(self, l, n):
        """See Roca (43) and Riera (60)
        The returned result should be real.
        """
        mu_n = self.mu_nl(n=n, l=l)
        if mu_n is None:
            return lambda r: 0  # TODO find a better way
        ediv = self._eps_div
        dj_mu = _j(l, d=1)(mu_n)
        j_mu = _j(l)(mu_n)
        abs_u = self._norm_u(n=n, l=l)

        def phifn(r):
            r0 = self.r_0
            if r <= r0:
                phicplx = np.sqrt(r0) / abs_u * (
                    _j(l)(mu_n*r/r0) -
                    (mu_n * dj_mu + (l + 1) * ediv * j_mu) *
                    (r/r0)**l / (l + (l + 1) * ediv)
                )
            else:
                phicplx = np.sqrt(r0) / abs_u * (
                    (-mu_n * dj_mu + l * ediv * j_mu) *
                    (r0/r)**(l+1) / (l + (l + 1) * ediv)
                )
            return complex(phicplx).real  # TODO: Should this be here?
        return phifn

    def _b(self, l, m, n):
        return qt.qeye(2)  # TODO

    def iter_lmn(self):
        for l in range(self.l_max+1):
            for m in range(-l, l+1):
                for mu_n, n in self.iter_mu_n(l=l):
                    yield l, m, n

    def iter_mu_n(self, l):
        for n in range(-self.n_max, self.n_max+1):
            mu = self.mu_nl(n=n, l=l)
            if mu is not None:
                yield mu, n

    def iter_omega_n(self, l):
        for mu, n in self.iter_mu_n(l):
            yield self.omega(mu=mu), n

    # -- Getters --
    def omega(self, l=None, n=None, mu=None, nu=None):
        if mu is not None and nu is not None:
            omega2 = self._omega2(mu2=mu**2, nu2=nu**2)
            return np.sqrt(omega2)
        else:
            return self.omega(mu=self.mu_nl(n=n, l=l), nu=self.nu_nl(n=n, l=l))

    def _omega2(self, mu2, nu2):
        return (
            (self.omega_T2 + self.omega_L2) / 2 -
            (self.beta_L2 * mu2 + self.beta_T2 * nu2) / 2 / self.r_0**2
        )

    def _Q2(self, mu, nu):
        return (
            (self.omega_T2 - self._omega2(mu2=mu**2, nu2=nu**2)) / self.beta_T2
        )

    def _q2(self, mu, nu):
        return (
            (self.omega_L2 - self._omega2(mu2=mu**2, nu2=nu**2)) / self.beta_L2
        )

    def _F_l(self, l, nu):
        r0 = self.r_0
        g = _g(l)(nu)
        dg = _g(l, d=1)(nu)
        return (
            self.gamma * (r0/nu)**2 * l * (nu * dg - l * g) +
            (l + (l + 1) * self._eps_div) * (nu * dg - g)
        )

    def _G_l(self, l, nu):
        r0 = self.r_0
        dg = _g(l, d=1)(nu)
        g = _g(l)(nu)
        ediv = self._eps_div
        ll = (l + (l + 1) * ediv)
        return self.gamma * (r0/nu)**2 * ediv * -(nu * dg - l * g) + ll * g

    def _norm_u(self, n, l):
        return np.sqrt(self._norm_u2(n=n, l=l))

    def _norm_u2(self, n, l):
        if (l, n) in self._u2:
            return self._u2[l, n]
        # If not in dictionary, calculate
        mu = self.mu_nl(n=n, l=l)
        nu = self.nu_nl(n=n, l=l)
        r0 = self.r_0
        mu0 = mu / r0
        nu0 = nu / r0
        pl = self._p_l(l=l, mu=mu, nu=nu)
        tl = self._t_l(l=l, mu=mu, nu=nu)

        def ifn(r):
            g = _g(l)(nu0 * r)
            dg = _g(l, d=1)(nu0 * r)
            j = _j(l)(mu0 * r)
            dj = _j(l, d=1)(mu0 * r)
            abs_ur2 = abs(
                -mu0 * dj + l * (l+1) / r * pl * g -
                tl * l / r0 * (r/r0) ** (l-1)
            )**2
            if l == 0:
                return r**2 * abs_ur2
            else:
                return (
                    r**2 * abs_ur2 +
                    l * (l + 1) +
                    r**2 * abs(
                        -j + pl / l * (g + nu0 * r * dg) -
                        tl * (r/r0) ** l
                    )**2
                )
        self._u2[l, n] = integ.quad(func=ifn, a=0, b=self.r_0)[0]
        return self._norm_u2(l=l, n=n)

    def _p_l(self, l, mu, nu):
        muk = mu
        return (
            (muk * _j(l, d=1)(mu) - l * _j(l)(mu)) /
            (l * _g(l)(nu) - nu * _g(l, d=1)(nu))
        )

    def _t_l(self, l, mu, nu):
        muk = mu
        ediv = self._eps_div
        return (
            self.gamma * (self.r_0 / nu) ** 2 *
            (muk * _j(l, d=1)(mu) + (l + 1) * ediv * _j(l)(mu)) /
            (l + ediv*(l+1))
        )

    # -- Root finding --
    def nu_nl(self, n, l):
        if n < 0:
            return self.nu_nl(n=-n, l=l)  # TODO: Is this right?
        elif (l, n) in self._roots_nu:
            return self._roots_nu[l, n]
        else:
            return None  # TODO

    def mu_nl(self, n, l):
        if n < 0:
            return -self.mu_nl(n=-n, l=l)  # TODO: Is this right?
        elif (l, n) in self._roots_mu:
            return self._roots_mu[l, n]
        else:
            return None  # TODO

    def _nu(self, mu):
        mu = complex(mu)
        return np.sqrt(self._beta_div2 * mu**2 - self.r_0**2 * self.gamma)

    def plot_root_function_mu(self, l, xdat, show=True):
        mudat_p = xdat + self.low_mu
        mudat_m = -xdat - self.low_mu
        fn = self._get_root_function(l=l)

        def rootfn(xy):
            mu, nu = xy
            f, g = fn(np.array([mu, nu]))
            return np.array([f, g])

        def fpr(mu):
            nu = self._nu(mu=mu)
            return np.real(rootfn(np.array([mu, nu])))[0]

        fig, ax = plt.subplots(1, 1)
        ax.axhline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(0, ls='-', color='black', lw=1, alpha=.5)

        for mu, n in self.iter_mu_n(l=l):
            if n < 0:
                pltzero = mu.real + self.low_mu
            else:
                pltzero = mu.real - self.low_mu
            ax.axvline(pltzero, ls='--', color='green', lw=1, alpha=.5)

        xdat = np.concatenate((-xdat, xdat))
        xdat.sort()
        mudat = np.concatenate((mudat_m, mudat_p))
        mudat.sort()

        ydat = np.array([fpr(x) for x in mudat])
        ydat /= lin.norm(ydat, ord=2)
        ax.plot(xdat, ydat, '-', color='red')

        # ylog = np.log(np.abs(ydat))
        # ylog /= lin.norm(ylog, ord=2)
        # ax.plot(xdat, ylog, '--', color='red')

        ypr = np.real([self._nu(x) for x in mudat])
        ypr /= lin.norm(ypr, ord=2)
        ax.plot(xdat, ypr, '-', color='blue')
        ax.plot(xdat, -ypr, '-', color='blue')

        if show:
            plt.show()
        return fig, ax

    def _fill_roots_mu_nu(self):
        for l in range(self.num_l):
            for root, n in self._solve_roots_xy(l=l):
                mu, nu = root
                assert not np.isnan(mu)  # TODO
                assert not np.isnan(nu)  # TODO
                self._roots_mu[l, n] = mu
                self._roots_nu[l, n] = nu

    def _get_expected_roots_xy(self, l, *args, **kwargs):
        start, mu_roots = self._expected_roots_mu[l]
        for mu0, n in zip(mu_roots, range(start, self.num_n)):
            mu0 = complex(mu0+self.low_mu)
            nu0 = self._nu(mu=mu0)
            yield np.array([mu0, nu0], dtype=complex), n

    def _get_root_func1(self, l, *args, **kwargs):
        def rf1(mu, nu):
            dj_mu = _j(l, d=1)(mu)
            if self._large_R:
                return (
                    mu * nu * dj_mu * _g(l, d=1)(mu) *
                    (self.gamma * l / self._Q2(mu=mu, nu=nu) +
                     (l + self._eps_div * (l + 1)))
                )
            else:
                Fl_nu = self._F_l(l=l, nu=nu)
                jl_mu = _j(l)(mu)
                Gl_nu = self._G_l(l=l, nu=nu)
                return (
                    mu * dj_mu * Fl_nu -
                    l * (l + 1) * jl_mu * Gl_nu
                )
        return rf1

    def _get_root_func2(self, *args, **kwargs):
        def rf2(mu, nu):
            return (mu**2 * self._beta_div2 - nu**2) - self.r_0**2 * self.gamma
        return rf2

