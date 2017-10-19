"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definitions used in the theoretical model are based on

Electron Raman Scattering in Nanostructures
R. Betancourt-Riera, R. Riera, J. L. Marin
Centro de Investigación en Física, Universidad de Sonora, 83190 Hermosillo, Sonora, México
R. Rosas
Departamento de Física, Universidad de Sonora, 83000 Hermosillo, Sonora, México
"""
import itertools as it
from math import pi

import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from scipy import integrate as integ
from scipy import interpolate as interp
from scipy import optimize as opt

from helper_functions import Y_lm
from helper_functions import j_sph as _j
from helper_functions import g_sph as _g


def _get_roots(fun, expected_roots, dfun=None, round_place=None, verbose=False):
    def f(x):
        ans = fun(x[0] + 1j*x[1])
        return np.array([ans.real, ans.imag])

    def df(x):
        ans = dfun(x)
        return np.array([[ans.real, -ans.imag], [ans.imag, ans.real]])

    expected_roots = sorted(expected_roots, key=lambda x: abs(x))
    roots = []
    modes = []
    if verbose:
        print(expected_roots)
    n = 0
    for x0 in zip(expected_roots):
        x0 = complex(x0)
        if dfun is not None:
            result = opt.root(fun=f, x0=np.array([x0.real, x0.imag]), jac=df)
        else:
            result = opt.root(fun=f, x0=np.array([x0.real, x0.imag]))
        root = result.x[0]
        if round_place is not None:
            root = round(root, round_place)
        if not result.success and verbose:
            print('>>  FAIL: {}'.format(result.message))
        elif root not in roots:
            roots.append(result.x[0])
            modes.append(n)
            n += 1
            # if verbose:
            #     print('>>  SUCCESS')
            #     print('      root: x={}'.format(result.x[0]))
    if len(roots) < len(expected_roots) and verbose:
        print('>>  ERR: Found {} of {} expected roots'
              ''.format(len(roots), len(expected_roots)))
    if verbose:
        print('>> roots = {}'.format(roots))
    return roots, modes


def _get_roots_periodic(fun, expected_period, num_roots, dfun=None,
                        round_place=None, x0=None, verbose=False,
                        max_iter_per_root=10):
    def f(x):
        ans = fun(x[0] + 1j * x[1])
        return np.array([ans.real, ans.imag])

    def df(x):
        ans = dfun(x[0] + 1j * x[1])
        return np.array([[ans.real, -ans.imag], [ans.imag, ans.real]])

    roots = []
    modes = []
    if x0 is None:
        x0 = expected_period / 2 + 0.j
    else:
        x0 += expected_period / 2
    x0 = complex(x0)
    n = 0
    for i in range(max_iter_per_root*num_roots):
        if n >= num_roots:
            break
        if dfun is not None:
            result = opt.root(fun=f, x0=np.array([x0.real, x0.imag]), jac=df)
        else:
            result = opt.root(fun=f, x0=np.array([x0.real, x0.imag]))
        root = result.x[0]
        if round_place is not None:
            root = round(root, round_place)
        if not result.success and verbose:
            print('>>  FAIL: {}'.format(result.message))
        elif root not in roots:
            roots.append(result.x[0])
            modes.append(n)
            n += 1
        if n >= num_roots:
            break
        else:
            x0 += expected_period / 2
    if verbose:
        print('>> roots = {}'.format(roots))
    return roots, modes


def _get_roots_interpolate(
        fun, expected_period, num_roots, round_place=None, x0=None, dfun=None,
        verbose=False, npts=1000,
):
    if x0 is None:
        x0 = expected_period / 2 + 0.j
    else:
        x0 += expected_period / 2
    xdat = np.linspace(x0, x0 + num_roots * expected_period, npts)
    ydat = np.array([fun(x) for x in xdat])

    print(xdat)
    print(ydat)

    interpfn_re = interp.make_interp_spline(xdat, np.real(ydat))
    interpfn_im = interp.make_interp_spline(xdat, np.imag(ydat))

    # def fn(x):
    #     ans_re = interpfn_re(x[0] + 1j * x[1])
    #     ans_im = interpfn_im(x[0] + 1j * x[1])
    #     return np.array([ans_re, ans_im])

    roots = interp.sproot(interpfn_re, num_roots)
    return sorted(list(roots)), list(range(len(roots)))


def plot_dispersion_function(xdat, rootfn, iterfn, fig=None, ax=None,
                             show=True):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        ax = fig.add_subplot(111)

    xdat = np.array(xdat)
    ydatr = np.array([rootfn(x) for x in xdat])
    # ydati = np.array([rootfn(1j * x) for x in xdat])

    # ax.plot(xdat, np.real(ydatr), '-')
    ax.plot(xdat, np.real(ydatr), '-', color='red')
    # ax.plot(xdat, np.real(ydati), '-', color='blue')

    zxdat = np.array(list(iterfn))
    zydat = np.zeros(shape=zxdat.shape, dtype=np.float)
    ax.plot(np.real(zxdat), zydat, 'o', color='green', alpha=.75)
    # ax.plot(np.imag(zxdat), zydat, 'o', color='orange', alpha=.75)

    ax.axhline(0., ls='--', lw=1., color='gray', alpha=.5)
    ax.axvline(0., ls='--', lw=1., color='gray', alpha=.5)
    if show:
        plt.show()
    return ydatr, ax


def _plot_dispersion_function2d(xdat, ydat, rootfn, iterfn):
    fig, ax = plt.subplots(1, 1)

    xdat = np.array(xdat)
    ydat = np.array(ydat)
    xdat, ydat = np.meshgrid(xdat, ydat)

    zdat = np.empty_like(xdat)
    for i in range(zdat.shape[0]):
        for j in range(zdat.shape[1]):
            zdat[i, j] = abs(rootfn(xdat[i, j] + 1j * ydat[i, j]))

    plt.contourf(xdat, ydat, zdat, len(zdat), alpha=.5)

    # Plot zero contours
    cgraph = ax.contour(xdat, ydat, zdat)
    ax.clabel(cgraph, inline=1, fontsize=10)

    # Plot zeros
    zeros = np.array(list(iterfn))
    zxdat = np.real(zeros)
    zydat = np.imag(zeros)
    ax.plot(zxdat, zydat, 'o', color='green')
    plt.show()

    return fig, ax


class ModelSpaceEPI:
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


class HamiltonianEPI:
    def __init__(self, r_0, omega_LO, l_max, max_n_modes,
                 epsilon_0_qdot, epsilon_inf_qdot,
                 epsilon_inf_env, constants, beta_T, beta_L,
                 large_R_approximation=False,
                 periodic_roots=False, interp_roots=False,
                 periodic_roots_start_dict=None,
                 expected_root_period_dict=None,
                 num_roots=None,
                 verbose_roots=False, expected_mu_dict=None,
                 known_mu_dict=None, mu_round=None,
                 num_points_interp=1000):
        self.r_0 = r_0
        self.V = 4/3 * pi * self.r_0**3
        self.l_max = l_max
        self.n_max = max_n_modes - 1
        self.ms = ModelSpaceEPI(nmax=self.n_max, nfock=2)
        self.const = constants
        self.eps_a_0 = epsilon_0_qdot
        self.eps_a_inf = epsilon_inf_qdot
        self.eps_b_inf = epsilon_inf_env
        self._eps_div = self.eps_b_inf / self.eps_a_inf
        self.beta_L2 = beta_L**2
        self.beta_T2 = beta_T**2
        self._beta_div2 = self.beta_L2 / self.beta_T2
        self.omega_L = omega_LO
        self.omega_L2 = self.omega_L**2
        self.omega_T2 = self.omega_L2 * self.eps_a_inf / self.eps_a_0
        self.omega_T = np.sqrt(self.omega_T2)
        self.omega_F2 = (
            self.omega_T2 * (self.const.epsilon_0 + 2 * self.eps_b_inf) /
            (self.eps_a_inf + 2 * self.eps_b_inf)
        )
        self.omega_F = np.sqrt(self.omega_F2)
        # NOTE: The following definition for gamma is different from gamma_0
        # of Riera
        self.gamma = (self.omega_L2 - self.omega_T2) / self.beta_T2
        self.C_F = -np.sqrt(
            2 * pi * self.const.e ** 2 * self.const.hbar * self.omega_L / self.V *
            (1/self.eps_a_inf - 1/self.eps_a_0)
        )

        # Root finding
        self._roots_mu = dict()  # (l, n) -> mu_ln
        self._large_R = large_R_approximation
        self._mu_round = mu_round
        self._verbose_roots = verbose_roots

        self._periodic_roots = periodic_roots
        self._interp_roots = interp_roots
        self._expected_root_period = dict()
        self._expected_num_roots = num_roots
        self._periodic_roots_start = periodic_roots_start_dict
        self._num_points_interp = num_points_interp

        # self._periodic_roots_start = np.sqrt(
        #     self.gamma * self.r_0**2 / self._beta_div2
        # )

        self._expected_roots_mu = dict()

        if known_mu_dict is not None:
            self._roots_mu.update(known_mu_dict)
        if expected_mu_dict is not None:
            self._expected_roots_mu.update(expected_mu_dict)
        if expected_root_period_dict is not None:
            self._expected_root_period.update(expected_root_period_dict)
        if self._periodic_roots_start is None:
            for l in range(l_max + 1):
                self._periodic_roots_start[l] = np.sqrt(
                    self.gamma * self.r_0**2 / self._beta_div2
                )

        if self._expected_num_roots is None:
            self._expected_num_roots = 2 * self.n_max + 1

        self._fill_roots_mu()

    def H_epi(self, r, theta, phi):
        """Electron-phonon interaction Hamiltonian at spherical coordinates
        (r, theta, phi). The definition is based on (Riera, Eq. 59).
        """
        h = 0
        for l, m, n in self.iter_lmn():
            h += (
                self.r_0 * self.C_F * np.sqrt(4*pi/3) *
                self._get_Phi_ln(l=l, n=n)(r=r) *
                Y_lm(l=l, m=m)(theta=theta, phi=phi) *
                (self._b(l=l, m=m, n=n) + self._b(l=l, m=m, n=n).dag())
            )
        return

    def _b(self, l, m, n):
        return qt.qeye(2)  # TODO

    def iter_lmn(self):
        for l in range(self.l_max+1):
            for m in range(-l, l+1):
                for mu_n, n in self.iter_mu_n(l=l):
                    yield l, m, n

    def iter_mu_n(self, l):
        for n, i in zip(range(self.n_max + 1), it.count()):
            mu = self.mu_nl(n=n, l=l)
            # yield mu, i
            yield mu, n

    def iter_mu(self, l):
        for mu, n in self.iter_mu_n(l=l):
            yield mu

    def iter_omega_n(self, l):
        for mu, n in self.iter_mu_n(l):
            yield self.omega(mu=mu), n

    def omega(self, mu, d_mu=0):
        if d_mu == 0:
            return np.sqrt(self._omega2(mu=mu))
        elif d_mu == 1:
            return 1/2/self.omega(mu=mu) * self._omega2(mu=mu, d_mu=1)
        else:
            raise RuntimeError

    def _omega2(self, mu, d_mu=0):
        if d_mu == 0:
            return self.omega_L2 - self.beta_L2 * self._q2(mu=mu)
        elif d_mu > 0:
            return -self.beta_L2 * self._q2(mu=mu, d_mu=d_mu)
        else:
            raise RuntimeError

    def Q(self, mu, d_mu=0):
        if d_mu == 0:
            return np.sqrt(self._Q2(mu=mu))
        elif d_mu == 1:
            return (
                1/2/self.Q(mu=mu) * self._Q2(mu=mu, d_mu=1)
            )
        else:
            raise RuntimeError  # TODO

    def _Q2(self, mu, d_mu=0):
        if d_mu == 0:
            return (self.omega_T2 - self._omega2(mu=mu)) / self.beta_T2
        elif d_mu > 0:
            return -1/self.beta_T2 * self._omega2(mu=mu, d_mu=d_mu)
        else:
            raise RuntimeError  # TODO

    def q(self, mu, d_mu=0):
        if d_mu == 0:
            return complex(mu / self.r_0)
        elif d_mu == 1:
            return complex(1 / self.r_0)
        elif d_mu > 1:
            return 0+0j
        else:
            raise RuntimeError  # TODO

    def _q2(self, mu, d_mu=0):
        if d_mu == 0:
            return self.q(mu=mu)**2
        elif d_mu == 1:
            return 2 * self.q(mu=mu) * self.q(mu=mu, d_mu=1)
        else:
            raise RuntimeError  # TODO

    def nu(self, mu, d_mu=0):
        return self.r_0 * self.Q(mu=mu, d_mu=d_mu)

    def _F_l(self, l, nu, d_nu=0):
        r0 = self.r_0
        g = _g(l)(nu)
        dg = _g(l, d=1)(nu)
        if d_nu == 0:
            return (
                self.gamma * (r0/nu)**2 * l * (nu * dg - l * g) +
                (l + (l + 1) * self._eps_div) * (nu * dg - g)
            )
        elif d_nu == 1:
            d2g = _g(l, d=2)(nu)
            return (
                self.gamma * r0**2 * l * (
                    -2*nu**(-3) * (nu * dg - l * g) +
                    nu**(-2) * (dg + nu * d2g - l * dg)
                ) +
                (l + (l + 1) * self._eps_div) * (dg + nu * d2g - dg)
            )
        else:
            raise RuntimeError  # TODO

    def _G_l(self, l, nu, d_nu=0):
        r0 = self.r_0
        dg = _g(l, d=1)(nu)
        g = _g(l)(nu)
        ediv = self._eps_div
        ll = (l + (l + 1) * ediv)
        if d_nu == 0:
            return (
                self.gamma * (r0/nu)**2 * ediv * -(nu * dg - l * g) +
                ll * g
            )
        elif d_nu == 1:
            d2g = _g(l, d=2)(nu)
            return (
                self.gamma * r0**2 * ediv * (
                    2 / nu**3 * (nu * dg - l * g) -
                    1 / nu**2 * (dg + nu * d2g - l * dg)
                ) +
                ll * dg
            )
        else:
            raise RuntimeError

    def _get_Phi_ln(self, l, n):
        fact = np.sqrt(self.r_0) / self._norm_u(n=n, l=l)
        mu_n = self.mu_nl(n=n, l=l)
        ediv = self._eps_div
        def phifn(r):
            r0 = self.r_0
            if r <= r0:
                return fact * (
                    _j(l)(mu_n * r / r0) -
                    (mu_n * _j(l, d=1)(mu_n) + (l + 1) * ediv * _j(l)(mu_n)) /
                    (l + (l + 1) * ediv) * (r/r0) ** l
                )
            else:
                return fact * (
                    -(mu_n * _j(l, d=1)(mu_n) + l * ediv * _j(l)(mu_n)) /
                    (l + (l + 1) * ediv) * (r/r0) ** l
                )
        return phifn

    def mu_nl(self, n, l):
        # if l not in self._expected_roots_mu:
        #     return None
        # elif n > self.n_max:
        if n > self.n_max:
            return None
        elif (l, n) not in self._roots_mu:
            return 0
            # raise RuntimeError  # TODO
            # return 0  # TODO get rid of this
        else:
            return self._roots_mu[l, n]

    def _get_expected_roots_mu(self, l):
        if l in self._expected_roots_mu:
            return sorted(self._expected_roots_mu[l], key=lambda x: abs(x))
        else:
            return []

    def _get_expected_root_period(self, l):
        if l in self._expected_root_period:
            return self._expected_root_period[l]
        else:
            return None

    def _get_periodic_root_start(self, l):
        if self._periodic_roots_start is None or l not in self._periodic_roots_start:
            return np.sqrt(self.gamma * self.r_0**2 / self._beta_div2)
        else:
            return self._periodic_roots_start[l]

    def _get_root_function_mu(self, l):
        def rootfn(mu):
            nu = self.nu(mu=mu)
            dj_mu = _j(l, d=1)(mu)
            if self._large_R:
                return (
                    mu * nu * dj_mu * _g(l, d=1)(mu) *
                    (self.gamma * l / self.Q(mu=mu)**2 +
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
        return rootfn

    def _get_root_function_deriv(self, l):
        def drootfn(mu):
            nu = self.nu(mu=mu)
            dnu = self.nu(mu=mu, d_mu=1)
            j = _j(l=l)(mu)
            dj = _j(l=l, d=1)(mu)
            d2j = _j(l=l, d=2)(mu)
            F = self._F_l(l=l, nu=nu)
            dF = self._F_l(l=l, nu=nu, d_nu=1)
            G = self._G_l(l=l, nu=nu)
            dG = self._G_l(l=l, nu=nu, d_nu=1)
            if not self._large_R:
                return (
                    j*F + mu*d2j*F + mu*dj*dnu*dF -
                    l*(l+1) * (dj*G + j*dnu*dG)
                )
            else:
                return 0  # TODO
        return drootfn

    def plot_root_function_mu(self, l, xdat, cplx=False, fig=None, ax=None,
                              show=True):
        if not cplx:
            return plot_dispersion_function(
                xdat=xdat, rootfn=self._get_root_function_mu(l=l),
                iterfn=self.iter_mu(l=l), fig=fig, ax=ax, show=show
            )
        else:
            return _plot_dispersion_function2d(
                xdat=xdat, ydat=xdat, rootfn=self._get_root_function_mu(l=l),
                iterfn=self.iter_mu(l=l)
            )

    def _norm_u(self, n, l):
        return np.sqrt(self._norm_u2(n=n, l=l))

    def _norm_u2(self, n, l):
        mu = self.mu_nl(n=n, l=l)
        r0 = self.r_0
        mu0 = mu/r0
        nu0 = self.nu(mu=mu)/r0
        pl = self._p_l(l=l, mu=mu)
        tl = self._t_l(l=l, mu=mu)

        def ifn(r):
            return r**2 * (
                (
                    -mu0 * _j(l, d=1)(mu0 * r) +
                    l * (l+1) / r * pl * _g(l)(nu0 * r) -
                    tl * l / r0 * (r/r0) ** (l-1)
                )**2 +
                l * (l + 1) / r**2 *
                (
                    -_j(l)(mu0 * r) + pl / l *
                    (_g(l)(nu0 * r) + nu0 * r * _g(l, d=1)(nu0 * r)) -
                    tl * (r/r0) ** l
                )**2
            )
        return integ.quad(func=ifn, a=0, b=self.r_0)[0]

    def _p_l(self, l, mu, k=0):
        nu = self.nu(mu=mu)
        # muk = self.mu_nl(n=k, l=l)
        muk = mu
        return (
            (muk * _j(l, d=1)(mu) - l * _j(l)(mu)) /
            (l * _g(l)(nu) - nu * _g(l, d=1)(nu))
        )

    def _t_l(self, l, mu, k=0):
        # muk = self.mu_nl(n=k, l=l)
        muk = mu
        ediv = self._eps_div
        return (
            self.gamma * (self.r_0/self.nu(mu=muk)) ** 2 *
            (muk * _j(l, d=1)(mu) + (l + 1) * ediv * _j(l)(mu)) /
            (l + ediv*(l+1))
        )

    def _fill_roots_mu(self):
        for l in range(self.l_max + 1):
            # Get roots from expected
            if self._periodic_roots:
                if l in self._expected_root_period:
                    new_roots, modes = _get_roots_periodic(
                        fun=self._get_root_function_mu(l=l),
                        dfun=self._get_root_function_deriv(l=l),
                        expected_period=self._expected_root_period[l],
                        num_roots=self._expected_num_roots,
                        x0=self._get_periodic_root_start(l=l)*(1.+1.e-4),
                        verbose=self._verbose_roots,
                    )
                else:
                    new_roots, modes = [], []
            elif self._interp_roots:
                if l in self._expected_root_period:
                    new_roots, modes = _get_roots_interpolate(
                        fun=self._get_root_function_mu(l=l),
                        dfun=self._get_root_function_deriv(l=l),
                        expected_period=self._expected_root_period[l],
                        num_roots=self._expected_num_roots,
                        x0=self._get_periodic_root_start(l=l)*(1.+1.e-4),
                        verbose=self._verbose_roots,
                        npts=self._num_points_interp
                    )
                else:
                    new_roots, modes = [], []
            else:
                new_roots, modes = _get_roots(
                    fun=self._get_root_function_mu(l=l),
                    dfun=self._get_root_function_deriv(l=l),
                    expected_roots=self._get_expected_roots_mu(l=l),
                    verbose=self._verbose_roots
                )
            # Sort and update
            for root, mode in zip(new_roots, modes):
                self._roots_mu[l, mode] = root
