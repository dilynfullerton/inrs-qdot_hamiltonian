"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definitions used in the theoretical model are based on

Electron Raman Scattering in Nanostructures
R. Betancourt-Riera, R. Riera, J. L. Marín
Centro de Investigación en Física, Universidad de Sonora, 83190 Hermosillo, Sonora, México
R. Rosas
Departamento de Física, Universidad de Sonora, 83000 Hermosillo, Sonora, México
"""
from math import pi
import qutip as qt
import numpy as np
from scipy import optimize as opt
from scipy import special as sp
from scipy import integrate as integ
from matplotlib import pyplot as plt
from matplotlib import mlab
import itertools as it


def _get_roots(rootfn, expected_roots):
    def fn(x):
        ans = rootfn(x[0] + 1j*x[1])
        return np.array([ans.real, ans.imag])
    roots = []
    for x0 in expected_roots:
        result = opt.root(fun=fn, x0=np.array([x0.real, x0.imag]))
        if not result.success:
            print('>>  FAIL: {}'.format(result.message))
        else:
            # print('>>  SUCCESS: {}'.format(result.message))
            print('>>  SUCCESS')
            print('      root: x={}'.format(result.x[0]))
            roots.append(result.x[0])
    if len(roots) < len(expected_roots):
        print('>>  ERR: Found {} of {} expected roots'
              ''.format(len(roots), len(expected_roots)))
    return sorted(roots)


def _plot_dispersion_function(xdat, rootfn, iterfn, ximag=False):
    fig, ax = plt.subplots(1, 1)

    xdat = np.array(xdat)
    if ximag:
        xdat = np.imag(xdat)
    else:
        xdat = np.real(xdat)

    ydat = np.array([rootfn(x) for x in xdat])
    ax.plot(xdat, np.real(ydat), '-', color='red')

    zxdat = np.array(list(iterfn))
    zydat = np.zeros_like(zxdat)
    ax.plot(zxdat, zydat, 'o', color='green')

    ax.axhline(0., ls='--', lw=1., color='gray', alpha=.5)
    ax.axvline(0., ls='--', lw=1., color='gray', alpha=.5)
    plt.show()
    return fig, ax


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


def _get_Y_lm(l, m):
    def yfn(theta, phi):
        return sp.sph_harm(m, l, phi, theta)
    return yfn


def _J(l, d=0):
    def fn_j(z):
        z = complex(z)
        return sp.jvp(v=l, z=z, n=d)
    return fn_j


def _spherical_bessel1(l, func, d=0):
    def fn_spb(x):
        if d == 0:
            return func(n=l, z=x, derivative=False)
        elif d == 1:
            return func(n=l, z=x, derivative=True)
        else:
            raise RuntimeError  # TODO
        # elif l == 0:
        #     return -_spherical_bessel1(l=1, func=func, d=d-1)(x)
        # else:
        #     return (
        #         _spherical_bessel1(l=l-1, func=func, d=d-1)(x) -
        #         (l+1) / 2 * _spherical_bessel1(l=l, func=func, d=d-1)(x)
        #     )
    return fn_spb


def _j(l, d=0):
    return _spherical_bessel1(l=l, func=sp.spherical_jn, d=d)


def _i(l, d=0):
    return _spherical_bessel1(l=l, func=sp.spherical_in, d=d)


def _g(l, d=0):
    def fn_g(z):
        z = complex(z)
        if (z**2).real >= 0:
            j = _j(l, d)(z)
            return j
        else:
            i = _i(l, d)(z)
            return i
    return fn_g


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
    def __init__(self, r_0, omega_LO, l_max, n_max,
                 epsilon_0_qdot, epsilon_inf_qdot,
                 epsilon_inf_env, constants, beta_T, beta_L,
                 large_R_approximation=True,
                 expected_mu_dict=None, known_mu_dict=None):
        self.ms = ModelSpaceEPI(nmax=n_max, nfock=2)
        self.r_0 = r_0
        self.V = 4/3 * pi * self.r_0**3
        self.l_max = l_max
        self.n_max = n_max
        self.const = constants
        self.eps_a_0 = epsilon_0_qdot
        self.eps_a_inf = epsilon_inf_qdot
        self._eps_div = epsilon_inf_env / self.eps_a_inf
        self.beta_L2 = beta_L**2
        self.beta_T2 = beta_T**2
        self._beta_div2 = self.beta_L2 / self.beta_T2
        self._omega_L = omega_LO
        self.omega_L2 = self._omega_L**2
        self.omega_T2 = self.omega_L2 * self.eps_a_inf / self.eps_a_0
        # NOTE: The following definition for gamma is different from gamma_0
        # of Riera
        self.gamma = (self.omega_L2 - self.omega_T2) / self.beta_T2
        self.C_F = -np.sqrt(
            2*pi * self.const.e**2 * self.const.hbar * self._omega_L / self.V *
            (1/self.eps_a_inf - 1/self.eps_a_0)
        )
        self._expected_roots_mu = dict()
        self._roots_mu = dict()

        if known_mu_dict is not None:
            self._roots_mu.update(known_mu_dict)
        if expected_mu_dict is not None:
            self._expected_roots_mu.update(expected_mu_dict)

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
                _get_Y_lm(l=l, m=m)(theta=theta, phi=phi) *
                (self._b(l=l, m=m, n=n) + self._b(l=l, m=m, n=n).dag())
            )
        return

    def _b(self, l, m, n):
        return qt.qeye(2)  # TODO

    def iter_lmn(self):
        for l in range(self.l_max+1):
            for m in range(-l, l+1):
                for mu_n, n in zip(self.iter_mu(l), range(self.n_max+1)):
                    yield l, m, n

    def iter_mu(self, l):
        for n in range(self.n_max + 1):
            mu = self.mu_nl(n=n, l=l)
            if mu is None:
                break  # TODO: Anything else?
            else:
                yield mu

    def iter_mu_n(self, l):
        for mu, n in zip(self.iter_mu(l=l), range(self.n_max+1)):
            yield mu, n

    def iter_omega_n(self, l):
        for mu, n in self.iter_mu_n(l):
            yield self.omega(mu=mu), n

    # TODO continue search for errors
    def omega(self, mu):
        return np.sqrt(
            complex(self.omega_L2 - self.beta_L2 * self.q(mu=mu)**2)
        )

    def _get_Phi_ln(self, l, n):
        fact = np.sqrt(self.r_0) / self._norm_u(n=n, l=l)
        mu_n = self.mu_nl(n=n, l=l)
        ediv = self._eps_div

        def phifn(r):
            r0 = self.r_0
            if r <= r0:
                return fact * (
                    _j(l)(mu_n*r/r0) -
                    (mu_n * _j(l, d=1)(mu_n) + (l+1) * ediv * _j(l)(mu_n)) /
                    (l + (l + 1) * ediv) * (r/r0)**l
                )
            else:
                return fact * (
                    -(mu_n * _j(l, d=1)(mu_n) + l * ediv * _j(l)(mu_n)) /
                    (l + (l + 1) * ediv) * (r/r0)**l
                )
        return phifn

    def mu_nl(self, n, l):
        if l not in self._expected_roots_mu:
            return None
        elif n > self.n_max:
            return None
        elif l not in self._roots_mu or len(self._roots_mu[l]) == 0:
            roots = _get_roots(
                rootfn=self._get_root_function_mu(l=l),
                expected_roots=self._get_expected_roots_mu(l=l)
            )
            self._roots_mu[l] = sorted(roots)
            return self.mu_nl(n=n, l=l)
        elif n >= len(self._roots_mu[l]):
            print('>> No root found for l={} and n={}'.format(l, n))
            return None  # TODO
        else:
            return self._roots_mu[l][n]

    def _get_expected_roots_mu(self, l):
        if l in self._expected_roots_mu:
            return sorted(self._expected_roots_mu[l])
        else:
            return []

    def _get_root_function_mu(self, l):
        def rootfn(mu):
            if mu < 0:
                print('hi')
            nu = self.nu(mu=mu)
            dj_mu = _j(l, d=1)(mu)
            fl_nu = self._F_l(l=l)(nu=nu)
            jl_mu = _j(l)(mu)
            gl_nu = self._G_l(l=l)(nu=nu)
            ans = (
                nu * dj_mu * fl_nu -
                l * (l + 1) * jl_mu * gl_nu
            )
            assert not np.isnan(ans)
            return ans
        return rootfn

    def plot_root_function_mu(self, l, xdat, ximag=False, cplx=False):
        if not cplx:
            return _plot_dispersion_function(
                xdat=xdat, rootfn=self._get_root_function_mu(l=l),
                iterfn=self.iter_mu(l=l), ximag=ximag
            )
        else:
            return _plot_dispersion_function2d(
                xdat=xdat, ydat=xdat, rootfn=self._get_root_function_mu(l=l),
                iterfn=self.iter_mu(l=l)
            )

    def nu(self, Q=None, mu=None):
        if Q is not None:
            return Q * self.r_0
        elif mu is not None:
            return self.nu(Q=self.Q(mu=mu))
        else:
            raise RuntimeError  # TODO

    def Q(self, omega2=None, omega=None, mu=None):
        return np.sqrt(complex(self._Q2(omega2=omega2, omega=omega, mu=mu)))

    def _Q2(self, omega2=None, omega=None, mu=None):
        if omega2 is not None:
            return (self.omega_T2 - omega2) / self.beta_T2
        elif omega is not None:
            return self._Q2(omega2=omega**2)
        elif mu is not None:
            return self._Q2(omega=self.omega(mu=mu))
        else:
            raise RuntimeError  # TODO

    def q(self, mu):
        return mu / self.r_0

    def _F_l(self, l):
        def ffn(nu):
            return (
                self.gamma * (self.r_0/nu)**2 * l *
                (nu * _g(l, d=1)(nu) - l * _g(l)(nu)) +
                (l + (l + 1) * self._eps_div) *
                (nu * _g(l, d=1)(nu) - _g(l)(nu))
            )
        return ffn

    def _G_l(self, l):
        def gfn(nu):
            return (
                self.gamma * (self.r_0/nu)**2 * self._eps_div *
                -(nu * _g(l, d=1)(nu) - l * _g(l)(nu)) +
                (l + (l + 1) * self._eps_div) * _g(l)(nu)
            )
        return gfn

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
                    -mu0 * _j(l, d=1)(mu0*r) +
                    l*(l+1)/r * pl * _g(l)(nu0*r) -
                    tl * l/r0 * (r/r0)**(l-1)
                )**2 +
                l * (l + 1) / r**2 *
                (
                    -_j(l)(mu0*r) + pl/l *
                    (_g(l)(nu0*r) + nu0*r * _g(l, d=1)(nu0*r)) -
                    tl * (r/r0)**l
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
            self.gamma * (self.r_0/self.nu(mu=muk))**2 *
            (muk * _j(l, d=1)(mu) + (l+1)*ediv*_j(l)(mu)) /
            (l + ediv*(l+1))
        )

    def _fill_roots_mu(self):
        for l in range(self.l_max + 1):
            new_roots = _get_roots(
                rootfn=self._get_root_function_mu(l=l),
                expected_roots=self._get_expected_roots_mu(l=l)
            )
            if l in self._roots_mu:
                old_roots = filter(lambda r: r not in new_roots,
                                   self._roots_mu[l])
                new_roots = sorted(new_roots + list(old_roots))
            self._roots_mu[l] = new_roots
