import numpy as np
from math import pi
from matplotlib import pyplot as plt
from scipy import linalg as lin
from scipy import integrate as integ
from helper_functions import Y_lm, j_sph as _j, g_sph as _g
from helper_functions import basis_roca
from root_solver_2d import RootSolverComplex2d
from ModelSpace import ModelSpace
from collections import namedtuple


PhononMode = namedtuple('PhononMode', ['l', 'm', 'n'])


class PhononModelSpace(ModelSpace, RootSolverComplex2d):
    def __init__(
            self, nmax, lmax, radius, omega_L, omega_T, beta_T, beta_L,
            epsilon_inf_qdot, epsilon_inf_env, expected_roots_mu_l,
            large_R_approximation=False
    ):
        # Model constants
        self.nmax = nmax
        self.lmax = lmax
        self.num_n = self.nmax + 1
        self.num_l = self.lmax + 1

        # Physical constants
        self.R = radius
        self.eps_inf_in = epsilon_inf_qdot
        self.eps_inf_out = epsilon_inf_env
        self.beta_L2 = beta_L**2
        self.beta_T2 = beta_T**2
        self.omega_L = omega_L
        self.omega_T = omega_T

        # Derived physical constants
        self.volume = 4 / 3 * pi * self.R**3
        self._eps_div = self.eps_inf_out / self.eps_inf_in
        self._beta_div2 = self.beta_L2 / self.beta_T2
        self.omega_L2 = self.omega_L**2
        self.omega_T2 = self.omega_T**2
        self.eps0_in = self.eps_inf_in * self.omega_L2 / self.omega_T2
        self.omega_F2 = (
            self.omega_T2 * (self.eps0_in + 2 * self.eps_inf_out) /
            (self.eps_inf_in + 2 * self.eps_inf_out)
        )
        self.omega_F = np.sqrt(self.omega_F2)
        # NOTE: The following definition for gamma is different from gamma_0
        # of Riera
        self.gamma = (self.omega_L2 - self.omega_T2) / self.beta_T2
        self.C_F = np.sqrt(  # Riera (45)
            2 * pi * self.omega_L /
            self.volume * (1 / self.eps_inf_in - 1 / self.eps0_in)
        )
        self.high_mu = self.R / np.sqrt(self._beta_div2) * np.sqrt(self.gamma)
        self.alpha = np.sqrt(
            (self.eps0_in - self.eps_inf_in) * self.omega_T2 / 4 * np.pi)

        self._u_norm_dict = dict()

        # Root finding
        self._roots_mu = dict()  # l, n -> mu
        self._roots_nu = dict()  # l, n -> nu
        self._expected_roots_mu = expected_roots_mu_l  # l -> mu0
        self._large_R = large_R_approximation
        self._fill_roots_mu_nu()

    def nfock(self, mode):
        return 2

    def modes(self):
        for l in range(self.num_l):
            for m in range(-l, l+1):
                for n in range(self.num_n):
                    yield PhononMode(l=l, m=m, n=n)

    def states(self):
        """Returns an iterator over single-particle modes that have
        valid solutions
        """
        for mode in self.modes():
            if self.mu(mode) is not None:
                yield mode

    def get_nums(self, mode):
        """Gets the quantum numbers associated with the given
        single-particle state
        """
        return mode.l, mode.m, mode.n

    def get_ket(self, mode):
        return self.create(mode=mode) * self.vacuum_ket()

    def get_omega(self, state):
        """Returns the frequency associated with the given single-particle
        state
        """
        # TODO: Verify
        mu = self.mu(state)
        nu = self.nu(state)
        omega2 = self.omega2(mu=mu, nu=nu)
        return np.sqrt(omega2)

    def Phi_ln(self, state):
        """See Riera (60)
        """
        # TODO: Figure out whether this equation is correct
        # Note that for r > r0, there is a major difference between the
        # expressions of Roca and that of Riera; the former depends on a
        # difference of terms in the numerator while the latter depends on
        # a sum. I believe Roca's expression is correct, but this should be
        # verified.
        def phifn(r):
            l = state.l
            mu_n = self.mu(state)
            ediv = self._eps_div
            dj_mu = _j(l, d=1)(mu_n)
            j_mu = _j(l)(mu_n)
            r0 = self.R
            if r < r0:
                phi = np.sqrt(self.R) / self._u_norm(state) * (
                    _j(l)(mu_n*r/r0) -
                    (mu_n * dj_mu + (l + 1) * ediv * j_mu) /
                    (l + (l + 1) * ediv) * (r/r0)**l
                )
            else:
                phi = np.sqrt(self.R) / self._u_norm(state) * (
                    (-mu_n * dj_mu + l * ediv * j_mu) / (l + (l + 1) * ediv) *
                    (r/r0)**(-l-1)
                )
            return complex(phi).real  # TODO
        return phifn

    # -- States --
    def iter_mu_n(self, l):
        for state in self.states():
            if state.l == l and state.m == 0:
                yield self.mu(state), state.n

    # -- Getters --
    def omega2(self, mu, nu):
        mu2, nu2 = mu**2, nu**2
        return (
            (self.omega_T2 + self.omega_L2) / 2 -
            (self.beta_L2 * mu2 + self.beta_T2 * nu2) / 2 / self.R ** 2
        )

    def _Q2(self, mu, nu):
        return (self.omega_T2 - self.omega2(mu=mu, nu=nu)) / self.beta_T2

    def _q2(self, mu, nu):
        return (self.omega_L2 - self.omega2(mu=mu, nu=nu)) / self.beta_L2

    def _F_l(self, l, nu):
        r0 = self.R
        g = _g(l)(nu)
        dg = _g(l, d=1)(nu)
        return (
            self.gamma * (r0/nu)**2 * l * (nu * dg - l * g) +
            (l + (l + 1) * self._eps_div) * (nu * dg - g)
        )

    def _G_l(self, l, nu):
        r0 = self.R
        dg = _g(l, d=1)(nu)
        g = _g(l)(nu)
        ediv = self._eps_div
        ll = (l + (l + 1) * ediv)
        return self.gamma * (r0/nu)**2 * ediv * -(nu * dg - l * g) + ll * g

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
            self.gamma * (self.R / nu)**2 *
            (muk * _j(l, d=1)(mu) + (l + 1) * ediv * _j(l)(mu)) /
            (l + ediv*(l+1))
        )

    def _u_norm(self, state):
        """See Riera (64)
        """
        k = self._root_dict_key(state)
        if k in self._u_norm_dict:
            return self._u_norm_dict[k]

        mu, nu = self.mu(state), self.nu(state)
        r0 = self.R
        l = state.l
        jl = _j(l)
        djl = _j(l, d=1)
        gl = _g(l)
        dgl = _g(l, d=1)
        pl = self._p_l(l, mu=mu, nu=nu)
        tl = self._t_l(l, mu=mu, nu=nu)

        def int_norm2(r):
            u_norm2 = abs(
                -mu*r0 * djl(mu*r/r0) + l*(l+1)/r*pl*gl(nu*r/r0) -
                tl*l/r0*(r/r0)**(l-1)
            )**2
            if l > 0:
                u_norm2 += l*(l+1)/r**2 * abs(
                    -jl(mu*r/r0) + pl/l *
                    (gl(nu*r/r0) + nu*r/r0*dgl(nu*r/r0)) - tl*(r/r0)**l
                )**2
            return u_norm2 * r**2

        ans = integ.quad(int_norm2, 0, r0)[0]
        self._u_norm_dict[k] = np.sqrt(ans)
        return self._u_norm(state=state)

    # -- Root finding --
    def _root_dict_key(self, state):
        return state.l, state.n

    def nu(self, state):
        k = self._root_dict_key(state)
        if k in self._roots_nu:
            return self._roots_nu[k]
        else:
            return None  # TODO

    def mu(self, state):
        k = self._root_dict_key(state)
        if k in self._roots_mu:
            return self._roots_mu[k]
        else:
            return None  # TODO

    def _nu_from_mu(self, mu):
        mu = complex(mu)
        return np.sqrt(self._beta_div2 * mu ** 2 - self.R ** 2 * self.gamma)

    def plot_root_function_mu(self, l, xdat, show=True):
        mudat = xdat
        fn = self._get_root_function(l=l)

        def rootfn(xy):
            mu, nu = xy
            f, g = fn(np.array([mu, nu]))
            return np.array([f, g])

        def fpr(mu):
            nu = self._nu_from_mu(mu=mu)
            return rootfn(np.array([mu, nu]))[0]

        fig, ax = plt.subplots(1, 1)

        # Plot axes
        ax.axhline(0, color='gray', lw=1, alpha=.5)
        if min(xdat) < 0 < max(xdat):
            ax.axvline(0, color='gray', lw=1, alpha=.5)

        # Plot high_mu boundaries
        if min(xdat) <= self.high_mu <= max(xdat):
            ax.axvline(self.high_mu, ls='-', color='black', lw=1, alpha=.5)
        if min(xdat) <= -self.high_mu <= max(xdat):
            ax.axvline(-self.high_mu, ls='-', color='black', lw=1, alpha=.5)

        # Plot real and imaginary parts of the root function
        ydat = np.array([fpr(x) for x in mudat])
        ydat /= lin.norm(ydat, ord=2)
        linei, = ax.plot(xdat, np.imag(ydat), '-', color='blue')
        liner, = ax.plot(xdat, np.real(ydat), '-', color='red')

        # Plot log function to aid in identifying roots
        ylog = np.log(np.abs(ydat))
        ylog /= lin.norm(ylog, ord=2)
        ax.plot(xdat, ylog, '--', color='brown')

        # Region of imaginary Q
        span_imQ = None
        x_min_imQ = max(min(xdat), -self.high_mu)
        x_max_imQ = min(max(xdat), self.high_mu)
        if x_min_imQ < x_max_imQ:
            span_imQ = ax.axvspan(x_min_imQ, x_max_imQ, color='blue', alpha=.3)

        # Regions of real Q
        span_reQ = None
        x_min_reQ = max(min(xdat), self.high_mu)
        x_max_reQ = max(xdat)
        if x_min_reQ < x_max_reQ:
            span_reQ = ax.axvspan(x_min_reQ, x_max_reQ, color='red', alpha=.3)
        x_min_reQ_m = min(xdat)
        x_max_reQ_m = min(max(xdat), -self.high_mu)
        if x_min_reQ_m < x_max_reQ_m:
            span_reQ = ax.axvspan(x_min_reQ_m, x_max_reQ_m, color='red', alpha=.3)

        # Roots
        for mu, n in self.iter_mu_n(l=l):
            if min(xdat) < mu.real < max(xdat):
                ax.axvline(mu.real, ls='--', color='green', lw=1, alpha=.5)
            else:
                print('Root out of range: mu={}'.format(mu.real))

        # Title and legend
        ax.set_title('Phonon root function for l = {}'.format(l))
        ax.legend(
            (liner, linei, span_imQ, span_reQ),
            ('root equation (real part)', 'root equation (imaginary part)',
             'imaginary Q', 'real Q')
        )

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
            mu0 = complex(mu0)
            nu0 = self._nu_from_mu(mu=mu0)
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
            return (mu**2 * self._beta_div2 - nu**2) - self.R ** 2 * self.gamma
        return rf2
