import numpy as np
from math import pi
from matplotlib import pyplot as plt
from scipy import linalg as lin
from scipy import integrate as integ
from helper_functions import Y_lm, j_sph as _j, g_sph as _g
from helper_functions import basis_roca
from root_solver_2d import RootSolverComplex2d
from ModelSpace import ModelSpace


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
        self.volume = 4 / 3 * pi * self.R ** 3
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
        self.C_F = - np.sqrt(  # Riera (45)
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
                    yield (l, m, n)

    def states(self):
        return self.iter_states()

    def get_nums(self, state):
        return state

    def get_ket(self, state):
        return self.create(mode=state) * self.vacuum_ket()

    def get_omega(self, state):
        # TODO: Verify
        mu = self.mu_nl(state)
        nu = self.nu_nl(state)
        omega2 = self.omega2(mu=mu, nu=nu)
        return np.sqrt(omega2)

    def get_phi_rad(self, state):
        return self.phi_rad_ln(state)

    def get_phi_ang(self, state):
        return self.phi_ang_lm(state)

    def phi_rad_ln(self, state):
        def hfunc(r):
            if state is self.vacuum_state():
                return 0
            else:
                l, m, n = self.get_nums(state)
                return (
                    self.C_F * self.R / self.mu_nl(state) *
                    (2 * l + 1) * 1j ** (l % 2) * np.sqrt(2*pi) *
                    self.Phi_ln(state)(r)
                )
        return hfunc

    def phi_ang_lm(self, state):
        if state is self.vacuum_state():
            return 0
        else:
            l, m, n = self.get_nums(state)
            return Y_lm(l=l, m=m)

    def Phi_ln(self, state):
        """See Roca (43) and Riera (60)
        """
        def phifn(r, dr=0):
            if state is self.vacuum_state():
                return 0
            l, m, n = self.get_nums(state)
            mu_n = self.mu_nl(state)
            ediv = self._eps_div
            dj_mu = _j(l, d=1)(mu_n)
            j_mu = _j(l)(mu_n)
            r0 = self.R
            if dr == 0 and r <= r0:
                return (
                    -_j(l)(mu_n*r/r0) * (l + (l + 1) * ediv) +
                    (mu_n * dj_mu + (l + 1) * ediv * j_mu) * (r/r0)**l
                )
            elif dr == 0:
                return (mu_n * dj_mu - l * ediv * j_mu) * (r/r0)**(-l-1)
            elif dr == 1 and r <= r0:
                return (
                    -mu_n/r0 * _j(l, d=1)(mu_n*r/r0) * (l + (l + 1) * ediv) +
                    (mu_n * dj_mu + (l + 1) * ediv * j_mu) *
                    l/r0 * (r/r0)**(l-1)
                )
            elif dr == 1:
                return (
                    (mu_n * dj_mu - l * ediv * j_mu) * (-l-1)/r0 *
                    (r/r0)**(-l-2)
                )
            else:
                return None  # TODO
        return phifn

    # -- States --
    def iter_states(self):
        """Returns an iterator for the set of basis states for which
        the boundary value problem has solutions
        """
        for state in self.modes():
            if self.mu_nl(state) is not None:
                yield state

    def iter_mu_n(self, l):
        for state in self.iter_states():
            l0, m0, n0 = self.get_nums(state)
            if l0 == l and m0 == 0:
                yield self.mu_nl(state), n0

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
            self.gamma * (self.R / nu) ** 2 *
            (muk * _j(l, d=1)(mu) + (l + 1) * ediv * _j(l)(mu)) /
            (l + ediv*(l+1))
        )

    def _u_unnormalized(self, state):
        r0 = self.R
        mu, nu = self.mu_nl(state), self.nu_nl(state)

        def ufunc(r, theta, phi):
            if state is self.vacuum_state():
                return np.zeros(3)
            elif r > r0:
                return np.zeros(3)  # TODO: Is this right?
            l, m, n = self.get_nums(state)
            pl = self._p_l(l=l, mu=mu, nu=nu)
            tl = self._t_l(l=l, mu=mu, nu=nu)
            u = np.zeros(3, dtype=complex)
            basis = basis_roca(theta, phi, l=l, m=m)
            ur = (
                -mu/r0 * _j(l, d=1)(mu * r/r0) +
                l*(l + 1)/r * pl * _g(l)(nu * r/r0) -
                l * tl/r0 * (r/r0)**(l-1)
            )
            u += ur * Y_lm(l, m)(theta, phi) * basis[0]
            if l == 0:
                return u
            u3 = -1j * np.sqrt(l*(l+1))/r * (
                -_j(l)(mu * r/r0) +
                pl * (_g(l)(nu * r/r0) + nu*r/r0 * _g(l, d=1)(nu * r/r0)) -
                tl * (r/r0)**l
            )
            u += u3 * basis[2]
            return u
        return ufunc

    def _u_norm(self, state):
        nums = self.get_nums(state)
        if state is self.vacuum_state():
            return 1
        elif nums in self._u_norm_dict:
            return self._u_norm_dict[nums]

        ufunc = self._u_unnormalized(state)

        def intfunc(r, theta, phi):
            u = ufunc(r, theta, phi)
            return lin.norm(u, ord=2)**2 * r**2 * np.sin(theta)
        ans_real = integ.nquad(
            intfunc, ranges=[(0, self.R), (0, np.pi), (0, 2 * np.pi)])[0]
        self._u_norm_dict[nums] = np.sqrt(ans_real)
        return self._u_norm(state=state)

    def u(self, state):
        def ufunc(r, theta, phi):
            return (
                self._u_unnormalized(state)(r, theta, phi) /
                self._u_norm(state)
            )
        return ufunc

    # TODO: Check correctness
    def phi_rad(self, state):
        """Returns the phi function with the proper normalization so as
        to match with the normalized u return by `u`
        """
        def phifn(r):
            l, m, n = self.get_nums(state)
            if r < self.R:
                epsinf = self.eps_inf_in
            else:
                epsinf = self.eps_inf_out
            return (
                4 * np.pi * self.alpha / epsinf /
                (l + (l + 1) * self._eps_div) / self._u_norm(state) *
                self.Phi_ln(state)(r)
            )
        return phifn

    def phi(self, state):
        """Returns the phi function with the proper normalization so as
        to match with the normalized u return by `u`
        """
        def phifn(r, theta, phi):
            l, m, n = self.get_nums(state)
            return self.phi_rad(state)(r) * Y_lm(l, m)(theta, phi)
        return phifn

    def grad_phi(self, state):
        """Returns the gradient of phi as a function of spatial coordinates
        (r, theat, phi). To take the gradient, we use the handy
        equation (A10) of Roca
        """
        def gphifunc(r, theta, phi):
            l, m, n = self.get_nums(state)
            basis = basis_roca(theta=theta, phi=phi, l=l, m=m)
            phi_r = self.Phi_ln(state)(r)
            phi_ang = Y_lm(l=l, m=m)(theta, phi)
            grad_phi_r = self.Phi_ln(state)(r, dr=1) * basis[0]
            grad_phi_ang = -1j/r * np.sqrt(l*(l+1)) * basis[2]
            return phi_r * grad_phi_ang + phi_ang * grad_phi_r
        return gphifunc

    # TODO: Check correctness
    def P(self, state):
        """Returns a function of spatial coordinates (r, theta, phi) which
        gives the polarization vector defined in Roca (4)
        """
        def pfunc(r, theta, phi):
            if r < self.R:
                epsinf = self.eps_inf_in
            else:
                epsinf = self.eps_inf_out
            return (
                self.alpha * self.u(state)(r, theta, phi) -
                (epsinf - 1)/np.pi/4 *
                self.grad_phi(state)(r, theta, phi)
            )
        return pfunc

    # -- Root finding --
    def nu_nl(self, state):
        if state is self.vacuum_state():
            return 0
        l, m, n = self.get_nums(state)
        if (l, n) in self._roots_nu:
            return self._roots_nu[l, n]
        else:
            return None  # TODO

    def mu_nl(self, state):
        if state is self.vacuum_state():
            return 0
        l, m, n = self.get_nums(state)
        if (l, n) in self._roots_mu:
            return self._roots_mu[l, n]
        else:
            return None  # TODO

    def _nu(self, mu):
        mu = complex(mu)
        return np.sqrt(self._beta_div2 * mu ** 2 - self.R ** 2 * self.gamma)

    def plot_root_function_mu(self, l, xdat, show=True):
        mudat_p = xdat
        mudat_m = -xdat
        fn = self._get_root_function(l=l)

        def rootfn(xy):
            mu, nu = xy
            f, g = fn(np.array([mu, nu]))
            return np.array([f, g])

        def fpr(mu):
            nu = self._nu(mu=mu)
            return rootfn(np.array([mu, nu]))[0]

        fig, ax = plt.subplots(1, 1)
        ax.axhline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(self.high_mu, ls='-', color='black', lw=1, alpha=.5)
        ax.axvline(-self.high_mu, ls='-', color='black', lw=1, alpha=.5)

        xdat = np.concatenate((-xdat, xdat))
        xdat.sort()
        mudat = np.concatenate((mudat_m, mudat_p))
        mudat.sort()

        ydat = np.array([fpr(x) for x in mudat])
        ydat /= lin.norm(ydat, ord=2)
        liner, = ax.plot(xdat, np.real(ydat), '-', color='red')
        linei, = ax.plot(xdat, np.imag(ydat), '-', color='blue')

        ylog = np.log(np.abs(ydat))
        ylog /= lin.norm(ylog, ord=2)
        ax.plot(xdat, ylog, '--', color='brown')

        ypr = np.array([self._nu(x) for x in mudat])
        ypr /= lin.norm(ypr, ord=2)

        # Region of imaginary Q
        x_min_imQ = max(min(xdat), -self.high_mu)
        x_max_imQ = min(max(xdat), self.high_mu)
        span_imQ = ax.axvspan(x_min_imQ, x_max_imQ, color='blue', alpha=.3)

        # Regions of real Q
        x_min_reQ = max(min(xdat), self.high_mu)
        x_max_reQ = max(xdat)
        span_reQ = ax.axvspan(x_min_reQ, x_max_reQ, color='red', alpha=.3)
        ax.axvspan(-x_max_reQ, -x_min_reQ, color='red', alpha=.3)

        for mu, n in self.iter_mu_n(l=l):
            if n < 0:
                pltzero = mu.real
            else:
                pltzero = mu.real
            ax.axvline(pltzero, ls='--', color='green', lw=1, alpha=.5)

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
            return (mu**2 * self._beta_div2 - nu**2) - self.R ** 2 * self.gamma
        return rf2
