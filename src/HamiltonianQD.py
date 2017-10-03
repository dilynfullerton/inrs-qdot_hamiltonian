"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definition used is based on "Electron Raman Scattering in Nanostructure,"
with the relevant Hamiltonian expression in Eq. (59)
"""
from math import sqrt, pi
from scipy.special import sph_harm, spherical_jn, spherical_in
from scipy import integrate


def _Y_lm(l, m):
    def y(theta, phi):
        return sph_harm(m, l, phi, theta)
    return y


def _j_l(l, deriv=False):
    def j(z):
        return spherical_jn(n=l, z=z, derivative=deriv)
    return j


def _i_l(l, deriv=False):
    def i(z):
        return spherical_in(n=l, z=z, derivative=deriv)
    return i


def _g_l(l, deriv=False):
    def g(z):
        if z**2 > 0:
            return _j_l(l=l, deriv=deriv)
        else:
            return _i_l(l=l, deriv=deriv)
    return g


class ModelSpaceQD:
    def __init__(self, n_max, l_max):
        self.n_max = n_max
        self.l_max = l_max

    def __iter__(self):
        return self.lnm()

    def b(self, l, n, m):
        pass

    def lnm(self):
        for l in range(self.l_max):
            for n in range(self.n_max):
                for m in range(-l, l+1):
                    yield l, n, m


class HamiltonianQD:
    def __init__(self, model_space):
        self.ms = model_space
        self._b = self.ms.b
        self.r_0 = 1  # TODO
        self.C_F = 1  # TODO
        self.epsilon_a_inf = 1  # TODO
        self.epsilon_b_inf = 1  # TODO
        self.omega_L = 1  # TODO
        self.omega_T = 1  # TODO
        self.beta_L = 1  # TODO
        self.beta_T = 1  # TODO
        self._gamma = (
            (self.omega_L**2 - self.omega_T**2) * (self.r_0/self.beta_T)**2
        )
        assert isinstance(self.ms, ModelSpaceQD)

    def h_int(self, r, theta, phi):
        h = 0
        for l, n, m in self.ms:
            h += self._h_int(l, n, m)(r, theta, phi) * self._x_b(l, n, m)
        return h

    def _h_int(self, l, n, m):
        Phi = self._Phi_ln(l, n)
        Y = _Y_lm(l, m)

        def _h_int_lnm(r, theta, phi):
            return self.r_0 * self.C_F * sqrt(4*pi/3) * Phi(r) * Y(theta, phi)
        return _h_int_lnm

    def _Phi_ln(self, l, n):
        mu_n = self._mu_n(n)
        nu_n = self._nu_n(n)
        r0 = self.r_0
        j_l = _j_l(l, deriv=False)
        dj_l = _j_l(l, deriv=True)
        eps = self.epsilon_b_inf / self.epsilon_a_inf

        def phi(r):
            phi = sqrt(r0) / self._norm_u_ln(l, n)
            if r <= r0:
                return phi * (
                    j_l(mu_n*r/r0) - (r/r0)**l *
                    (mu_n * dj_l(mu_n) + (l + 1) * eps * j_l(mu_n)) /
                    (l + (l + 1) * eps)
                )
            else:
                return phi * (
                    -(r/r0)**l * (mu_n * dj_l(mu_n) + l * eps * j_l(mu_n)) /
                    (l + (l + 1) * eps)
                )
        return phi

    def _mu_n(self, n):
        return 0  # TODO

    def _nu_n(self, n):
        return 0  # TODO

    def _F_l(self, l):
        g = _g_l(l=l, deriv=False)
        dg = _g_l(l=l, deriv=True)
        eps = self.epsilon_b_inf / self.epsilon_a_inf

        def retfunc(nu):
            return (
                self._gamma / nu**2 * l * (nu*dg(nu)-l*g(nu)) +
                (l+(l+1)*eps) * (nu*dg(nu)+g(nu))
            )
        return retfunc

    def _G_l(self, l):
        g = _g_l(l=l, deriv=False)
        dg = _g_l(l=l, deriv=True)
        eps = self.epsilon_b_inf / self.epsilon_a_inf

        def retfunc(nu):
            return (
                self._gamma / nu**2 * eps * (l*g(nu)-nu*dg(nu)) +
                (l+(l+1)*eps) * g(nu)
            )
        return retfunc

    def _norm_u_ln(self, l, n):
        j = _j_l(l, deriv=False)
        dj = _j_l(l, deriv=True)
        g = _g_l(l, deriv=False)
        dg = _g_l(l, deriv=True)
        mu = self._mu_n(n)
        nu = self._nu_n(n)
        p = self._p_l(l=l)(mu, nu)
        t = self._t_l(l=l)(mu, nu)
        r0 = self.r_0

        def ifn(r):
            rmu = mu*r/r0
            rnu = nu*r/r0
            return r**2 * (
                (
                    (-mu/r0)*dj(rmu) + l*(l+1)/r*p*g(rnu) -
                    t/r0*l*(r/r0)**(l-1)
                )**2 +
                l*(l+1)/r**2 * (
                    -j(rmu) + p/l * (g(rnu) + nu*r/r0 * dg(rnu)) -
                    t*(r/r0)**l
                )**2
            )
        return integrate.quad(func=ifn, a=0, b=r0)[0]

    def _p_l(self, l):
        j = _j_l(l, deriv=False)
        dj = _j_l(l, deriv=True)
        g = _g_l(l, deriv=False)
        dg = _g_l(l, deriv=True)

        def p(mu, nu):
            return (
                (mu * dj(mu) - l * j(mu)) / (l * g(nu) - nu * dg(nu))
            )
        return p

    def _t_l(self, l):
        j = _j_l(l=l, deriv=False)
        dj = _j_l(l=l, deriv=True)

        def t(mu, nu):
            eps = self.epsilon_b_inf / self.epsilon_a_inf
            return (
                self._gamma / nu**2 * (mu*dj(mu) + (l+1)*eps*j(mu)) /
                (l + eps*(l+1))
            )
        return t

    def _x_b(self, l, n, m):
        return self._b(l, n, m) + self._b(l, n, m).dag()
