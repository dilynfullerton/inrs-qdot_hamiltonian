"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import numpy as np
import itertools as it
from scipy import integrate as integ
from scipy import optimize as opt
from HamiltonianQD import J, K
from HamiltonianQD import plot_dispersion_function
from matplotlib import pyplot as plt
import qutip as qt


def _lc_tensor(a, b, c):
    if a < b < c or b < c < a or c < a < b:
        return 1
    elif a < c < b or b < a < c or c < b < a:
        return -1
    else:
        return 0


def basis_cartesian():
    return np.eye(3, 3)


def basis_pmz():
    isq2 = 1/np.sqrt(2)
    return np.array([[isq2,  1j*isq2, 0],
                     [isq2, -1j*isq2, 0],
                     [0,     0,       1]])


def basis_spherical(theta, phi):
    xph = -np.sin(phi)
    yph = np.cos(phi)
    zph = 0
    zr = np.cos(theta)
    zth = -np.sin(theta)
    xth = zr * yph
    yth = zr * -xph
    xr = -zth * yph
    yr = -zth * -xph
    return np.array([[xr, xth, xph],
                     [yr, yth, yph],
                     [zr, zth, zph]])


def _dirac_delta(x, x0=0):
    return 0  # TODO


def Pi(k, p, l, m):
    if k == 1 and p == 1:
        return -np.sqrt(
            (l + m + 2) * (l + m + 1) / 2 / (2 * l + 1) / (2 * l + 3))
    elif k == 1 and p == 2:
        return -Pi(k=1, p=1, l=l, m=-m)
    elif k == 1 and p == 3:
        return np.sqrt(
            (l + m + 1) * (l + m - 1) / (2 * l + 1) / (2 * l + 3)
        )
    elif k == 2 and p == 1:
        return -Pi(k=1, p=1, l=l-1, m=-m-1)
    elif k == 2 and p == 2:
        return -Pi(k=2, p=1, l=l, m=-m)
    elif k == 2 and p == 3:
        return -Pi(k=1, p=3, l=l-1, m=m)
    else:
        raise RuntimeError


class ModelSpaceElectronHolePair:
    def __init__(self, num_n):
        self.dim_n = num_n

    def iter_states_SP(self, num_l):
        for l, n in it.product(range(num_l), self.dim_n):
            for m in range(-l, l+1):
                yield (l, m, n), self.make_state_SP(l=l, m=m, n=n)

    def iter_states_EHP(self, num_l, num_l2=None):
        if num_l2 is None:
            num_l2 = num_l
        for l1, l2 in it.product(range(num_l), range(num_l2)):
            for m1, m2 in it.product(range(-l1, l1+1), range(-l2, l2+1)):
                for n1, n2 in it.product(self.dim_n, repeat=2):
                    yield (
                        (l1, m1, n1, l2, m2, n2),
                        self.make_state_EHP(l1, m1, n1, l2, m2, n2)
                    )

    def make_state_SP(self, l, m, n):
        """Make single-particle state
        """
        return qt.tensor(
            qt.fock_dm(self.dim_n, n),
            qt.spin_state(j=l, m=m, type='dm')
        )

    def make_state_EHP(self, l1, m1, n1, l2, m2, n2):
        return qt.tensor(
            self.make_state_SP(l1, m1, n1),
            self.make_state_SP(l2, m2, n2)
        )

    def zero_SP(self, l):
        return qt.qzero([self.dim_n, 2*l+1])

    def zero_EHP(self, l1, l2):
        return qt.tensor(self.zero_SP(l1), self.zero_SP(l2))

    def one_SP(self, l):
        return qt.qeye([self.dim_n, 2*l+1])

    def one_EHP(self, l1, l2):
        return qt.tensor(self.one_SP(l1), self.one_SP(l2))

    def am_SP(self, l):
        """Single particle annihilation operator
        """
        return qt.tensor(qt.destroy(self.dim_n), qt.qeye(2*l+1))

    def am_EHP(self, j, l1, l2=None):
        if j == 1:  # electron state
            return qt.tensor(
                self.am_SP(l1), self.one_SP(l1)
            )
        else:  # hole state
            if l2 is None:
                l2 = l1
            return qt.tensor(
                self.one_SP(l2), self.am_SP(l2)
            )

    def Lm_SP(self, l):
        return qt.tensor(qt.qeye(self.dim_n), qt.spin_Jm(j=l))

    def Lm_EHP(self, j, l1, l2=None):
        if j == 1:  # electron state
            return qt.tensor(
                self.Lm_SP(l1), self.one_SP(l1)
            )
        else:  # hole state
            if l2 is None:
                l2 = l1
            return qt.tensor(
                self.one_SP(l2), self.Lm_SP(l2)
            )


class RamanQD:
    def __init__(self):
        self.const = None  # TODO
        self.E_g = 0  # TODO
        self.r_0 = 0  # TODO
        self.mu_r = 0  # TODO
        self.V = 0  # TODO
        self.Gamma_f = 0  # TODO
        self.E_0 = 1 / 2 / self.mu_r / self.r_0**2  # Riera (151)
        self.delta_f = self.Gamma_f / self.E_0  # Riera (152)

        self._num_l = 0  # TODO
        self._num_n = 0  # TODO
        self.ms = ModelSpaceElectronHolePair(num_n=self._num_n)

        self._x = dict()  # l, n, j -> x
        self._y = dict()  # l, n, j -> y
        self._expected_roots_x = dict()  # TODO
        self._fill_roots_xy()

    def cross_section(self, omega_l, e_l, omega_s, e_s):
        cs = 0
        for p in range(4):
            cs += self._cross_section(omega_l=omega_l, e_l=e_l,
                                      omega_s=omega_s, e_s=e_s, p=p)
        return cs

    def _cross_section(self, omega_l, e_l, omega_s, e_s, p):
        t0 = (27/4 * self.sigma_0(omega_s=omega_s, omega_l=omega_l, e_l=e_l) *
              (omega_s/self.E_0)**2 * self.delta_f)
        if p == 0:
            t1 = 0
            for se, sh in it.product(self.states_SP(), repeat=2):
                M11 = self.Mj(1, 1, estate=se, hstate=sh, omega_s=omega_s)
                M12 = self.Mj(1, 2, estate=se, hstate=sh, omega_s=omega_s)
                M21 = self.Mj(2, 1, estate=se, hstate=sh, omega_s=omega_s)
                M22 = self.Mj(2, 2, estate=se, hstate=sh, omega_s=omega_s)
                g2 = self.g2(x1=self.x(j=1, state=se),
                             x2=self.x(j=2, state=sh),
                             omega_l=omega_l, omega_s=omega_s)
                t2 = self.S(p=0, e_s=e_s) / (g2**2 + self.delta_f**2)
                t1 += t2 * (
                    M11.real * M22.real + M12.real * M21.real +
                    M11.imag * M22.imag + M12.imag * M21.imag
                )
            return t0 * t1
        else:
            t1 = 0
            for se, sh in it.product(self.states_SP(), repeat=2):
                M1p = self.Mj(1, p, estate=se, hstate=sh, omega_s=omega_s)
                M2p = self.Mj(2, p, estate=se, hstate=sh, omega_s=omega_s)
                g2 = self.g2(x1=self.x(j=1, state=se),
                             x2=self.x(j=2, state=sh),
                             omega_l=omega_l, omega_s=omega_s)
                t2 = self.S(p=p, e_s=e_s) / (g2**2 + self.delta_f**2)
                t1 += abs(M1p + M2p)**2 * t2
            return t0 * t1

    def S(self, p, e_s):
        """Polarization S_p. See Riera (149)
        """
        pms = basis_pmz()
        if p == 0:
            return abs(np.dot(e_s, pms[:, 0])) * abs(np.dot(e_s, pms[:, 1]))
        elif p == 1:
            return abs(np.dot(e_s, pms[:, 0]))**2
        elif p == 2:
            return abs(np.dot(e_s, pms[:, 1]))**2
        elif p == 3:
            return abs(np.dot(e_s, pms[:, 2]))**2
        else:
            raise RuntimeError  # TODO

    def g2(self, x1, x2, omega_l, omega_s):
        """See Riera (150)
        """
        return (
            (omega_l - self.E_g) / self.E_0 - omega_s / self.E_0 -
            self.beta(1) * x1**2 - self.beta(2) * x2**2
        )

    def sigma_0(self, omega_s, omega_l, e_l):
        """See Riera (104)
        """
        return (
            (4*np.sqrt(2) * self.V * self.const.e**4 *
             abs(np.dot(e_l, self.p_cv(0)))**2 * self.eta(omega_s) *
             self.mu_r**(1/2) * self.E_0**(3/2)) /
            (9*np.pi**2 * self.const.mu_0**2 * self.eta(omega_l) *
             omega_l * omega_s)
        )

    def p_cv(self, x):
        return np.array([0, 0, 0])  # TODO

    def eta(self, omega):
        """Returns the refractive index for the given frequency
        """
        return 0  # TODO

    def states_SP(self):
        return self.ms.iter_states_SP(num_l=self._num_l)

    def states_EHP(self):
        return self.ms.iter_states_EHP(num_l=self._num_l)

    def get_numbers(self, state):
        return state[0]

    def Mj(self, j, p, estate, hstate, omega_s):
        l1i, m1i, n1i = self.get_numbers(state=estate)
        l2i, m2i, n2i = self.get_numbers(state=hstate)
        if j == 2:
            l1i, m1i, n1i, l2i, m2i, n2i = l2i, m2i, n2i, l1i, m1i, n1i
        t1 = 0
        for festate in self.states_SP():
            l1f, m1f, n1f = self.get_numbers(state=festate)
            # Continue if zero (check delta functions)
            if m1f != m2i or l1f != l2i:
                continue
            elif p == 1 and m1f != m1i + 1:
                continue
            elif p == 2 and m1f != m1i - 1:
                continue
            elif p == 3 and m1f != m1i:
                continue
            # Evaluate terms
            if l1f == l1i + 1:
                t11_num = Pi(1, p, l=l1i, m=m1i)  # TODO: verify correctness
            elif l1f == l1i - 1:
                t11_num = Pi(2, p, l=l1i, m=m1i)  # TODO: verify correctness
            else:
                continue
            a = self.get_index(state=festate, j=j)  # TODO I have no idea if this is what 'a' means
            t11_denom = (
                omega_s / self.E_0 + self.beta(j) *
                (self.x(l=l1i, n=n1i, j=j)**2 - self.x(l=l1f, n=n1f, j=j)**2)
                + 1j*self.Gamma(a, j)/self.E_0
            )
            t12 = (self.I(l1f, n1f, l1i, n1i, j=j) *
                   self.T(l1f, n1f, n2i, j=j, nu=1/2))
            t1 += t11_num / t11_denom * t12
        t0 = -self.beta(j)
        return complex(t0 * t1)

    def Gamma(self, a, j):
        return 0  # TODO

    def x(self, j, l=None, n=None, state=None):
        if l is not None and n is not None and (l, j, j) in self._x:
            return self._x[l, n, j]
        elif state is not None:
            l, m, n = self.get_numbers(state=state)
            return self.x(j=j, l=l, n=n)
        else:
            raise RuntimeError  # TODO

    def y(self, l, n, j):
        if (l, j, j) in self._y:
            return self._y[l, n, j]
        else:
            raise RuntimeError  # TODO

    def _fill_roots_xy(self):
        for j, l in it.product([1, 2], range(self._num_l)):
            for root, n in zip(self._solve_roots_xy(l=l, j=j),
                               range(self._num_n)):
                x, y = root
                assert not np.isnan(x)
                assert not np.isnan(y)
                self._x[l, n, j] = x
                self._y[l, n, j] = y

    def _solve_roots_xy(self, l, j):
        fun, jac = self.get_rootf_rootdf_xy(l=l, j=j)
        for x0y0, n in zip(self.expected_roots_xy(l=l, j=j)):
            assert opt.check_grad(func=fun, grad=jac, x0=x0y0)
            result = opt.root(fun=fun, jac=jac, x0=x0y0)
            if not result.success:
                print('FAILED')  # TODO
            else:
                yield result.x

    def _y2(self, x2, j):
        return (
            2 * self.mu_0(j) * self.r_0**2 * self.V_j(j) -
            self.mu_0(j) / self.mu_i(j) * x2
        )

    def plot_root_fn_x(self, l, j, xdat):
        rootfn = self.get_rootf_rootdf_xy(l=l, j=j)[0]

        def fpr(x):
            y = np.sqrt(self._y2(x2=x**2, j=j))
            return np.real(rootfn(np.array([x, y])))[0]

        def fmr(x):
            y = -np.sqrt(self._y2(x2=x**2, j=j))
            return np.real(rootfn(np.array([x, y])))[0]

        def fpi(x):
            y = np.sqrt(self._y2(x2=x**2, j=j))
            return np.imag(rootfn(np.array([x, y])))[0]

        def fmi(x):
            y = -np.sqrt(self._y2(x2=x**2, j=j))
            return np.imag(rootfn(np.array([x, y])))[0]

        def ix():
            for n in range(self._num_n):
                yield self.x(l, n, j)

        fig, ax = plt.subplots(1, 1)
        plot_dispersion_function(
            xdat=xdat, rootfn=fpr, iterfn=ix, fig=fig, ax=ax, show=False)
        plot_dispersion_function(
            xdat=xdat, rootfn=fpi, iterfn=ix, fig=fig, ax=ax, show=False)
        plot_dispersion_function(
            xdat=xdat, rootfn=fmr, iterfn=ix, fig=fig, ax=ax, show=False)
        plot_dispersion_function(
            xdat=xdat, rootfn=fmi, iterfn=ix, fig=fig, ax=ax, show=False)
        plt.show()
        return fig, ax

    def expected_roots_xy(self, l, j):
        for x0 in self._expected_roots_x[l, j]:
            y0 = np.sqrt(self._y2(x2=x0**2, j=j))
            yield np.array([x0, y0])
            yield np.array([x0, -y0])

    def get_rootf_rootdf_xy(self, l, j):
        rf1 = self._get_rootf1(l=l, j=j)
        rf2 = self._get_rootf2(j=j)

        def rf(xy):
            x, y = xy
            f1 = rf1(x, y)
            f2 = rf2(x, y)
            return np.array([f1, f2])

        def rdf(xy):
            x, y = xy
            f1x = rf1(x, y, partial_x=1)
            f1y = rf1(x, y, partial_y=1)
            f2x = rf2(x, y, partial_x=1)
            f2y = rf2(x, y, partial_y=1)
            return np.array([[f1x, f1y],
                             [f2x, f2y]])

        return rf, rdf

    def _get_rootf1(self, l, j):
        l = l+1/2
        mu0 = self.mu_0(j)
        mui = self.mu_i(j)

        def rf1(x, y, partial_x=0, partial_y=0):
            if partial_x == 0 and partial_y == 0:
                return (
                    mu0 * x * J(l, d=1)(x) * K(l)(y) -
                    mui * y * J(l)(x) * K(l, d=1)(y)
                )
            elif (partial_x, partial_y) == (1, 0):
                return (
                    mu0 * (J(l, d=1)(x) + x * J(l, d=2)(x)) * K(l)(y) -
                    mui * y * J(l, d=1)(x) + K(l, d=1)(y)
                )
            elif (partial_x, partial_y) == (0, 1):
                return (
                    mu0 * x * J(l, d=1)(x) * K(l, d=1)(y) -
                    mui * (K(l, d=1)(y) + y * K(l, d=2)(y)) * J(l)(x)
                )
            else:
                raise RuntimeError  # TODO
        return rf1

    def _get_rootf2(self, j):
        mu0 = self.mu_0(j)
        mui = self.mu_i(j)
        vj = self.V_j(j)

        def _fx2(x, deriv=0):
            if deriv == 0:
                return x**2
            elif deriv == 1:
                return 2*x
            elif deriv == 2:
                return 2
            elif deriv > 2:
                return 0

        def rf2(x, y, partial_x=0, partial_y=0):
            if partial_x == 0 and partial_y == 0:
                tc = 2 * mu0 * self.r_0**2 * vj
            else:
                tc = 0
            # TODO: verify formula
            return (
                _fx2(y, deriv=partial_y) +
                mu0 / mui * _fx2(x, deriv=partial_x) - tc)

        return rf2

    def V_j(self, j):
        return 0  # TODO

    def mu_0(self, j):
        return 0  # TODO

    def mu_i(self, j):
        return 0  # TODO

    def beta(self, j):
        """See Riera (92)
        """
        return self.mu_r / self.mu_i(j)

    def get_index(self, state, j):
        return 0  # TODO

    def I(self, lf, nf, li, ni, j, nu=1/2):
        r0 = self.r_0
        xi = self.x(l=li, n=ni, j=j)
        xf = self.x(l=lf, n=nf, j=j)
        yi = self.y(l=li, n=ni, j=j)
        yf = self.y(l=lf, n=nf, j=j)
        Jli = J(l=li+nu)
        Jlf = J(l=lf+nu)
        Kli = K(l=li+nu)
        Klf = K(l=lf+nu)

        def ifn_j(r):
            return Jli(z=xi*r/r0) * Jlf(z=xf*r/r0) * r
        ans_j = integ.quad(ifn_j, a=0, b=self.r_0)[0]

        def ifn_k(r):
            return Kli(z=yi*r/r0) * Klf(z=yf*r/r0) * r
        ans_k = integ.quad(ifn_k, a=self.r_0, b=np.inf)[0]

        return (
            self.A(l=li, n=ni, j=j) * self.A(l=lf, n=nf, j=j) * ans_j +
            self.B(l=li, n=ni, j=j) * self.B(l=lf, n=nf, j=j) * ans_k
        )

    def T(self, l, na, nb, j, nu=1/2):
        return self.I(lf=l, nf=nb, li=l, ni=na, j=j, nu=nu)

    def _Jl_Kl(self, l, n, j):
        l = l+1/2
        return J(l=l)(self.x(l=l, n=n, j=j)), K(l=l)(self.y(l=l, n=n, j=j))

    def A(self, l, n, j):
        Jl, Kl = self._Jl_Kl(l=l, n=n, j=j)
        return 1/np.sqrt(abs(Jl)**2 + (Jl/Kl)**2 * abs(Kl)**2)

    def B(self, l, n, j):
        Jl, Kl = self._Jl_Kl(l=l, n=n, j=j)
        return Jl/Kl * self.A(l=l, n=n, j=j)
