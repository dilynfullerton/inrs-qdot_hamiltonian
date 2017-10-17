"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import numpy as np
import itertools as it
from scipy import integrate as integ
from scipy import optimize as opt
from HamiltonianQD import J, K


def _lc_tensor(a, b, c):
    if a < b < c or b < c < a or c < a < b:
        return 1
    elif a < c < b or b < a < c or c < b < a:
        return -1
    else:
        return 0


# def _intsphere_Ylm(l, m):
#     intfn = Y_lm(l, m)
#     # TODO make sure this is right
#     return integ.dblquad(
#         func=intfn, a=0, b=2*np.pi, gfun=lambda x: 0, hfun=lambda x: np.pi
#     )[0]
#

def basis_cartesian():
    return np.eye(3, 3)


def basis_pmz():
    isq2 = 1/np.sqrt(2)
    return np.array([[isq2,  1j*isq2, 0],
                     [isq2, -1j*isq2, 0],
                     [   0,        0, 1]])


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


class Raman:
    def __init__(self, V, energies, states, istate, fstates,
                 lifetimes_elec, lifetimes_hole, refidx_func=None):
        self.V = V
        self.eta = refidx_func
        self.energies = energies
        self.states = states
        self.istate = istate
        self.fstates = fstates
        self._Gamma1 = lifetimes_elec
        self._Gamma2 = lifetimes_hole

    def cross_section(self, omega_s, e_s, omega_l, e_l, omega_q=None, e_q=None,
                      n_s=None, n_l=None, phonon_assist=False):
        fact = (
            self.V**2 * omega_s**2 / 8 / np.pi**3 *
            self.W(omega_s=omega_s, e_s=e_s, omega_l=omega_l, e_l=e_l,
                   omega_q=omega_q, e_q=e_q, phonon_assist=phonon_assist)
        )
        if n_s is not None and n_l is not None:
            return (
                fact * n_s / n_l
            )
        elif self.eta is not None:
            return fact * self.eta(omega_s) / self.eta(omega_l)
        else:
            raise RuntimeError  # TODO

    def W(self, omega_s, e_s, omega_l, e_l,
          omega_q=None, e_q=None, phonon_assist=False):
        w = 0
        for f in self.fstates:
            m1 = self.Mj(f=f, j=1,
                         omega_s=omega_s, e_s=e_s,
                         omega_l=omega_l, e_l=e_l,
                         omega_q=omega_q, e_q=e_q,
                         phonon_assist=phonon_assist)
            m2 = self.Mj(f=f, j=2,
                         omega_s=omega_s, e_s=e_s,
                         omega_l=omega_l, e_l=e_l,
                         omega_q=omega_q, e_q=e_q,
                         phonon_assist=phonon_assist)
            w += (
                2*np.pi * abs(m1 + m2)**2 *
                _dirac_delta(self.diff_Ef_Ei(f=f, i=self.istate))
            )
        return w

    def diff_Ef_Ei(self, f, i):
        return 0  # TODO

    def Mj(self, f, j, omega_s, e_s, omega_l, e_l,
           omega_q=None, e_q=None, phonon_assist=False):
        # j=1 : electron
        # j=2 : hole
        if not phonon_assist:
            return self._Mj(f=f, j=j,
                            omega_s=omega_s, e_s=e_s,
                            omega_l=omega_l, e_l=e_l)
        elif omega_q is not None and e_q is not None:
            return self._Mj_phonon_assist(f=f, j=j,
                                          omega_s=omega_s, e_s=e_s,
                                          omega_l=omega_l, e_l=e_l,
                                          omega_q=omega_q, e_q=e_q)

    def _Mj(self, f, j, omega_s, e_s, omega_l, e_l):
        i = self.istate
        m = 0
        E_ia = self.Ej(j) - self.Ej(j, p=1) + omega_s
        E_ib = E_ia - omega_s - omega_l
        for a in self.states:
            Gam = self.Gamma(state=a, j=j)
            m += (
                self.matelt_Hjs(f, a, j=j, omega_s=omega_s, e_s=e_s) *
                self.matelt_Hjl(a, i, j=j, omega_l=omega_l, e_l=e_l) /
                (E_ia + 1j * Gam)
            )
            m += (
                self.matelt_Hjl(f, a, j=j, omega_l=omega_l, e_l=e_l) *
                self.matelt_Hjs(a, i, j=j, omega_s=omega_s, e_s=e_s) /
                (E_ib + 1j * Gam)
            )
        return m

    def _Mj_phonon_assist(self, f, j, omega_s, e_s, omega_l, e_l,
                          omega_q, e_q):
        i = self.istate
        m = 0
        E_ia = (self.Ej(j) - self.Ej(j, p=2) + omega_s + omega_q)
        E_ib = (self.Ej(j) - self.Ej(j, p=1) + omega_s)
        E_ic = E_ib - omega_s + omega_q
        for a, b in it.product(self.states, repeat=2):
            denom_a = (E_ia + 1j * self.Gamma(state=a, j=j))
            Gam_b = self.Gamma(state=b, j=j)
            m += (
                self.matelt_Hjs(f, b, j=j, omega_s=omega_s, e_s=e_s) *
                self.matelt_Hjph(b, a, j=j, omega_q=omega_q, e_q=e_q) *
                self.matelt_Hjl(a, i, j=j, omega_l=omega_l, e_l=e_l) /
                denom_a / (E_ib + 1j * Gam_b)
            )
            m += (
                self.matelt_Hjph(f, b, j=j, omega_q=omega_q, e_q=e_q) *
                self.matelt_Hjs(b, a, j=j, omega_s=omega_s, e_s=e_s) *
                self.matelt_Hjl(a, i, j=j, omega_l=omega_l, e_l=e_l) /
                denom_a / (E_ic + 1j * Gam_b)
            )
        return m

    def Ej(self, j, p=0):
        return 0  # TODO

    def Gamma(self, state, j):
        if j == 1:
            return self._Gamma1[state]
        elif j == 2:
            return self._Gamma2[state]
        else:
            raise RuntimeError  # TODO

    def get_numbers(self, state, j):
        return 0  # TODO

    def matelt_Hjs(self, bra, ket, j, omega_s, e_s):
        raise NotImplementedError  # TODO

    def matelt_Hjl(self, bra, ket, j, omega_l, e_l):
        raise NotImplementedError  # TODO

    def matelt_Hjph(self, bra, ket, j, omega_q, e_q):
        raise NotImplementedError  # TODO


class RamanQD:
    def __init__(self):
        self.sigma_0 = 0  # TODO
        self.E_0 = 0  # TODO
        self.delta_f = 0  # TODO
        self.g = 0  # TODO
        self.beta_1 = 0  # TODO
        self.r_0 = 0  # TODO

    def cross_section(self, omega_s):
        return sum(
            (self._cross_section(omega_s=omega_s, p=p) for p in range(4)), 0)

    def _cross_section(self, omega_s, p):
        t0 = 27/4 * self.sigma_0 * (omega_s/self.E_0)**2 * self.delta_f
        if p == 0:
            t1 = 0
            for se, sh in it.product(self.estates(), self.hstates()):
                M11 = self.Mj(1, 1, estate=se, hstate=sh, omega_s=omega_s)
                M12 = self.Mj(1, 2, estate=se, hstate=sh, omega_s=omega_s)
                M21 = self.Mj(2, 1, estate=se, hstate=sh, omega_s=omega_s)
                M22 = self.Mj(2, 2, estate=se, hstate=sh, omega_s=omega_s)
                t2 = self.S(0) / (self.g**4 + self.delta_f**2)
                t1 += t2 * (
                    M11.real * M22.real + M12.real * M21.real +
                    M11.imag * M22.imag + M12.imag * M21.imag
                )
            return t0 * t1
        else:
            t1 = 0
            for se, sh in it.product(self.estates(), self.hstates()):
                M1p = self.Mj(1, p, estate=se, hstate=sh, omega_s=omega_s)
                M2p = self.Mj(2, p, estate=se, hstate=sh, omega_s=omega_s)
                t2 = self.S(p) / (self.g**4 + self.delta_f**2)
                t1 += abs(M1p + M2p)**2 * t2
            return t0 * t1

    def S(self, p):
        return 0  # TODO

    def states(self, j):
        return ()  # TODO

    def estates(self):
        return self.states(j=1)

    def hstates(self):
        return self.states(j=2)

    def get_numbers(self, state):
        return 0, 0, 0  # TODO

    def Mj(self, j, p, estate, hstate, omega_s):
        l1i, m1i, n1i = self.get_numbers(state=estate)
        l2i, m2i, n2i = self.get_numbers(state=hstate)
        if j == 2:
            l1i, m1i, n1i, l2i, m2i, n2i = l2i, m2i, n2i, l1i, m1i, n1i
        t1 = 0
        for festate in self.estates():
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
            t11_denom = (
                omega_s / self.E_0 + self.beta(j) *
                (self.x(l=l1i, n=n1i, j=j)**2 - self.x(l=l1f, n=n1f, j=j)**2)
            )
            a = self.get_index(state=festate, j=j)  # TODO I have no idea if this is what 'a' means
            if a == 1:
                t11_denom += 1j
            t12 = (self.I(l1f, n1f, l1i, n1i, j=j) *
                   self.T(l1f, n1f, n2i, j=j, nu=1/2))
            t1 += t11_num / t11_denom * t12
        t0 = -self.beta(j)
        return complex(t0 * t1)

    def x(self, l, n, j):
        return 0  # TODO

    def y(self, l, n, j):
        return 0  # TODO

    def _solve_roots_xy(self, l, j):
        fun, jac = self.get_rootf_rootdf_xy(l=l, j=j)
        roots = []
        for x0y0 in self.expected_roots_xy(l=l, j=j):
            assert opt.check_grad(func=fun, grad=jac, x0=x0y0)
            result = opt.root(fun=fun, jac=jac, x0=x0y0)
            if not result.success:
                print('FAILED')  # TODO
            else:
                roots.append(result.x)
        return roots

    def expected_roots_xy(self, l, j):
        return []  # TODO

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
        return 0  # TODO

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
