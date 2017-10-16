"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import numpy as np
import itertools as it
from scipy import integrate as integ
from HamiltonianQD import J, K


BASIS_e_x = np.array([1, 0, 0])
BASIS_e_y = np.array([0, 1, 0])
BASIS_e_z = np.array([0, 0, 1])
BASIS_e_p = 1/np.sqrt(2) * (BASIS_e_x + 1j * BASIS_e_y)
BASIS_e_m = 1/np.sqrt(2) * (BASIS_e_x - 1j * BASIS_e_y)


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

    def get_state(self, state, j):
        return 0  # TODO

    def matelt_Hjs(self, bra, ket, j, omega_s, e_s):
        raise NotImplementedError  # TODO

    def matelt_Hjl(self, bra, ket, j, omega_l, e_l):
        raise NotImplementedError  # TODO

    def matelt_Hjph(self, bra, ket, j, omega_q, e_q):
        raise NotImplementedError  # TODO


class RamanQD(Raman):
    def __init__(self, r_0, constants, V, energies, states, istate, fstates,
                 lifetimes_elec, lifetimes_hole, refidx_func=None):
        super(RamanQD, self).__init__(
            V=V, energies=energies, states=states, istate=istate,
            fstates=fstates, lifetimes_elec=lifetimes_elec,
            lifetimes_hole=lifetimes_hole, refidx_func=refidx_func,
        )
        self.r_0 = r_0
        self._const = constants

    def matelt_Hjl(self, bra, ket, j, omega_l, e_l):
        n1i, l1i, m1i = self.get_state(state=ket, j=1)
        n2i, l2i, m2i = self.get_state(state=ket, j=2)
        n1f, l1f, m1f = self.get_state(state=bra, j=1)
        n2f, l2f, m2f = self.get_state(state=bra, j=2)
        if j == 1 and l1f == l2i and m1f == m2i:
            t0 = self.T(l=l1f, na=n1f, nb=n2i, nu=1/2)
        elif j == 2 and l1i == l2f and m1i == m2f:
            t0 = self.T(l=l2f, na=n2f, nb=n1i, nu=1/2)
        else:
            return 0
        t1 = (
            abs(self._const.e) / self._mu(0) *
            np.sqrt(2*np.pi / self.V / omega_l) *
            np.dot(e_l, self._p_cv(0))
        )
        return t0 * t1

    def matelt_Hjs(self, bra, ket, j, omega_s, e_s):
        return sum(
            (self._matelt_Hjs_p(bra, ket, j=j, omega_s=omega_s, e_s=e_s, p=i)
             for i in range(1, 4)),
            0
        )

    def _matelt_Hjs_p(self, bra, ket, j, omega_s, e_s, p):
        ni, li, mi = self.get_state(ket, j=j)
        nf, lf, mf = self.get_state(bra, j=j)
        if p == 1 and mf == mi + 1:
            t0 = np.dot(e_s, BASIS_e_p)
        elif p == 2 and mf == mi - 1:
            t0 = np.dot(e_s, BASIS_e_m)
        elif p == 3 and mf == mi:
            t0 = np.dot(e_s, BASIS_e_z)
        else:
            return 0
        if lf == li + 1:
            t1 = Pi(1, p, l=li, m=mi)
        elif lf == li - 1:
            t1 = Pi(2, p, l=li, m=mi)
        else:
            return 0
        t2 = (
            1j*(-1)**j * abs(self._const.e) / self.r_0 / self._mu(j=j) *
            np.sqrt(2*np.pi / self.V / omega_s)
        )
        t3 = self.I(lf=lf, nf=nf, li=li, ni=ni)
        return t0 * t1 * t2 * t3

    def I(self, lf, nf, li, ni, nu=1/2):
        r0 = self.r_0
        xi = self.x(l=li, n=ni)
        xf = self.x(l=lf, n=nf)
        yi = self.y(l=li, n=ni)
        yf = self.y(l=lf, n=nf)
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
            self.A(l=li, n=ni) * self.A(l=lf, n=nf) * ans_j +
            self.B(l=li, n=ni) * self.B(l=lf, n=nf) * ans_k
        )

    def _mu(self, j):
        """Effective mass
            j=0 : Surrounding medium
            j=1 : Electron
            j=2 : Hole
        :param j: Labeling described above
        :return: The effective mass of the medium, electron, or hole
        """
        return 0  # TODO

    def _p_cv(self, x):
        return np.zeros(3)  # TODO

    def T(self, l, na, nb, nu):
        return self.I(lf=l, nf=nb, li=l, ni=na, nu=nu)

    def x(self, l, n):
        return 0  # TODO

    def y(self, l, n):
        return 0  # TODO

    def _Jl_Kl(self, l, n):
        l = l+1/2
        return J(l=l)(self.x(l=l, n=n)), K(l=l)(self.y(l=l, n=n))

    def A(self, l, n):
        Jl, Kl = self._Jl_Kl(l=l, n=n)
        return 1/np.sqrt(abs(Jl)**2 + (Jl/Kl)**2 * abs(Kl)**2)

    def B(self, l, n):
        Jl, Kl = self._Jl_Kl(l=l, n=n)
        return Jl/Kl * self.A(l=l, n=n)
