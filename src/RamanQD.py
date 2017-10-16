"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import numpy as np
import itertools as it
from scipy import integrate as integ
from HamiltonianQD import J, K, Y_lm
import qutip as qt


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
        return self._get_matelt(
            j=j,
            H=self.get_Hj(
                j=j,
                mu_i=self._mu(j=j),  # TODO
                omega_i=omega_l, e_i=e_l),
            bra=bra, ket=ket
        )

    def matelt_Hjs(self, bra, ket, j, omega_s, e_s):
        return self._get_matelt(
            j=j,
            H=self.get_Hj(
                j=j,
                mu_i=self._mu(j=j),  # TODO
                omega_i=omega_s, e_i=e_s),
            bra=bra, ket=ket
        )

    def get_wavefunction(self, j, state, d_r=0):
        l, m, n = self.get_numbers(state=state, j=j)
        r0 = self.r_0
        Ylm = Y_lm(l=l, m=m)
        x = self.x(l=l, n=n, j=j)
        y = self.y(l=l, n=n, j=j)
        Aln = self.A(l=l, n=n, j=j)
        Bln = self.B(l=l, n=n, j=j)

        def psifn(r, theta, phi):
            ylm = Ylm(theta=theta, phi=phi)
            uj = self.u_j(j)(r=r)
            duj = self.u_j(j)(r=r, d_r=1)
            J0 = J(l=l+1/2)(z=x*r/r0)
            dJ0 = J(l=l+1/2, d=1)(z=x*r/r0)
            K0 = K(l=l+1/2)(z=y*r/r0)
            dK0 = K(l=l+1/2, d=1)(z=y*r/r0)
            if r <= r0:
                t1 = Aln * J0
                t3 = Aln * x * dJ0
            else:
                t1 = Bln * K0
                t3 = Bln * y * dK0
            if d_r == 0:  # No derivative
                t0 = ylm * uj / np.sqrt(r)
                return t0 * t1
            elif d_r == 1:  # One derivative
                t0 = ylm * (duj - uj / 2 / r**2) / np.sqrt(r)
                t2 = ylm * uj / np.sqrt(r) / r0
                return t0 * t1 + t2 * t3
            else:
                return 0  # TODO
        return psifn

    def get_Hj(self, j, mu_i, omega_i, e_i):
        def hj(state, r, theta, phi):
            t0 = abs(self._const.e) / mu_i * np.sqrt(2*np.pi/self.V/omega_i)
            r_unit = basis_spherical(theta=theta, phi=phi)[:, 0]
            dr_psi = self.get_wavefunction(j=j, state=state, d_r=1)
            t1 = np.dot(e_i, r_unit) * dr_psi
            t2 = 1j/r
            t3 = 0
            for a, b, c in it.permutations(range(3), r=3):
                eps = _lc_tensor(a, b, c)
                wf = self.get_wavefunction(j=j, state=self._L(c)(state))
                t3 += (
                    eps * omega_i[a] * r_unit[b] * wf(r=r, theta=theta, phi=phi)
                )
            return t0 * (t1 + t2 * t3)
        return hj

    def _get_matelt(self, j, H, bra, ket):
        psi = self.get_wavefunction(j, state=bra)

        def intfn(r, theta, phi):
            return r * np.sin(theta) * (
                np.conj(psi(r=r, theta=theta, phi=phi)) *
                H(state=ket, r=r, theta=theta, phi=phi)
            )
        return integ.nquad(
            func=intfn, ranges=[(0, np.inf), (0, np.pi), (0, 2*np.pi)]
        )[0]

    def _L(self, i):
        def lfun(state):
            return 0  # TODO
        return lfun

    def x(self, l, n, j):
        return 0  # TODO

    def y(self, l, n, j):
        return 0  # TODO

    def u_j(self, r):
        def ufn(r):
            return 1  # TODO
        return ufn

    def _mu(self, j):
        """Effective mass
            j=0 : Surrounding medium
            j=1 : Electron
            j=2 : Hole
        :param j: Labeling described above
        :return: The effective mass of the medium, electron, or hole
        """
        return 0  # TODO

    def _Jl_Kl(self, l, n, j):
        l = l+1/2
        return J(l=l)(self.x(l=l, n=n, j=j)), K(l=l)(self.y(l=l, n=n, j=j))

    def A(self, l, n, j):
        Jl, Kl = self._Jl_Kl(l=l, n=n, j=j)
        return 1/np.sqrt(abs(Jl)**2 + (Jl/Kl)**2 * abs(Kl)**2)

    def B(self, l, n, j):
        Jl, Kl = self._Jl_Kl(l=l, n=n, j=j)
        return Jl/Kl * self.A(l=l, n=n, j=j)
