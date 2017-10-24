"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import itertools as it
from collections import namedtuple

import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from scipy import integrate as integ
from scipy import linalg as lin

from helper_functions import J, K, basis_pmz, basis_spherical, threej
from root_solver_2d import RootSolverComplex2d

__all__ = ['ModelSpaceElectronHolePair', 'RamanQD']


def Lambda(lbj, mbj, lcj, mcj, la, ma):
    return (
        (-1)**(lbj % 2) *
        np.sqrt((2*lbj+1)*(2*lcj+1)*(2*la+1)/4/np.pi) *
        threej(lbj, la, lcj, -mbj, ma, mcj) *
        threej(lbj, la, lcj, 0, 0, 0)
    )


def Pi(j, p, laj, maj):
    if j == 1 and p == 1:
        return -np.sqrt(
            (laj + maj + 2) * (laj + maj + 1) /
            2 / (2 * laj + 1) / (2 * laj + 3)
        )
    elif j == 1 and p == 2:
        return -Pi(j=1, p=1, laj=laj, maj=-maj)
    elif j == 1 and p == 3:
        return np.sqrt(
            (laj - maj + 1) * (laj + maj + 1) / (2 * laj + 1) / (2 * laj + 3)
        )
    elif j == 2 and p == 1:
        return -Pi(j=1, p=1, laj=laj-1, maj=-maj-1)
    elif j == 2 and p == 2:
        return -Pi(j=2, p=1, laj=laj, maj=-maj)
    elif j == 2 and p == 3:
        return Pi(j=1, p=3, laj=laj-1, maj=maj)
    else:
        raise RuntimeError


class ModelSpaceElectronHolePair:
    def __init__(self, num_n):
        self.dim_n = num_n

    def iter_states_SP(self, num_l):
        for l, n in it.product(range(num_l), range(self.dim_n)):
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


class RamanQD(RootSolverComplex2d):
    def __init__(self, hamiltonian, unit_cell, refidx,
                 V1, V2, E_g, mu_i1, mu_i2, mu_01, mu_02,
                 free_electron_mass,
                 Gamma_f, Gamma_a1, Gamma_a2, Gamma_b1, Gamma_b2,
                 expected_roots_x_lj, num_n=None, num_l=None,
                 verbose=False):
        # Model constants
        self.num_n = num_n
        self.num_l = num_l
        self.hamiltonian = hamiltonian
        self.ms = ModelSpaceElectronHolePair(num_n=self.num_n)
        if self.num_n is None:
            self.num_n = self.hamiltonian.num_n
        if self.num_l is None:
            self.num_l = self.hamiltonian.num_l

        # j-dependent physical constants
        self._V_j = [V1, V2]
        self._Gamma_aj = [Gamma_a1, Gamma_a2]
        self._Gamma_bj = [Gamma_b1, Gamma_b2]
        self._mu_ij = [mu_i1, mu_i2]  # QDOT effective mass?
        self._mu_0j = [mu_01, mu_02]  # Environment effective mass

        # Physical constants
        self._a_matrix = unit_cell
        self.E_g = E_g  # Gap energy
        self.mu_0 = free_electron_mass  # Free electron mass
        self.Gamma_f = Gamma_f

        # Functions
        self.eta = refidx  # omega -> refractive index

        # Derived/inherited physical constants
        self.e = abs(self.hamiltonian.electron_charge)
        self.r_0 = self.hamiltonian.r_0
        self.mu_r = 1/(1 / self.mu_ij(1) + 1 / self.mu_ij(2))  # Riera (92)
        self.E_0 = 1 / 2 / self.mu_r / self.r_0**2  # Riera (151)
        self.delta_f = self.Gamma_f / self.E_0  # Riera (152)
        self.V = 4/3 * np.pi * self.r_0**2  # QD Volume
        self._p_cv0 = None

        # Root finding
        self._x = dict()  # l, n, j -> x
        self._y = dict()  # l, n, j -> y
        self._expected_roots_x = expected_roots_x_lj  # l, j -> x0
        self._fill_roots_xy()

        # Matrix elements
        self._mat_elts = dict()
        self._mat_elts_ph = dict()
        self._intS = dict()

        # Other
        self.verbose = verbose

    def _print(self, string):
        if self.verbose:
            print('  >> {}'.format(string))

    # -- Raman cross section --
    def differential_raman_efficiency(self, omega_l, e_l, omega_s, e_s=None,
                                      phonon_assist=False):
        """Differential Raman scattering efficiency.
                1/sigma_0 * d^2 sigma / (d Omega d omega_s)
        If e_s is None, this is integrated over the solid angle Omega to give
                1/sigma_0 * (d sigma) / (d omega_s)
        See Riera (201)
        for Raman cross section or (215) for phonon-assisted tranistions.
        Note: I have removed the factor of sigma_0 (or sigma_ph) from this
        calculation. If the user wants to see the true cross section, the
        same should multilply the result of this function with that of
        'sigma_0' (or 'sigma_ph' for phonon-assist)
        """
        self._print('Obtaining cross section for')
        self._print('  omega_l = {}'.format(omega_l))
        self._print('  omega_s = {}'.format(omega_s))
        self._print('  e_l = {}'.format(e_l))
        self._print('  e_s = {}'.format(e_s))
        cs = 0
        for p in range(4):
            self._print('Calculating cross section for p = {}'.format(p))
            cs += self._cross_section_p(
                omega_l=omega_l, omega_s=omega_s, e_s=e_s, p=p,
                phonon_assist=phonon_assist
            )
        if not phonon_assist:
            t0 = (27/4 * (omega_s/self.E_0)**2 * self.delta_f)
        else:
            t0 = 8/3 * (omega_s/self.E_0)**2 * self.delta_f
        ans = t0 * cs
        self._print('Cross section = {}'.format(ans))
        self._print('Done\n')
        return ans

    def _cross_section_p(self, omega_l, omega_s, e_s, p, phonon_assist=False):
        """See Riera (202, 203) [(216, 217) for phonon-assist]. Note that
        the computation of the common prefactor containing sigma_0, which is
        present in Riera (202, 203) has been moved to cross_section to
        avoid multiple computations
        """
        if not phonon_assist:
            return self._cross_section_p_reg(omega_l=omega_l, omega_s=omega_s,
                                             e_s=e_s, p=p)
        else:
            return self._cross_section_p_phonon(
                omega_l=omega_l, omega_s=omega_s, e_s=e_s, p=p)

    def _cross_section_p_phonon(self, omega_l, omega_s, e_s, p):
        t1 = 0
        for s in self.states_SP_phonon():
            for s1, s2 in it.product(self.states_SP(), repeat=2):
                l, m, n = self.get_numbers(state=s)
                # TODO: Figure out why omega_q is so large for (l, n) != (0, 0)
                omega_q = abs(self.hamiltonian.omega(l=l, n=n))  # TODO: is this right?
                G2 = self.G2(omega_l=omega_l, omega_s=omega_s, omega_q=omega_q,
                             xa1=self.x(j=1, state=s1), xa2=self.x(j=2, state=s2))
                t2 = self.S(p=p, e_s=e_s) / (G2**2 + self.delta_f**2)

                if p == 0 and m == 0:
                    M11 = self.Mph_j(1, 1, states=[s, s1, s2],
                                     omega_s=omega_s, omega_q=omega_q)
                    M12 = self.Mph_j(1, 2, states=[s, s1, s2],
                                     omega_s=omega_s, omega_q=omega_q)
                    M21 = self.Mph_j(2, 1, states=[s, s1, s2],
                                     omega_s=omega_s, omega_q=omega_q)
                    M22 = self.Mph_j(2, 2, states=[s, s1, s2],
                                     omega_s=omega_s, omega_q=omega_q)
                    t1 += t2 * (
                        M11.real * M22.real + M12.real * M21.real +
                        M11.imag * M22.imag + M12.imag * M21.imag
                    )
                elif p > 0:
                    M1p = self.Mph_j(1, p, states=[s, s1, s2],
                                     omega_s=omega_s, omega_q=omega_q)
                    M2p = self.Mph_j(2, p, states=[s, s1, s2],
                                     omega_s=omega_s, omega_q=omega_q)
                    t1 += abs(M1p + M2p)**2 * t2
        return t1

    def _cross_section_p_reg(self, omega_l, omega_s, e_s, p):
        """See Riera (202, 203). Note that
        the computation of the common prefactor containing sigma_0, which is
        present in Riera (202, 203) has been moved to cross_section to
        avoid multiple computations
        """
        t1 = 0
        for s1, s2 in it.product(self.states_SP(), repeat=2):
            g2 = self.g2(x1=self.x(j=1, state=s1), x2=self.x(j=2, state=s2),
                         omega_l=omega_l, omega_s=omega_s)
            t2 = self.S(p=p, e_s=e_s) / (g2**2 + self.delta_f**2)
            if p == 0:
                M11 = self.Mj(1, 1, states=[s1, s2], omega_s=omega_s)
                M12 = self.Mj(1, 2, states=[s1, s2], omega_s=omega_s)
                M21 = self.Mj(2, 1, states=[s1, s2], omega_s=omega_s)
                M22 = self.Mj(2, 2, states=[s1, s2], omega_s=omega_s)
                t1 += t2 * (
                    M11.real * M22.real + M12.real * M21.real +
                    M11.imag * M22.imag + M12.imag * M21.imag
                )
            else:
                M1p = self.Mj(1, p, states=[s1, s2], omega_s=omega_s)
                M2p = self.Mj(2, p, states=[s1, s2], omega_s=omega_s)
                t1 += abs(M1p + M2p)**2 * t2
        return t1

    def Mph_j(self, j, p, states, omega_s, omega_q):
        """See Riera (218, 219)
        """
        anums0, anums1, anums2 = [self.get_numbers(s) for s in states]

        # If stored in dict, use that
        if (j, p, anums0, anums1, anums2) in self._mat_elts_ph:
            mm = 0
            fracs = self._mat_elts_ph[j, p, anums0, anums1, anums2]
            for num1, denom1, num2, denom2 in fracs:
                mm += (
                    num1 / (denom1 + omega_s/self.E_0 + omega_q/self.E_0) *
                    num2 / (denom2 + omega_s/self.E_0)
                )
            return mm
        else:
            self._set_Mph_j(j=j, p=p,
                            anums0=anums0, anums1=anums1, anums2=anums2)
            return self.Mph_j(j=j, p=p, states=states,
                              omega_s=omega_s, omega_q=omega_q)

    def _set_Mph_j(self, j, p, anums0, anums1, anums2):
        anums = [anums1, anums2]
        j_ = j % 2 + 1
        la, ma, na = anums0
        laj, maj, naj = anums[j-1]
        laj_, maj_, naj_ = anums[j_-1]
        x2aj = self.x(j=j, l=laj, n=naj)**2
        fracs = list()
        for state_b in self.states_SP():
            lbj, mbj, nbj = self.get_numbers(state_b)
            # Check delta functions
            if p == 1 and mbj != maj + 1:
                continue
            elif p == 2 and mbj != maj - 1:
                continue
            elif p == 3 and mbj != maj:
                continue
            # Get first numerator and second denominator
            beta = self.beta(j)
            if lbj == laj + 1:
                first_numerator = beta * Pi(j, 1, laj=laj, maj=maj)
            elif lbj == laj - 1:
                first_numerator = beta * Pi(j, 2, laj=laj, maj=maj)
            else:
                continue
            x2bj = self.x(j=j, l=lbj, n=nbj)**2
            second_denominator = (beta * (x2aj - x2bj) +
                                  1j * self.Gamma_bj(j) / self.E_0)
            for state_c in self.states_SP():
                lcj, mcj, ncj = self.get_numbers(state_c)
                # Check delta function
                if lcj != laj_:
                    continue
                # Get first denominator and second numerator
                x2cj = self.x(j=j, l=lcj, n=ncj)**2
                first_denominator = (beta * (x2aj - x2cj) +
                                     1j * self.Gamma_aj(j) / self.E_0)
                second_numerator = (
                    self.II(lbj, nbj, lcj, ncj, la, na, j=j) *
                    Lambda(lbj, mbj, lcj, mcj, la, ma) *
                    self.I(lbj, nbj, laj, naj, j=j) *
                    self.T(lcj, ncj, naj_, j=j)
                )
                fracs.append(
                    (first_numerator, first_denominator,
                     second_numerator, second_denominator)
                )
        self._mat_elts_ph[j, p, anums0, anums1, anums2] = fracs

    def Mj(self, j, p, states, omega_s):
        """See Riera (204) and (205)
        """
        anums1, anums2 = [self.get_numbers(s) for s in states]

        # If stored in dict, use that
        if (j, p, anums1, anums2) in self._mat_elts:
            mm = 0
            fracs = self._mat_elts[j, p, anums1, anums2]
            for num, denom in fracs:
                mm += num / (denom + omega_s/self.E_0)
            return mm
        else:
            self._set_M_j(j=j, p=p, anums1=anums1, anums2=anums2)
            return self.Mj(j=j, p=p, states=states, omega_s=omega_s)

    def _set_M_j(self, j, p, anums1, anums2):
        anums = [anums1, anums2]
        # Calculate matrix element and store in dict
        j_ = j % 2 + 1
        laj, maj, naj = anums[j-1]
        laj_, maj_, naj_ = anums[j_-1]
        fracs = list()
        for state_b in self.states_SP():
            lbj, mbj, nbj = self.get_numbers(state_b)
            # Check delta functions
            if mbj != maj_ or lbj != laj_:
                continue
            elif p == 1 and mbj != maj + 1:
                continue
            elif p == 2 and mbj != maj - 1:
                continue
            elif p == 3 and mbj != maj:
                continue
            # Get numerator
            if lbj == laj + 1:
                numerator = Pi(j, 1, laj=laj, maj=maj)
            elif lbj == laj - 1:
                numerator = Pi(j, 2, laj=laj, maj=maj)
            else:
                continue
            beta = self.beta(j)
            numerator *= (
                beta * self.I(lbj, nbj, laj, naj, j=j) *
                self.T(lbj, nbj, naj_, j=j)
            )
            denominator = (
                beta *
                (self.x(j=j, l=laj, n=naj)**2 - self.x(j=j, l=lbj, n=nbj)**2)
                + 1j * self.Gamma_aj(j) / self.E_0
            )
            fracs.append((numerator, denominator))
        self._mat_elts[j, p, anums1, anums2] = fracs

    # -- States --
    def states_SP(self):
        for state in self.ms.iter_states_SP(num_l=self.num_l):
            l, m, n = self.get_numbers(state)
            if (l, n, 1) in self._x and (l, n, 2) in self._x:
                yield state

    def states_SP_phonon(self):
        for l, m, n in self.hamiltonian.iter_lmn():
            yield (l, m, n), None  # TODO

    def states_EHP(self):
        return self.ms.iter_states_EHP(num_l=self.num_l)

    def get_numbers(self, state):
        return state[0]

    # -- Getters --
    def S(self, p, e_s):
        """Polarization S_p. See Riera (149)
        """
        if e_s is None:
            return self.intS(p)
        pms = basis_pmz()
        if p == 1:
            return abs(np.dot(e_s, pms[:, 0]).sum())**2
        elif p == 2:
            return abs(np.dot(e_s, pms[:, 1]).sum())**2
        elif p == 3:
            return abs(np.dot(e_s, pms[:, 2]).sum())**2
        else:  # p = 0
            return (abs(np.dot(e_s, pms[:, 0]).sum()) *
                    abs(np.dot(e_s, pms[:, 1]).sum()))

    def intS(self, p):
        if p in self._intS:
            return self._intS[p]
        else:
            self._set_intS()
            return self.intS(p)

    def _set_intS(self):
        def ifn(theta, phi, p):
            e_s = basis_spherical(theta, phi)[:, 0]
            return self.S(p=p, e_s=e_s)
        for p in range(4):
            self._intS[p] = integ.nquad(ifn, ranges=[(0, np.pi), (0, 2*np.pi)],
                                        args=(p,))[0]

    def g2(self, x1, x2, omega_l, omega_s):
        """See Riera (150)
        """
        return (
            (omega_l - self.E_g) / self.E_0 - omega_s / self.E_0 -
            self.beta(1) * x1**2 - self.beta(2) * x2**2
        )

    def G2(self, xa1, xa2, omega_l, omega_s, omega_q):
        """See Riera (166)
        """
        g2 = self.g2(x1=xa1, x2=xa2, omega_l=omega_l, omega_s=omega_s)
        return g2 - omega_q / self.E_0

    def p_cv0(self):
        """Momentum between valence and conduction bands, at k=0.
         See Riera (78).
        """
        if self._p_cv0 is not None:
            return self._p_cv0

        # The unit cell is defined by 3 vectors: a1, a2, a3, which
        # are the columns of...
        a_matrix = self._a_matrix

        # The function to integrate (in terms of the unit cell basis) is
        def intfn(c1, c2, c3, i):
            # Get position in cartesian basis
            x = np.dot(a_matrix, np.array([c1, c2, c3]))
            r = lin.norm(x, ord=2)
            # Evaluate u1 and u2 at r
            # TODO In Riera, this is written conj(u1'), but I don't know
            # what the prime is for
            u1_conj = np.conj(self.u_j(j=1)(r=r))
            du2 = self.u_j(j=2)(r=r, d_r=1)
            pu2 = x / r * du2
            return (u1_conj * pu2)[i]

        # Integrate real and imaginary parts separately
        def ifnr(c1, c2, c3, xi):
            return np.real(intfn(c1, c2, c3, i=xi))

        def ifni(c1, c2, c3, xi):
            return np.imag(intfn(c1, c2, c3, i=xi))

        # The integral is performed over the unit cell volume, 0 to 1 in
        # each coordinate
        x_re = integ.nquad(func=ifnr, ranges=[(0, 1)]*3, args=(0,))[0]
        x_im = integ.nquad(func=ifni, ranges=[(0, 1)]*3, args=(0,))[0]
        y_re = integ.nquad(func=ifnr, ranges=[(0, 1)]*3, args=(1,))[0]
        y_im = integ.nquad(func=ifni, ranges=[(0, 1)]*3, args=(1,))[0]
        z_re = integ.nquad(func=ifnr, ranges=[(0, 1)]*3, args=(2,))[0]
        z_im = integ.nquad(func=ifni, ranges=[(0, 1)]*3, args=(2,))[0]
        re = np.array([x_re, y_re, z_re])
        im = np.array([x_im, y_im, z_im])
        self._p_cv0 = re + 1j * im
        return self.p_cv0()

    def u_j(self, j):
        """Bloch function. I have not found the definition for this in
        Riera
        """
        def ujfn(r, d_r=0):
            return 1  # TODO
        return ujfn

    def sigma_0(self, omega_s, omega_l, e_l):
        """See Riera (104)
        """
        return (
            (4*np.sqrt(2) * self.V * self.e**4 *
             abs(np.dot(e_l, self.p_cv0()).sum())**2 * self.eta(omega_s) *
             self.mu_r**(1/2) * self.E_0**(3/2)) /
            (9*np.pi**2 * self.mu_0**2 * self.eta(omega_l) *
             omega_l * omega_s)
        )

    def Gamma_aj(self, j):
        return self._Gamma_aj[j-1]

    def Gamma_bj(self, j):
        return self._Gamma_bj[j-1]

    def V_j(self, j):
        return self._V_j[j-1]

    def mu_ij(self, j):
        return self._mu_ij[j-1]

    def mu_0j(self, j):
        return self._mu_0j[j-1]

    def beta(self, j):
        """See Riera (92)
        """
        return self.mu_r / self.mu_ij(j)

    # -- Coupling tensors --
    def I(self, lb, nb, la, na, j):
        r0 = self.r_0

        xa = self.x(l=la, n=na, j=j)
        xb = self.x(l=lb, n=nb, j=j)

        def ifn_j(r):
            return J(la+1/2)(z=xa*r/r0) * J(lb+1/2)(z=xb*r/r0) * r
        ans_j_re = integ.quad(lambda x: ifn_j(x).real, a=0, b=self.r_0)[0]
        ans_j_im = integ.quad(lambda x: ifn_j(x).imag, a=0, b=self.r_0)[0]
        ans_j = ans_j_re + 1j * ans_j_im

        ya = self.y(l=la, n=na, j=j)
        yb = self.y(l=lb, n=nb, j=j)

        def ifn_k(r):
            return K(la+1/2)(z=ya*r/r0) * K(lb+1/2)(z=yb*r/r0) * r
        ans_k_re = integ.quad(lambda x: ifn_k(x).real, a=self.r_0, b=np.inf)[0]
        ans_k_im = integ.quad(lambda x: ifn_k(x).imag, a=self.r_0, b=np.inf)[0]
        ans_k = ans_k_re + 1j * ans_k_im

        Aa, Ba = self._A_B(l=la, n=na, j=j)
        Ab, Bb = self._A_B(l=lb, n=nb, j=j)
        return Aa * Ab * ans_j + Ba * Bb * ans_k

    def T(self, l, na, nb, j):
        """See Riera (134). Using nu=1/2 always.
        """
        return self.I(lb=l, nb=nb, la=l, na=na, j=j)

    def II(self, lbj, nbj, lcj, ncj, la, na, j):
        """See Riera (210). Note: In Riera (210), the 'y' given to K is
        'x', but I don't think that is correct.
        """
        r0 = self.r_0
        Phi_ln = self.hamiltonian.Phi_ln(l=la, n=na)

        xb = self.x(l=lbj, n=nbj, j=j)
        xc = self.x(l=lcj, n=ncj, j=j)

        def ifnj(r):
            return (
                J(lbj+1/2)(xb * r / r0) * Phi_ln(r) *
                J(lcj+1/2)(xc * r / r0) * r
            )
        resultj_re = integ.quad(lambda x: ifnj(x).real, 0, r0)[0]
        resultj_im = integ.quad(lambda x: ifnj(x).imag, 0, r0)[0]
        resultj = resultj_re + 1j * resultj_im

        yb = self.y(l=lbj, n=nbj, j=j)
        yc = self.y(l=lcj, n=ncj, j=j)

        def ifnk(r):
            return (
                K(lbj+1/2)(yb * r / r0) * Phi_ln(r) *
                K(lcj+1/2)(yc * r / r0) * r

            )
        # TODO: Should the upper bound here be 0 or inf?
        resultk_re = integ.quad(lambda x: ifnk(x).real, r0, np.inf)[0]
        resultk_im = integ.quad(lambda x: ifnk(x).imag, r0, np.inf)[0]
        resultk = resultk_re + 1j * resultk_im

        Ab, Bb = self._A_B(l=lbj, n=nbj, j=j)
        Ac, Bc = self._A_B(l=lcj, n=ncj, j=j)
        return Ab * Ac * resultj + Bb * Bc * resultk

    def _Jl_Kl(self, l, n, j):
        return (J(l=l+1/2)(self.x(l=l, n=n, j=j)),
                K(l=l+1/2)(self.y(l=l, n=n, j=j)))

    def _A_B(self, l, n, j):
        Jln = J(l=l+1/2)(self.x(l=l, n=n, j=j))
        Kln = K(l=l+1/2)(self.y(l=l, n=n, j=j))
        denom = np.sqrt(Kln**2 * abs(Jln)**2 + Jln**2 * abs(Kln)**2)
        return Kln / denom, Jln / denom

    # -- Obtaining roots --
    def x(self, j, l=None, n=None, state=None):
        if l is not None and n is not None and (l, n, j) in self._x:
            return self._x[l, n, j]
        elif state is not None:
            l, m, n = self.get_numbers(state=state)
            return self.x(j=j, l=l, n=n)
        elif n == 0 and l >= self.num_l:
            return 0+0j
        else:
            return None

    def y(self, l, n, j):
        if (l, n, j) in self._y:
            return self._y[l, n, j]
        elif n == 0 and l >= self.num_l:
            return np.sqrt(self._y2(x2=0+0j, j=j))
        else:
            return None

    def _y2(self, x2, j):
        return (
            2 * self.mu_0j(j) * self.r_0**2 * self.V_j(j) -
            self.mu_0j(j) / self.mu_ij(j) * x2
        )

    def plot_root_fn_x(self, l, j, xdat, show=True):
        fn = self._get_root_function(l=l, j=j)

        def rootfn(xy):
            x, y = xy
            fr, fi, gr, gi = fn(np.array([x.real, x.imag, y.real, y.imag]))
            return np.array([fr + 1j * fi, gr + 1j * gi])

        def fpr(x):
            x = complex(x)
            y = np.sqrt(self._y2(x2=x**2, j=j))
            result = rootfn(np.array([x, y]))[0]
            # return np.real(result)
            if np.real(x) < 0:
                return np.imag(result)
            else:
                return np.real(result)

        fig, ax = plt.subplots(1, 1)
        ax.axhline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(0, color='gray', lw=1, alpha=.5)

        for n in range(self.num_n):
            x = self.x(j=j, l=l, n=n)
            if x is not None:
                ax.axvline(x.real, ls='--', color='green', lw=1, alpha=.5)
            else:
                print('crap')

        ydat = np.array([fpr(x) for x in xdat])
        # ydat /= lin.norm(ydat, ord=2)
        ax.plot(xdat, ydat, '-', color='red')

        ylog = np.log(np.abs(ydat))
        # ylog /= lin.norm(ylog, ord=2)
        ax.plot(xdat, ylog, '--', color='red')

        ypr = np.real(np.sqrt([self._y2(x2=x**2, j=j) for x in xdat]))
        # ypr /= lin.norm(ypr, ord=2)
        ax.plot(xdat, ypr, '-', color='blue')
        ax.plot(xdat, -ypr, '-', color='blue')
        # ax.set_ylim(bottom=0)

        if show:
            plt.show()
        return fig, ax

    def _fill_roots_xy(self):
        for j, l in it.product([1, 2], range(self.num_l)):
            for root, n in zip(self._solve_roots_xy(l=l, j=j),
                               range(self.num_n)):
                x, y = root
                assert not np.isnan(x)  # TODO
                assert not np.isnan(y)  # TODO
                self._x[l, n, j] = x
                self._y[l, n, j] = y

    def _get_expected_roots_xy(self, l, j, *args, **kwargs):
        for x0 in self._expected_roots_x[l, j]:
            x0 = complex(x0)
            y0 = np.sqrt(self._y2(x2=x0**2, j=j))
            yield np.array([x0, y0])

    def _get_root_func1(self, l, j, *args, **kwargs):
        l = l+1/2
        mu0 = self.mu_0j(j)
        mui = self.mu_ij(j)

        def rf1(x, y, partial_x=0, partial_y=0):
            J_x = J(l)(x)
            dJ_x = J(l, d=1)(x)
            K_y = K(l)(y)
            dK_y = K(l, d=1)(y)
            if partial_x == 0 and partial_y == 0:
                return (
                    mu0 * x * dJ_x * K_y - mui * y * J_x * dK_y
                )
            elif (partial_x, partial_y) == (1, 0):
                d2J_x = J(l, d=2)(x)
                return (
                    mu0 * (dJ_x + x * d2J_x) * K_y - mui * y * dJ_x + dK_y
                )
            elif (partial_x, partial_y) == (0, 1):
                d2K_y = K(l, d=2)(y)
                return (
                    mu0 * x * dJ_x * dK_y - mui * (dK_y + y * d2K_y) * J_x
                )
            else:
                raise RuntimeError  # TODO
        return rf1

    def _get_root_func2(self, j, *args, **kwargs):
        mu0 = self.mu_0j(j)
        mui = self.mu_ij(j)
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
