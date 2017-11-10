"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import itertools as it
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integ
from helper_functions import J, K, basis_pmz, basis_spherical, Y_lm


def _three_y(l1, m1, l2, m2, l3, m3):
    return 0  # TODO


class RamanQD:
    def __init__(self, phonon_space, exiton_space, E_gap):
        # Model constants
        self.phonon_space = phonon_space
        self.exiton_space = exiton_space

        # Physical constants
        self.volume = self.phonon_space.volume
        self.E_gap = E_gap

    # -- Raman cross section --
    def differential_resonant_raman_cross_section(
            self, omega_l, omega_s, e_l, e_s, n_l, n_s):
        """Differential Raman scattering cross section based on
        Chamberlain (41).
        """
        print('Obtaining cross section for')
        print('  omega_l = {}'.format(omega_l))
        print('  omega_s = {}'.format(omega_s))
        cs = 0
        for s_ph in self.phonon_space.states():
            if s_ph.l == 0:
                m = self.M_FI_0()
            elif s_ph.l == 1:
                m = self.M_FI_1()
            else:
                continue
            cs += self.lorenzian_term() * abs(m)**2
        ans = self.S_0() * cs
        print('Cross section = {}'.format(ans))
        print('Done\n')
        return ans

    def M_FI_0(self, state_ph, omega_l, omega_s):
        """Matrix element based on Chamberlain (27)
        :param state_ph:
        :param omega_l:
        :param omega_s:
        :return:
        """
        melt = 0
        for state_e, state_h in it.product(
                self.exiton_space.electron_states(),
                self.exiton_space.hole_states()
        ):
            if state_e.l != state_h.l or state_e.m != state_h.m:
                continue
            l, m = state_e.l, state_e.m

            # Compute first term: Matrix element of 3 spherical harmonics
            term1 = _three_y(l1=l, m1=m, l2=state_ph.l, m2=state_ph.m,
                             l3=l, m3=m)

            # Compute second term: Overlap of state_e and state_h
            term2 = (
                self.F_0(n1=state_e.n, n2=state_h.n) /
                (
                    omega_l - self.E_gap -
                    self.exiton_space.get_omega(state_e) -
                    self.exiton_space.get_omega(state_h) +
                    1j * (self.Gamma(state_e) + self.Gamma(state_h))
                )
            )

            # Compute third term: Phonon matrix elements for electron
            # transitions
            term3 = 0
            for next_state_e in self.exiton_space.electron_states():
                if next_state_e.l != state_e.l:
                    continue
                term31 = self.F_0(n1=state_h.n, n2=next_state_e.n)
                term32 = self.phonon_matelt(
                    bra=next_state_e, mid=state_ph, ket=state_e)
                term3 += term31 * term32 / (
                    omega_s - self.E_gap -
                    self.exiton_space.get_omega(next_state_e) -
                    self.exiton_space.get_omega(state_h) +
                    1j * (self.Gamma(next_state_e) + self.Gamma(state_h))
                )

            # Compute fourth term: Phonon matrix elements for electron
            # transitions
            # NOTE: I have delegated the negation of this term in this equation
            # (which comes from the fact that the hole has opposite charge
            # to the electron) to the phonon_matelt function
            term4 = 0
            for next_state_h in self.exiton_space.hole_states():
                if next_state_h.l != state_h.l:
                    continue
                term41 = self.F_0(n1=next_state_h.n, n2=state_e.n)
                term42 = self.phonon_matelt(
                    bra=next_state_h, mid=state_ph, ket=state_h)
                term4 += term41 * term42 / (
                    omega_s - self.E_gap -
                    self.exiton_space.get_omega(next_state_h) -
                    self.exiton_space.get_omega(state_e) +
                    1j * (self.Gamma(next_state_h) + self.Gamma(state_e))
                )

            melt += term1 * term2 * (term3 + term4)
        return melt

    def M_FI_1(self):
        return 0  # TODO

    def lorenzian_term(self):
        return 0  # TODO

    def S_0(self):
        return 0  # TODO

    def Gamma(self, state):
        return 0  # TODO

    def F_0(self, n1, n2):
        return 0  # TODO

    def phonon_matelt(self, bra, mid, ket):
        return 0  # TODO
