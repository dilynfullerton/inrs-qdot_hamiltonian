"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import itertools as it

import numpy as np
from scipy import integrate as integ
from helper_functions import basis_pmz, threej
from ExitonModelSpace import ExitonModelSpace


def Lambda(lbj, mbj, lcj, mcj, la, ma):
    return (
        (-1)**(lbj % 2) *
        np.sqrt((2*lbj+1)*(2*lcj+1)*(2*la+1)/4/np.pi) *
        threej(lbj, la, lcj, -mbj, ma, mcj) *
        threej(lbj, la, lcj, 0, 0, 0)
    )


class RamanQD:
    def __init__(self, phonon_space, exiton_space):
        # Model parameters
        self.ph_space = phonon_space
        self.ex_space = exiton_space

        # Physical constants
        self.meff_reduced = self.ex_space.free_electron_mass / (
            1/self.ex_space.meff_in(band=ExitonModelSpace.BAND_COND) +
            1/self.ex_space.meff_in(band=ExitonModelSpace.BAND_VAL)
        )
        self.E_0 = 1/2/self.meff_reduced/self.ph_space.R**2

        # Matrix elements
        self._matelts_ph = dict()
        self._I_dict = dict()
        self._II_dict = dict()

    # -- Raman cross section --
    def differential_raman_efficiency(self, omega_l, e_l, omega_s, e_s,
                                      Gamma_f, Gamma_a, Gamma_b, E_gap):
        """Differential Raman scattering efficiency for phonon assisted
        transitions. This is based on (Riera 215), divided by sigma_ph
        (Riera 126)
        """
        cs = 0
        for p in range(4):
            cs += self._differential_raman_efficiency_p(
                omega_l=omega_l, omega_s=omega_s, e_s=e_s, p=p,
                Gamma_f=Gamma_f, Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                E_gap=E_gap
            )
        t0 = 8/3 * (omega_s/self.E_0)**2 * Gamma_f/self.E_0
        ans = t0 * cs
        return ans

    def _differential_raman_efficiency_p(self, omega_l, omega_s, e_s, p,
                                         Gamma_f, Gamma_a, Gamma_b, E_gap):
        """This is the expression of (Riera 216, 217) divided by sigma_ph.
        """
        t1 = 0
        for state_ph, state_e, state_h in it.product(
                self.ph_space.states(),
                self.ex_space.electron_states(),
                self.ex_space.hole_states()
        ):
            omega_ph = self.ph_space.get_omega(state_ph)
            G2 = self.G2(
                omega_l=omega_l, omega_s=omega_s, omega_ph=omega_ph,
                xa1=self.ex_space.x(state_e), xa2=self.ex_space.x(state_h),
                E_gap=E_gap,
            )
            t2 = self.S(p=p, e_s=e_s) / (G2**2 + (Gamma_f/self.E_0)**2)

            if p == 0 and state_ph.m == 0:
                M11 = self.Mph_j(
                    state_e.band, p=1,
                    state_ph=state_ph, state_e=state_e, state_h=state_h,
                    omega_s=omega_s, omega_ph=omega_ph,
                    Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                )
                M12 = self.Mph_j(
                    state_e.band, p=2,
                    state_ph=state_ph, state_e=state_e, state_h=state_h,
                    omega_s=omega_s, omega_ph=omega_ph,
                    Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                )
                M21 = self.Mph_j(
                    state_h.band, p=1,
                    state_ph=state_ph, state_e=state_e, state_h=state_h,
                    omega_s=omega_s, omega_ph=omega_ph,
                    Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                )
                M22 = self.Mph_j(
                    state_h.band, p=2,
                    state_ph=state_ph, state_e=state_e, state_h=state_h,
                    omega_s=omega_s, omega_ph=omega_ph,
                    Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                )
                t1 += t2 * (
                    M11.real * M22.real + M12.real * M21.real +
                    M11.imag * M22.imag + M12.imag * M21.imag
                )
            elif p > 0:
                M1p = self.Mph_j(
                    state_e.band, p,
                    state_ph=state_ph, state_e=state_e, state_h=state_h,
                    omega_s=omega_s, omega_ph=omega_ph,
                    Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                )
                M2p = self.Mph_j(
                    state_h.band, p,
                    state_ph=state_ph, state_e=state_e, state_h=state_h,
                    omega_s=omega_s, omega_ph=omega_ph,
                    Gamma_a=Gamma_a, Gamma_b=Gamma_b,
                )
                t1 += abs(M1p + M2p)**2 * t2
        return t1

    def _key_mph_j(self, transition_band, p, state_ph, state_e, state_h):
        return (
            transition_band, p, self.ph_space.get_nums(state_ph),
            self.ex_space.get_nums(state_e), self.ex_space.get_nums(state_h)
        )

    def Mph_j(self, transition_band, p,
              state_ph, state_e, state_h, omega_s, omega_ph,
              Gamma_a, Gamma_b):
        """See Riera (218, 219)
        """
        # If stored in dict, use that
        k = self._key_mph_j(
            transition_band=transition_band, p=p,
            state_ph=state_ph, state_e=state_e, state_h=state_h
        )
        if k in self._matelts_ph:
            mm = 0
            fracs = self._matelts_ph[k]
            for num1, denom1, num2, denom2 in fracs:
                mm += (
                    num1 / (
                        denom1 + (omega_s + omega_ph + 1j * Gamma_a) / self.E_0
                    ) *
                    num2 / (denom2 + (omega_s + 1j * Gamma_b) / self.E_0)
                )
            return mm
        else:
            self._set_Mph_j(
                transition_band=transition_band, p=p,
                state_ph=state_ph, state_e=state_e, state_h=state_h
            )
            return self.Mph_j(
                transition_band=transition_band, p=p,
                state_ph=state_ph, state_e=state_e,
                state_h=state_h, omega_s=omega_s, omega_ph=omega_ph,
                Gamma_a=Gamma_a, Gamma_b=Gamma_b
            )

    def _set_Mph_j(self, transition_band, p, state_ph, state_e, state_h):
        print('Computing matrix element for')
        print('  phonon:     {}'.format(state_ph))
        print('  electron:   {}'.format(state_e))
        print('  hole:       {}'.format(state_h))
        print('  transition: {}'.format(transition_band))
        print('  p:          {}'.format(p))
        if state_e.band == transition_band:
            trans_state = state_e
            next_states = self.ex_space.electron_states
            mute_state = state_h
        else:
            trans_state = state_h
            next_states = self.ex_space.hole_states
            mute_state = state_e
        laj, maj = trans_state.l, trans_state.m
        x2aj = self.ex_space.x(trans_state)**2
        fracs = list()
        for state_b in next_states():
            lbj, mbj = state_b.l, state_b.m
            # Check delta functions
            if p == 1 and mbj != maj + 1:
                continue
            elif p == 2 and mbj != maj - 1:
                continue
            elif p == 3 and mbj != maj:
                continue
            # Get first numerator and second denominator
            beta = self.beta(transition_band)
            if lbj == laj + 1:
                first_numerator = beta * self.Pi(
                    transition_band, 1, laj=laj, maj=maj
                )
            elif lbj == laj - 1:
                first_numerator = beta * self.Pi(
                    transition_band, 2, laj=laj, maj=maj
                )
            else:
                continue
            x2bj = self.ex_space.x(state_b)**2
            second_denominator = beta * (x2aj - x2bj)
            for state_c in next_states():
                # Check delta function
                if state_c.l != mute_state.l:
                    continue
                # Get first denominator and second numerator
                x2cj = self.ex_space.x(state_c)**2
                first_denominator = beta * (x2aj - x2cj)
                second_numerator = (
                    self.II(
                        state_b=state_b, state_c=state_c, state_ph=state_ph) *
                    Lambda(
                        state_b.l, state_b.m, state_c.l, state_c.m,
                        state_ph.l, state_ph.m,
                    ) *
                    self.I(state_b=state_b, state_a=trans_state) *
                    self.I(state_b=mute_state, state_a=state_c)
                )
                fracs.append(
                    (first_numerator, first_denominator,
                     second_numerator, second_denominator)
                )
        anums0 = self.ph_space.get_nums(state_ph)
        anums1 = self.ex_space.get_nums(state_e)
        anums2 = self.ex_space.get_nums(state_h)
        self._matelts_ph[transition_band, p, anums0, anums1, anums2] = fracs

    # -- Getters --
    def S(self, p, e_s):
        """Polarization S_p. See Riera (149)
        """
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

    def g2(self, x1, x2, omega_l, omega_s, E_gap):
        """See Riera (150)
        """
        return (
            (omega_l - E_gap - omega_s) / self.E_0 -
            self.beta(1) * x1**2 - self.beta(2) * x2**2
        )

    def G2(self, xa1, xa2, omega_l, omega_s, omega_ph, E_gap):
        """See Riera (166)
        """
        g2 = self.g2(
            x1=xa1, x2=xa2, omega_l=omega_l, omega_s=omega_s, E_gap=E_gap)
        return g2 - omega_ph / self.E_0

    def u_j(self, j):
        """Bloch function. I have not found the definition for this in
        Riera
        """
        def ujfn(r, d_r=0):
            return 1  # TODO
        return ujfn

    def beta(self, j):
        """See Riera (92)
        """
        return self.meff_reduced / self.ex_space.meff_in(band=j)

    def Pi(self, j, p, laj, maj):
        if j == ExitonModelSpace.BAND_COND and p == 1:
            return -np.sqrt(
                (laj + maj + 2) * (laj + maj + 1) /
                2 / (2 * laj + 1) / (2 * laj + 3)
            )
        elif j == ExitonModelSpace.BAND_COND and p == 2:
            return -self.Pi(j=j, p=1, laj=laj, maj=-maj)
        elif j == ExitonModelSpace.BAND_COND and p == 3:
            return np.sqrt(
                (laj - maj + 1) * (laj + maj + 1) / (2 * laj + 1) /
                (2 * laj + 3)
            )
        elif p == 1:
            return -self.Pi(j=ExitonModelSpace.BAND_COND,
                            p=1, laj=laj-1, maj=-maj-1)
        elif p == 2:
            return -self.Pi(j=j, p=1, laj=laj, maj=-maj)
        elif p == 3:
            return self.Pi(j=ExitonModelSpace.BAND_COND,
                           p=3, laj=laj-1, maj=maj)

    # -- Matrix element integrals --
    def _key_I(self, state_b, state_a):
        return (
            state_b.band, state_b.l, state_b.n,
            state_a.band, state_a.l, state_a.n,
        )

    def I(self, state_b, state_a):
        """Returns the radial part of the matrix element
        <la ma na | lb mb nb>, obtained by integrating
        R_ln^dag R_ln from r = 0 to r = inf
        """
        k = self._key_I(state_b, state_a)
        if state_b > state_a:
            return np.conj(self.I(state_a, state_b))
        elif k in self._I_dict:
            return self._I_dict[k]
        psia = self.ex_space.wavefunction_envelope_radial(state_a)
        psib = self.ex_space.wavefunction_envelope_radial(state_b)

        def ifn(r):
            return complex(np.conj(psia(r)) * psib(r)) * r**2

        re = integ.quad(lambda x: ifn(x).real, 0, np.inf)[0]
        im = integ.quad(lambda x: ifn(x).imag, 0, np.inf)[0]
        ans = re + 1j * im
        self._I_dict[k] = ans
        return self.I(state_b=state_b, state_a=state_a)

    def _key_II(self, state_b, state_c, state_ph):
        return (state_b.band, state_b.l, state_b.n,
                state_c.band, state_c.l, state_c.n,
                state_ph.l, state_ph.n)

    def II(self, state_b, state_c, state_ph):
        """See Riera (210). Note: In Riera (210), the 'y' given to K is
        'x', but I don't think that is correct.

        Returns the radial part of the matrix element
        < nb lb mb | H_ph_na,la,ma | nc lc mc >
        """
        k = self._key_II(state_b, state_c, state_ph=state_ph)
        if state_b > state_c:
            return np.conj(self.II(state_c, state_b, state_ph))
        elif k in self._II_dict:
            return self._II_dict[k]
        Phi_ln = self.ph_space.Phi_ln(state_ph)
        psib = self.ex_space.wavefunction_envelope_radial(state_b)
        psic = self.ex_space.wavefunction_envelope_radial(state_c)

        def ifn(r):
            return complex(np.conj(psib(r)) * Phi_ln(r) * psic(r)) * r**2
        re = integ.quad(lambda x: ifn(x).real, 0, np.inf)[0]
        im = integ.quad(lambda x: ifn(x).imag, 0, np.inf)[0]
        ans = re + 1j * im
        self._II_dict[k] = ans
        return self.II(state_b=state_b, state_c=state_c, state_ph=state_ph)
