"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import itertools as it
import numpy as np
from scipy import integrate as integ
from helper_functions import threej, Y_lm, basis_spherical


def Lambda(lbj, mbj, lcj, mcj, la, ma):
    return (
        (-1)**(lbj % 2) *
        np.sqrt((2*lbj+1)*(2*lcj+1)*(2*la+1)/4/np.pi) *
        threej(lbj, la, lcj, -mbj, ma, mcj) *
        threej(lbj, la, lcj, 0, 0, 0)
    )


def _integ_matelt(bra, mid, ket, keyfunc, storedict, oper, intfunc):
    k = keyfunc(bra=bra, mid=mid, ket=ket)
    if bra < ket:
        return np.conj(
            _integ_matelt(
                bra=ket, mid=mid, ket=bra,
                keyfunc=keyfunc, storedict=storedict, oper=oper,
                intfunc=intfunc
            )
        )
    elif k in storedict:
        return storedict[k]
    else:
        storedict[k] = intfunc(bra=bra, ket=ket, oper=oper)
        return storedict[k]


class RamanQD:
    def __init__(
            self, phonon_space, exiton_space, phonon_lifetime,
            electron_lifetime, hole_lifetime,
            cavity_space=None, cavity_lifetime=None,
    ):
        # Model parameters
        self.ph_space = phonon_space
        self.ex_space = exiton_space
        self.cav_space = cavity_space

        # Lifetimes
        self.gamma_ph = phonon_lifetime
        self.gamma_e = electron_lifetime
        self.gamma_h = hole_lifetime
        self.gamma_cav = cavity_lifetime

        # Convenience references
        self.volume = self.ph_space.volume
        self.free_electron_mass = self.ex_space.free_electron_mass
        self.C_F = self.ph_space.C_F
        self.R = self.ph_space.R

        # Matrix elements
        self._mfi_ph_dict = dict()
        self._mfi_ph_cav_dict = dict()
        self._e_rad_e_radial_dict = dict()
        self._e_rad_e_angular_dict = dict()
        self._e_ph_e_radial_dict = dict()
        self._e_ph_e_angular_dict = dict()
        self._e_cav_e_dict = dict()

    # -- Raman cross section --
    def differential_raman_cross_section(
            self, omega_l, e_l, n_l, omega_s, e_s, n_s,
            include_cavity=False
    ):
        """See Chamberlain (1)
        """
        return (
            self.volume**2 * omega_s**3 * n_l * n_s**3 / 8 / np.pi**3 /
            omega_l * self.scattering_rate(
                omega_l=omega_l, e_l=e_l, n_l=n_l,
                omega_s=omega_s, e_s=e_s, n_s=n_s,
                include_cavity=include_cavity,
            )
        )

    def scattering_rate(self, omega_l, e_l, n_l, omega_s, e_s, n_s,
                        include_cavity=False):
        if not include_cavity:
            return self._scattering_rate_no_cavity(
                omega_l=omega_l, e_l=e_l, n_l=n_l,
                omega_s=omega_s, e_s=e_s, n_s=n_s
            )
        else:
            return self._scattering_rate_cavity(
                omega_l=omega_l, e_l=e_l, n_l=n_l,
                omega_s=omega_s, e_s=e_s, n_s=n_s
            )

    def _scattering_rate_no_cavity(self, omega_l, e_l, n_l, omega_s, e_s, n_s):
        """See Chamberlain (2)
        :param omega_s: Secondary frequency
        :param e_s: Secondary polarization
        """
        w = 0
        for phonon in self.ph_space.states():
            if phonon.m != 0:  # TODO
                continue
            mfi = self.M_FI(
                omega_s=omega_s, e_s=e_s, n_s=n_s,
                omega_l=omega_l, e_l=e_l, n_l=n_l, phonon=phonon
            )
            delta = self.delta(omega_s=omega_s, omega_l=omega_l,
                               phonon_state=phonon)
            w += 2 * np.pi * abs(mfi)**2 * delta
        return w

    def _scattering_rate_cavity(self, omega_l, e_l, n_l, omega_s, e_s, n_s):
        """Calculate the scattering rate for transitions that involve a
        cavity transition.

        The cases I currently consider are the sequential transitions
            (a1) cav-e, ph-h
                  0. |ph=0, cav=1, rad=L, e=0, h=0>
                  1. |ph=0, cav=1, rad=0, e=e1, h=h1>
                  2. |ph=0, cav=0, rad=0, e=e2, h=h1>
                  3. |ph=ph, cav=0, rad=0, e=e2, h=h2>
                  4. |ph=ph, cav=0, rad=S, e=0, h=0>

            (a2) cav-h, ph-e

            (b1) ph-e, cav-h
                  0. |ph=0, cav=1, rad=L, e=0, h=0>
                  1. |ph=0, cav=1, rad=0, e=e1, h=h1>
                  2. |ph=ph, cav=1, rad=0, e=e2, h=h1>
                  3. |ph=ph, cav=0, rad=0, e=e2, h=h2>
                  4. |ph=ph, cav=0, rad=S, e=0, h=0>

            (b2) ph-h, cav-e

            (c1) cav-e, ph-e
                  0. |ph=0, cav=0, rad=L, e=0, h=0>
                  1. |ph=0, cav=0, rad=0, e=e1, h=h>
                  2. |ph=0, cav=1, rad=0, e=e2, h=h>
                  3. |ph=ph, cav=1, rad=0, e=e3, h=h>
                  4. |ph=ph, cav=1, rad=S, e=0, h=0>

            (c2) cav-h, ph-h

            (d1) ph-e, cav-e
                  0. |ph=0, cav=0, rad=L, e=0, h=0>
                  1. |ph=0, cav=0, rad=0, e=e1, h=h>
                  2. |ph=ph, cav=0, rad=0, e=e2, h=h>
                  3. |ph=ph, cav=1, rad=0, e=e3, h=h>
                  4. |ph=ph, cav=1, rad=S, e=0, h=0>

            (d2) ph-h, cav-h

        :param omega_l:
        :param e_l:
        :param n_l:
        :param omega_s:
        :param e_s:
        :param n_s:
        :return:
        """
        w = 0
        for phonon in self.ph_space.states():
            if phonon.m != 0:  # TODO
                continue
            for final_cavity in self.cav_space.states():
                mfi = self.M_FI_cav(
                    omega_s=omega_s, e_s=e_s, n_s=n_s,
                    omega_l=omega_l, e_l=e_l, n_l=n_l,
                    phonon=phonon, final_cavity=final_cavity
                )
                delta = self.delta_cav(
                    omega_s=omega_s, omega_l=omega_l,
                    phonon_state=phonon, final_cavity=final_cavity
                )
                w += 2 * np.pi * abs(mfi)**2 * delta
        return w

    # --- Matrix elements ---
    def M_FI(self, omega_s, e_s, n_s, omega_l, e_l, n_l, phonon):
        """See Chamberlain (3)
        :param omega_s:
        :param omega_l:
        :param e_s:
        :param e_l:
        :param phonon:
        :return:
        """
        key = self._get_key_M_FI(phonon)
        if key in self._mfi_ph_dict:
            return self._mfi_ph_dict[key]
        else:
            self._set_M_FI(
                omega_s=omega_s, e_s=e_s, n_s=n_s,
                omega_l=omega_l, e_l=e_l, n_l=n_l,
                phonon=phonon, storedict=self._mfi_ph_dict, key=key
            )
            return self._mfi_ph_dict[key]

    def _get_key_M_FI(self, phonon):
        return phonon.l, phonon.m, phonon.n

    def _set_M_FI(self, omega_s, e_s, n_s, omega_l, e_l, n_l,
                  phonon, storedict, key):
        mfi = 0
        omega_s = omega_l - self.ph_space.get_omega(phonon)
        for mu1, mu2 in it.product(self.ex_space.electron_hole_states(),
                                   repeat=2):
            if not self.selection_rules(ehp1=mu1, ehp2=mu2, phonon=phonon):
                continue
            numerator = (
                self.matelt_her_p_F(
                    ket=mu2, omega_s=omega_s, e_s=e_s, n_s=n_s
                ) *
                self.matelt_hep(bra=mu2, ph=phonon, ket=mu1) *
                self.matelt_her_m_I(
                    bra=mu1, omega_l=omega_l, e_l=e_l, n_l=n_l)
            )
            denominator = (
                (omega_s - self.ex_space.get_omega_ehp(mu2) +
                 1j * self.Gamma_ehp(mu2)) *
                (omega_l - self.ex_space.get_omega_ehp(mu1) +
                 1j * self.Gamma_ehp(mu1))
            )
            mfi0 = numerator / denominator
            # print('    {}'.format(mfi0))
            mfi += mfi0
        print('  final phonon = {}'.format(phonon))
        print('  M_FI         = {}'.format(mfi))
        print()
        storedict[key] = mfi

    def M_FI_cav(self, omega_s, e_s, n_s, omega_l, e_l, n_l,
                 phonon, final_cavity):
        key = self._get_key_M_FI_cav(phonon=phonon, cavity=final_cavity)
        if key in self._mfi_ph_cav_dict:
            return self._mfi_ph_cav_dict[key]
        else:
            self._set_M_FI_cav(
                omega_s=omega_s, e_s=e_s, n_s=n_s,
                omega_l=omega_l, e_l=e_l, n_l=n_l,
                phonon=phonon, final_cavity=final_cavity,
                storedict=self._mfi_ph_cav_dict, key=key
            )
            return self._mfi_ph_cav_dict[key]

    def _get_key_M_FI_cav(self, phonon, cavity):
        return phonon.l, phonon.m, phonon.n, cavity.n

    def _set_M_FI_cav(self, omega_s, e_s, n_s, omega_l, e_l, n_l,
                      phonon, final_cavity, storedict, key):
        mfi = 0
        for mu1, mu2, mu3 in it.product(
                self.ex_space.electron_hole_states(), repeat=3
        ):
            rules_abcd = self.selection_rules_a_b_c_d(
                ehp1=mu1, ehp2=mu2, ehp3=mu3, ph_f=phonon,
                cav_f=final_cavity
            )
            if not any(rules_abcd):
                continue
            sela, selb, selc, seld = rules_abcd
            if sela or selc:
                # Matrix element 1->2 (destroy/create cavity)
                m12 = self.matelt_hec(bra=mu2, ket=mu1, cav=final_cavity)
                # Matrix element 2->3 (create phonon)
                m23 = self.matelt_hep(bra=mu3, ket=mu2, ph=phonon)
            else:
                # Matrix element 1->2 (create phonon)
                m12 = self.matelt_hep(bra=mu2, ket=mu1, ph=phonon)
                # Matrix element 2->3 (destroy/create cavity)
                m23 = self.matelt_hec(bra=mu3, ket=mu2, cav=final_cavity)

            if final_cavity == self.cav_space.lo:
                gcav_i = self.Gamma_cav_hi()
                gcav_f = self.Gamma_cav_lo()
                omega_cav_i = self.cav_space.get_omega(self.cav_space.hi)
                omega_cav_f = self.cav_space.get_omega(self.cav_space.lo)
            else:
                gcav_i = self.Gamma_cav_lo()
                gcav_f = self.Gamma_cav_hi()
                omega_cav_i = self.cav_space.get_omega(self.cav_space.lo)
                omega_cav_f = self.cav_space.get_omega(self.cav_space.hi)

            d_omega_cav = omega_cav_f - omega_cav_i
            omega_ph = self.ph_space.get_omega(phonon)
            omega_s = omega_l - omega_ph - d_omega_cav
            # Matrix element 0->1 (destroy radiation)
            m01 = self.matelt_her_m_I(bra=mu1, omega_l=omega_l,
                                      e_l=e_l, n_l=n_l)
            # Matrix element 3->4 (create radiation)
            m34 = self.matelt_her_p_F(ket=mu3, omega_s=omega_s,
                                      e_s=e_s, n_s=n_s)

            emu2 = self.ex_space.get_omega_ehp(mu2)

            d1 = omega_l - self.ex_space.get_omega_ehp(mu1)
            d3 = omega_s - self.ex_space.get_omega_ehp(mu3)

            gmu1 = self.Gamma_ehp(mu1)
            gmu2 = self.Gamma_ehp(mu2)
            gmu3 = self.Gamma_ehp(mu3)
            gph = self.Gamma_ph(phonon)
            g1 = gmu1 + gcav_i
            g3 = gmu3 + gcav_f + gph

            if sela:  # cavf = 0
                g2 = gmu2 + gcav_f
                d2 = omega_l + omega_cav_i - emu2
            elif selb:  # cavf = 0
                g2 = gmu2 + gcav_i + gph
                d2 = omega_s - emu2 - omega_cav_i
            elif selc:  # cavf = 1
                g2 = gmu2 + gcav_f
                d2 = omega_l - emu2 - omega_cav_f
            else:  # cavf = 1
                g2 = gmu2 + gcav_i + gph
                d2 = omega_s + omega_cav_f - emu2

            mfi += (
                m01 * m12 * m23 * m34 / (d1 + 1j * g1) / (d2 + 1j * g2) /
                (d3 + 1j * g3)
            )
        print('  final phonon = {}'.format(phonon))
        print('  final cavity = {}'.format(final_cavity))
        print('  M_FI         = {}'.format(mfi))
        print()
        storedict[key] = mfi

    def selection_rules(self, ehp1, ehp2, phonon):
        """Returns true if pass selection rules
        """
        e1, h1 = ehp1
        e2, h2 = ehp2
        return (
            (h1 == h2 and self._rule_phonon(e1, e2, ph=phonon)) or
            (e1 == e2 and self._rule_phonon(h1, h2, ph=phonon))
            # False
        )

    def selection_rules_a_b_c_d(self, ehp1, ehp2, ehp3, ph_f, cav_f):
        if ehp1 == ehp2 or ehp2 == ehp3:
            return (False,) * 4
        elif ehp1[0] != ehp2[0] and ehp1[1] != ehp2[1]:
            return (False,) * 4
        elif ehp3[0] != ehp2[0] and ehp3[1] != ehp2[1]:
            return (False,) * 4
        return (
            self.selection_rules_cav_a(
                ehp1=ehp1, ehp2=ehp2, ehp3=ehp3, ph_f=ph_f, cav_f=cav_f),
            self.selection_rules_cav_b(
                ehp1=ehp1, ehp2=ehp2, ehp3=ehp3, ph_f=ph_f, cav_f=cav_f),
            self.selection_rules_cav_c(
                ehp1=ehp1, ehp2=ehp2, ehp3=ehp3, ph_f=ph_f, cav_f=cav_f),
            self.selection_rules_cav_d(
                ehp1=ehp1, ehp2=ehp2, ehp3=ehp3, ph_f=ph_f, cav_f=cav_f),
        )

    def selection_rules_cav_a(self, ehp1, ehp2, ehp3, ph_f, cav_f):
        e1, h1 = ehp1
        e2, h2 = ehp2
        e3, h3 = ehp3
        if cav_f.n == 0 and h1 == h2 and e2 == e3:
            return (
                self._rule_phonon(state1=h2, state2=h3, ph=ph_f) and
                self._rule_cavity(state1=e1, state2=e2, cav=cav_f)
            )
        elif cav_f.n == 0 and e1 == e2 and h2 == h3:
            return (
                self._rule_phonon(state1=e2, state2=e3, ph=ph_f) and
                self._rule_cavity(state1=h1, state2=h2, cav=cav_f)
            )
        else:
            return False

    def selection_rules_cav_c(self, ehp1, ehp2, ehp3, ph_f, cav_f):
        e1, h1 = ehp1
        e2, h2 = ehp2
        e3, h3 = ehp3
        if cav_f.n == 1 and h1 == h2 and h2 == h3:
            return (
                self._rule_phonon(state1=e2, state2=e3, ph=ph_f) and
                self._rule_cavity(state1=e1, state2=e2, cav=cav_f)
            )
        elif cav_f.n == 1 and e1 == e2 and e2 == e3:
            return (
                self._rule_phonon(state1=h2, state2=h3, ph=ph_f) and
                self._rule_cavity(state1=h1, state2=h2, cav=cav_f)
            )
        else:
            return False

    def selection_rules_cav_b(self, ehp1, ehp2, ehp3, ph_f, cav_f):
        e1, h1 = ehp1
        e2, h2 = ehp2
        e3, h3 = ehp3
        if cav_f.n == 0 and h1 == h2 and e2 == e3:
            return (
                self._rule_phonon(state1=e1, state2=e2, ph=ph_f) and
                self._rule_cavity(state1=h2, state2=h3, cav=cav_f)
            )
        elif cav_f.n == 0 and e1 == e2 and h2 == h3:
            return (
                self._rule_phonon(state1=h1, state2=h2, ph=ph_f) and
                self._rule_cavity(state1=e2, state2=e3, cav=cav_f)
            )
        else:
            return False

    def selection_rules_cav_d(self, ehp1, ehp2, ehp3, ph_f, cav_f):
        e1, h1 = ehp1
        e2, h2 = ehp2
        e3, h3 = ehp3
        if cav_f.n == 1 and h1 == h2 and h2 == h3:
            return (
                self._rule_phonon(state1=e1, state2=e2, ph=ph_f) and
                self._rule_cavity(state1=e2, state2=e3, cav=cav_f)
            )
        elif cav_f.n == 1 and e1 == e2 and e2 == e3:
            return (
                self._rule_phonon(state1=h1, state2=h2, ph=ph_f) and
                self._rule_cavity(state1=h2, state2=h3, cav=cav_f)
            )
        else:
            return False

    def _rule_phonon(self, state1, state2, ph):
        # TODO: Justify this rule from Chamberlain
        if state1.band == self.ex_space.BAND_VAL:  # No interaction with holes?
            return False
        elif state1.band != state2.band:
            return False
        elif state1 == state2:  # Force a transition
            return False
        elif ph.m != 0:
            return False
        elif ph.l % 2 == 0:  # even
            return (state1.l, state1.m) == (state2.l, state2.m)
        else:  # odd
            return abs(state1.l - state2.l) == 1 and state1.m == state2.m

    def _rule_cavity(self, state1, state2, cav):
        # TODO: Figure out if this is correct
        if state1.band != state2.band:
            return False
        elif state1 == state2:  # Force a transition
            return False
        else:
            return abs(state1.l - state2.l) == 1
        # return state1.m == state2.m and abs(state1.l - state2.l) < 2

    def matelt_her_p_F(self, ket, omega_s, e_s, n_s):
        # TODO: Make sure this should actually just be the conjugate,
        # as I have assumed
        return np.conj(
            self.matelt_her_m_I(bra=ket, omega_l=omega_s, e_l=e_s, n_l=n_s)
        )

    def matelt_her_m_I(self, bra, omega_l, e_l, n_l):
        """See Chamberlain (23)
        :param bra:
        :return:
        """
        elec, hole = bra
        return (
            1/self.free_electron_mass * np.sqrt(2*np.pi/omega_l) / n_l /
            np.sqrt(self.volume) * np.vdot(e_l, self.p_cv()) *
            self._integ_e_rad_e(bra=elec, ket=hole, omega=omega_l)
        )

    def matelt_hep(self, bra, ph, ket):
        """See Chamberlain (25)
        :param bra: electron-hole state
        :param ph: phonon-state
        :param ket: electron-hole state
        """
        ebra, hbra = bra
        eket, hket = ket
        # The m and l requirements follow from selection rules
        if hbra == hket:
            m = self._integ_e_ph_e(bra=ebra, phonon=ph, ket=eket)
        elif ebra == eket:
            m = -self._integ_e_ph_e(bra=hbra, phonon=ph, ket=hket)
        else:
            m = 0
        return self.C_F / np.sqrt(self.R) * m

    def matelt_hec(self, bra, ket, cav):
        ebra, hbra = bra
        eket, hket = ket
        if hbra == hket:
            melt = self._integ_e_cav_e(bra=ebra, ket=eket, cav=cav)
        elif ebra == eket:
            melt = -self._integ_e_cav_e(bra=hbra, ket=hket, cav=cav)
        else:
            melt = 0
        return melt

    def p_cv(self):
        return np.array([1, 0, 0])  # TODO

    # --- Integrals ---
    def _get_key_e_cav_e(self, bra, mid, ket):
        return (
            bra.band, bra.l, bra.m, bra.n,
            ket.band, ket.l, ket.m, ket.n,
            mid.n
        )

    def _integ_e_cav_e(self, bra, ket, cav):
        return (
            self.cav_space.g_e * self.cav_space.area / self.cav_space.length**2
        )
        # return _integ_matelt(
        #     bra=bra, mid=cav, ket=ket,
        #     keyfunc=self._get_key_e_cav_e,
        #     storedict=self._e_cav_e_dict,
        #     oper=self.cav_space.potential(cav),
        #     intfunc=self._integ_e_op_e_volume_spherical,
        # )

    def _integ_e_ph_e(self, bra, phonon, ket):
        return (
            self._integ_e_ph_e_radial(bra=bra, ket=ket, phonon=phonon) *
            self._integ_e_ph_e_angular(bra=bra, ket=ket, phonon=phonon)
        )

    def _get_key_e_ph_e_radial(self, bra, mid, ket):
        return bra.band, bra.l, bra.n, ket.band, ket.l, ket.n, mid.l, mid.n

    def _integ_e_ph_e_radial(self, bra, ket, phonon):
        return _integ_matelt(
            bra=bra, mid=phonon, ket=ket,
            keyfunc=self._get_key_e_ph_e_radial,
            storedict=self._e_ph_e_radial_dict,
            oper=self.ph_space.Phi_ln(phonon),
            intfunc=self._integ_e_op_e_radial
        )

    def _get_key_e_ph_e_angular(self, bra, mid, ket):
        return min(
            (bra.l, bra.m, ket.l, ket.m, mid.l, mid.m),
            (bra.l, bra.m, mid.l, mid.m, ket.l, ket.m)
        )

    def _integ_e_ph_e_angular(self, bra, ket, phonon):
        return _integ_matelt(
            bra=bra, mid=phonon, ket=ket,
            keyfunc=self._get_key_e_ph_e_angular,
            storedict=self._e_ph_e_angular_dict,
            oper=Y_lm(l=phonon.l, m=phonon.m),
            intfunc=self._integ_e_op_e_angular
        )

    def _integ_e_rad_e(self, bra, ket, omega):
        r_k0, r_k1 = self._integ_e_rad_e_radial(bra=bra, ket=ket)
        a_k0, a_k1 = self._integ_e_rad_e_angular(bra=bra, ket=ket)
        return r_k0 * a_k0 + 1j * omega * r_k1 * a_k1

    def _get_key_e_rad_e_radial(self, bra, mid, ket):
        return bra.band, bra.l, bra.n, ket.band, ket.l, ket.n, mid

    def _integ_e_rad_e_radial(self, bra, ket):
        order_k0 = _integ_matelt(
            bra=bra, mid=0, ket=ket,
            keyfunc=self._get_key_e_rad_e_radial,
            storedict=self._e_rad_e_radial_dict,
            oper=lambda r: 1,
            intfunc=self._integ_e_op_e_radial
        )
        order_k1 = _integ_matelt(
            bra=bra, mid=1, ket=ket,
            keyfunc=self._get_key_e_rad_e_radial,
            storedict=self._e_rad_e_radial_dict,
            oper=lambda r: r,
            intfunc=self._integ_e_op_e_radial
        )
        # order_k1 = 0
        return order_k0, order_k1

    def _get_key_e_rad_e_angular(self, bra, mid, ket):
        return bra.l, bra.m, ket.l, ket.m, mid

    def _integ_e_rad_e_angular(self, bra, ket):
        order_k0 = _integ_matelt(
            bra=bra, mid=0, ket=ket,
            keyfunc=self._get_key_e_rad_e_angular,
            storedict=self._e_rad_e_angular_dict,
            oper=lambda the, phi: 1,
            intfunc=self._integ_e_op_e_angular
        )
        order_k1 = _integ_matelt(
            bra=bra, mid=1, ket=ket,
            keyfunc=self._get_key_e_rad_e_angular,
            storedict=self._e_rad_e_angular_dict,
            oper=lambda the, phi: basis_spherical(the, phi)[0][2],
            intfunc=self._integ_e_op_e_angular
        )
        # order_k1 = 0
        return order_k0, order_k1

    def _integ_e_op_e_radial(self, bra, ket, oper):
        wf_bra = self.ex_space.wavefunction_envelope_radial(bra)
        wf_ket = self.ex_space.wavefunction_envelope_radial(ket)

        def int_dr(r):
            return r**2 * np.conj(wf_bra(r)) * oper(r) * wf_ket(r)
        i_real = integ.quad(lambda r: int_dr(r).real, 0, np.inf)[0]
        i_imag = integ.quad(lambda r: int_dr(r).imag, 0, np.inf)[0]
        return i_real + 1j * i_imag

    def _integ_e_op_e_angular(self, bra, ket, oper):
        def int_dS(theta, phi):
            wf_bra = Y_lm(l=bra.l, m=bra.m)(theta, phi)
            wf_ket = Y_lm(l=ket.l, m=ket.m)(theta, phi)
            return np.conj(wf_bra) * oper(theta, phi) * wf_ket * np.sin(theta)
        ranges = [(0, np.pi), (0, 2 * np.pi)]
        i_real = integ.nquad(lambda t, p: int_dS(t, p).real, ranges=ranges)[0]
        i_imag = integ.nquad(lambda t, p: int_dS(t, p).imag, ranges=ranges)[0]
        return i_real + 1j * i_imag

    def _integ_e_op_e_volume_spherical(self, bra, ket, oper):
        """Spherical volume integral over all space
        :param bra: electron or hole state
        :param ket: electron or hole state
        :param oper: scalar function of spherical coordinates (r, theta, phi)
        """
        def int_dV(r, theta, phi):
            dV = np.sin(theta) * r**2
            wf_bra = self.ex_space.wavefunction_envelope(bra)(r, theta, phi)
            wf_ket = self.ex_space.wavefunction_envelope(ket)(r, theta, phi)
            return np.conj(wf_bra) * oper(r, theta, phi) * wf_ket * dV
        ranges = [(0, np.inf), (0, np.pi), (0, 2 * np.pi)]
        i_real = integ.nquad(lambda r, th, ph: int_dV(r, th, ph).real,
                             ranges=ranges)[0]
        i_imag = integ.nquad(lambda r, th, ph: int_dV(r, th, ph).imag,
                             ranges=ranges)[0]
        return i_real + 1j * i_imag

    # --- Lifetimes --
    def Gamma_ehp(self, ehp_state):
        # TODO
        estate, hstate = ehp_state
        Gamma_e = self.gamma_e
        Gamma_h = self.gamma_h
        return Gamma_e + Gamma_h

    def Gamma_ph(self, phonon_state):
        # TODO
        return self.gamma_ph

    def Gamma_cav_hi(self):
        # TODO
        return self.gamma_cav * self.cav_space.length**2 / self.cav_space.area

    def Gamma_cav_lo(self):
        return 0  # TODO

    def delta(self, omega_s, omega_l, phonon_state):
        omega_ph = self.ph_space.get_omega(phonon_state)
        gamma = self.Gamma_ph(phonon_state)
        return gamma / (abs(omega_l - omega_s - omega_ph)**2 + abs(gamma)**2)

    def delta_cav(self, omega_s, omega_l, phonon_state, final_cavity):
        omega_ph = self.ph_space.get_omega(phonon_state)
        gamma = self.Gamma_ph(phonon_state)
        omega_cav_hi = self.cav_space.omega_hi
        omega_cav_lo = self.cav_space.omega_lo
        if final_cavity == self.cav_space.hi:
            d_omega_cav = omega_cav_hi - omega_cav_lo
        else:
            d_omega_cav = omega_cav_lo - omega_cav_hi
        return gamma / (
            abs(omega_l - omega_s - omega_ph - d_omega_cav)**2 + abs(gamma)**2
        )
