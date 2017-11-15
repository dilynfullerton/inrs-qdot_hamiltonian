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
    elif k in storedict[k]:
        return storedict[k]
    else:
        storedict[k] = intfunc(bra=bra, ket=ket, oper=oper)
        return storedict[k]


class RamanQD:
    def __init__(
            self, phonon_space, exiton_space, phonon_lifetime, ehp_lifetime
    ):
        # Model parameters
        self.ph_space = phonon_space
        self.ex_space = exiton_space

        # Lifetimes
        self.gamma_ph = phonon_lifetime
        self.gamma_ehp = ehp_lifetime

        # Convenience references
        self.volume = self.ph_space.volume
        self.free_electron_mass = self.ex_space.free_electron_mass
        self.C_F = self.ph_space.C_F
        self.R = self.ph_space.R

        # Matrix elements
        self._e_rad_e_radial_dict = dict()
        self._e_rad_e_angular_dict = dict()
        self._e_ph_e_radial_dict = dict()
        self._e_ph_e_angular_dict = dict()

    # -- Raman cross section --
    def differential_raman_cross_section(
            self, omega_l, e_l, n_l, k_l, omega_s, e_s, n_s, k_s):
        """See Chamberlain (1)
        """
        return (
            self.volume**2 * omega_s**3 * n_l * n_s**3 / 8 / np.pi**3 /
            omega_l * self.scattering_rate(
                omega_l=omega_l, e_l=e_l, n_l=n_l, k_l=k_l,
                omega_s=omega_s, e_s=e_s, n_s=n_s, k_s=k_s,
            )
        )

    def scattering_rate(self, omega_l, e_l, n_l, k_l, omega_s, e_s, n_s, k_s):
        """See Chamberlain (2)
        :param omega_s: Secondary frequency
        :param e_s: Secondary polarization
        """
        w = 0
        for phonon in self.ph_space.states():
            mfi = self.M_FI(omega_s=omega_s, e_s=e_s, n_s=n_s, k_s=k_s,
                            omega_l=omega_l, e_l=e_l, n_l=n_l, k_l=k_l,
                            phonon=phonon)
            delta = self.delta(omega_s=omega_s, omega_l=omega_l,
                               omega_ph=self.ph_space.get_omega(phonon))
            w += 2 * np.pi * abs(mfi)**2 * delta
        return w

    def M_FI(self, omega_s, e_s, n_s, k_s, omega_l, e_l, n_l, k_l, phonon):
        """See Chamberlain (3)
        :param omega_s:
        :param omega_l:
        :param e_s:
        :param e_l:
        :param phonon:
        :return:
        """
        mfi = 0
        for mu1, mu2 in it.product(
                self.ex_space.electron_hole_states(), repeat=2):
            numerator = (
                self.matelt_her_p_F(
                    ket=mu2, omega_s=omega_s, e_s=e_s, n_s=n_s, k_s=k_s) *
                self.matelt_hep(bra=mu2, mid=phonon, ket=mu1) *
                self.matelt_her_m_I(
                    bra=mu1, omega_l=omega_l, e_l=e_l, n_l=n_l, k_l=k_l)
            )
            denominator = (
                (omega_s - self.ex_space.get_omega(mu2) + self.Gamma_ehp(mu2)) *
                (omega_l - self.ex_space.get_omega(mu1) + self.Gamma_ehp(mu1))
            )
            mfi += numerator / denominator
        return mfi

    def matelt_her_p_F(self, ket, omega_s, e_s, n_s, k_s):
        # TODO: Make sure this should actually just be the conjugate,
        # as I have assumed
        return np.conj(
            self.matelt_her_m_I(
                bra=ket, omega_l=omega_s, e_l=e_s, n_l=n_s, k_l=k_s
            )
        )

    def matelt_her_m_I(self, bra, omega_l, e_l, n_l, k_l):
        """See Chamberlain (23)
        :param bra:
        :return:
        """
        elec, hole = bra
        return (
            1/self.free_electron_mass * np.sqrt(2*np.pi/omega_l) / n_l /
            np.sqrt(self.volume) * np.vdot(e_l, self.p_cv()) *
            self._integ_e_rad_e(bra=elec, ket=hole, k=k_l, omega=omega_l)
        )

    def matelt_hep(self, bra, mid, ket):
        """See Chamberlain (25)
        :param bra: electron-hole state
        :param mid: phonon-state
        :param ket: electron-hole state
        """
        ebra, hbra = bra
        eket, hket = ket
        m = 0
        if hbra == hket:
            m += self._integ_e_ph_e(bra=ebra, phonon=mid, ket=eket)
        if ebra == eket:
            m -= self._integ_e_ph_e(bra=hbra, phonon=mid, ket=hket)
        return self.C_F / np.sqrt(self.R) * m

    def p_cv(self):
        return np.array([1, 0, 0])  # TODO

    def _integ_e_ph_e(self, bra, phonon, ket):
        return (
            self._integ_e_ph_e_radial(bra=bra, ket=ket, phonon=phonon) *
            self._integ_e_ph_e_angular(bra=bra, ket=ket, phonon=phonon)
        )

    def _get_key_e_ph_e_radial(self, bra, mid, ket):
        return bra.l, bra.n, ket.l, ket.n, mid.l, mid.n

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

    def _integ_e_rad_e(self, bra, ket, k, omega):
        r_k0, r_k1 = self._integ_e_rad_e_radial(bra=bra, ket=ket)
        a_k0, a_k1 = self._integ_e_rad_e_angular(bra=bra, ket=ket, k=k)
        return r_k0 * a_k0 + 1j * omega * r_k1 * a_k1

    def _get_key_e_rad_e_radial(self, bra, mid, ket):
        return bra.l, bra.n, ket.l, ket.n, mid

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
            keyfunc=self._get_key_e_ph_e_radial,
            storedict=self._e_rad_e_radial_dict,
            oper=lambda r: r,
            intfunc=self._integ_e_op_e_radial
        )
        return order_k0, order_k1

    def _get_key_e_rad_e_angular(self, bra, mid, ket):
        return bra.l, bra.m, ket.l, ket.m, mid

    def _integ_e_rad_e_angular(self, bra, ket, k):
        order_k0 = _integ_matelt(
            bra=bra, mid=0, ket=ket,
            keyfunc=self._get_key_e_rad_e_angular,
            storedict=self._e_rad_e_angular_dict,
            oper=lambda the, phi: 1,
            intfunc=self._integ_e_op_e_angular
        )
        order_k1 = _integ_matelt(
            bra=bra, mid=1, ket=ket,
            keyfunc=self._get_key_e_ph_e_angular,
            storedict=self._e_rad_e_angular_dict,
            oper=lambda the, phi: np.vdot(k, basis_spherical(the, phi)[0]),
            intfunc=self._integ_e_op_e_angular
        )
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

    def Gamma_ehp(self, ehp_state):
        return self.gamma_ehp  # TODO

    def Gamma_ph(self, phonon_state):
        return self.gamma_ph  # TODO

    def delta(self, omega_s, omega_l, phonon_state):
        omega_ph = self.ph_space.get_omega(phonon_state)
        gamma = self.Gamma_ph(phonon_state)
        return gamma / ((omega_l - omega_s - omega_ph)**2 + gamma**2)
