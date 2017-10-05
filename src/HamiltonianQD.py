"""HamiltonianQD.py
Definition of a macroscopic, continuous Hamiltonian for describing the
interaction of the phonon modes of a quantum dot with an electromagnetic
field.

The definitions used in the theoretical model are based on
Sec. III of "Polar optical vibrations in quantum dots" by Roca et. al.
"""
from math import sqrt, pi
import itertools as it
import qutip as qt


class HamiltonianRoca:
    def __init__(self):
        self.R = 0  # TODO

    def h(self, r, theta, phi):
        """Electron-phonon interaction Hamiltonian at spherical coordinates
        (r, theta, phi). The definition is based on (Roca, Eq. 42).
        """
        h = 0
        for nu_n, n in zip(self._gen_nu(), it.count()):
            for l, m in self._lm():
                h += (
                    self._C_F(r, theta, phi) * self.R / nu_n * (2*l + 1) *
                    1j**l * sqrt(2*pi) * self._Phibar(nu_n, l)(r) *
                    _Y(l, m)(theta, phi) * (self._b(n) + self._b(-n).dag())
                )
        return h

    def _gen_nu(self):
        yield 0  # TODO

    def _lm(self):
        yield 0, 0  # TODO

    def _C_F(self, r, theta, phi):
        return 0  # TODO

    def _Phibar(self, nu, l):
        def _fn_phibar(r):
            return 0  # TODO
        return _fn_phibar

    def _b(self, n):
        return qt.qzero(2)  # TODO
