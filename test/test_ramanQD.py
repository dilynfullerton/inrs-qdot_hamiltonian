import unittest
from unittest import TestCase
import numpy as np
from src.RamanQD import *

LATTICE_CONST_GaAs = .05635  # [nm]
REFIDX_GaAs = 3.8

ELECTRON_CHARGE = 1239.84193  # [nm]
FREE_ELECTRON_MASS = 0.00243  # [nm]


class TestRamanQD(TestCase):
    def setUp(self):
        gamma_a1 = 1239841.93009  # [nm]
        self.qd_GaAs = RamanQD(
            nmax=3, lmax=1,
            unit_cell=LATTICE_CONST_GaAs * np.eye(3, 3),
            refidx=lambda w: REFIDX_GaAs,
            radius=5.0,  # [nm]
            V1=1280.82844,  # [nm]
            V2=1921.34190,  # [nm]
            E_g=816.92161,  # [nm]
            mu_i1=.0665,  # [mu_0]
            mu_i2=.45,  # [mu_0]
            mu_01=1,  # [mu_0]
            mu_02=1,  # [mu_0]
            electron_charge=ELECTRON_CHARGE,
            free_electron_mass=FREE_ELECTRON_MASS,
            Gamma_f=gamma_a1*3,
            Gamma_a1=gamma_a1,
            Gamma_a2=gamma_a1,
            expected_roots_x_lj={
                (0, 1): [.001, 3.0, 5.86, 8.9],
                (0, 2): [.001, 3.1, 6.2462, 9.4],
                (1, 1): [.001, 4.2, 7.3, 10.3],
                (1, 2): [.001, 4.4, 7.67016736, 10.9],
            }
        )

    def test_instatiation(self):
        self.assertIsInstance(self.qd_GaAs, RamanQD)

    def test_roots_plot(self):
        for l in range(2):
            for j in range(1, 3):
                self.qd_GaAs.plot_root_fn_x(
                    l=l, j=j, xdat=np.linspace(1e-6, 15, 10000))
        self.assertFalse(False)
