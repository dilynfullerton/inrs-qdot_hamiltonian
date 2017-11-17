"""CavityModelSpace.py
Definition for a simple harmonic optical cavity, which provides
functions for interactions with phonon modes and electron/hole modes
"""
import numpy as np
from ModelSpace import ModelSpace
from collections import namedtuple


CavityMode = namedtuple('CavityMode', ['n'])


class CavityModelSpace(ModelSpace):
    def __init__(self, l, w, h, omega_c, g_e):
        self.length = l  # space between antennae
        self.width = w
        self.height = h
        self.area = 2 * l * (w + h)  # exposed area
        self.omega_c = omega_c
        self.g_e = g_e
        self.lo = CavityMode(n=0)
        self.hi = CavityMode(n=1)

    def nfock(self, mode):
        return 2

    def get_nums(self, mode):
        return mode.n,

    def get_ket(self, mode):
        pass

    def states(self):
        return self.modes()

    def modes(self):
        yield self.lo
        yield self.hi

    def get_omega(self, mode):
        if mode.n == 1:
            return self.omega_c
        else:
            return 0

    def potential(self, mode):
        # TODO: Find a theoretically-justified potential
        def phi(x):
            omega = self.get_omega(mode)
            if mode.n == 1:
                return self.g_e * self.length / omega / 2 * (
                    np.exp(self.get_omega(mode) * (-x/self.length - 1/2)) +
                    np.exp(self.get_omega(mode) * (x/self.length - 1/2))
                )
            else:
                return 0
        return phi

