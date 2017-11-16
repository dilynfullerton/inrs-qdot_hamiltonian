"""CavityModelSpace.py
Definition for a simple harmonic optical cavity, which provides
functions for interactions with phonon modes and electron/hole modes
"""
from ModelSpace import ModelSpace
from collections import namedtuple


CavityMode = namedtuple('CavityMode', ['n'])


class CavityModelSpace(ModelSpace):
    def __init__(self, l, w, h, omega_c):
        self.length = l  # space between antennae
        self.width = w
        self.height = h
        self.area = 2 * l * (w + h)  # exposed area
        self.omega_c = omega_c

    def nfock(self, mode):
        return 2

    def get_nums(self, mode):
        return mode.n,

    def get_ket(self, mode):
        pass

    def states(self):
        return self.modes()

    def modes(self):
        yield CavityMode(n=0)
        yield CavityMode(n=1)

    def get_omega(self, mode):
        if mode.n == 1:
            return self.omega_c
        else:
            return 0
