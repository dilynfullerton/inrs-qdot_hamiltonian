import numpy as np
import qutip as qt
from scipy import special as sp


def Y_lm(l, m):
    def yfn(theta, phi):
        return sp.sph_harm(m, l, phi, theta)
    return yfn


def J(l, d=0):
    def fn_j(z):
        z = complex(z)
        return sp.jvp(v=l, z=z, n=d)
    return fn_j


def K(l, d=0):
    def fn_k(z):
        z = complex(z)
        return sp.kvp(v=l, z=z, n=d)
    return fn_k


def _spherical_bessel1(l, func, z, dz=0):
    z = complex(z)
    if dz == 0:
        return func(n=l, z=z)
    elif l == 0:
        return -_spherical_bessel1(l=1, func=func, z=z, dz=dz-1)
    else:
        k = dz - 1
        jk = _spherical_bessel1(l=l-1, func=func, z=z, dz=k)
        rest = 0
        for m in range(k+1):
            rest += (
                (-1)**m * sp.factorial(m) * z**(-m-1) * sp.binom(k, m) *
                _spherical_bessel1(l=l, func=func, z=z, dz=k-m)
            )
        return jk - (l + 1) * rest


def j_sph(l, d=0):
    def jz(z):
        return _spherical_bessel1(l=l, func=sp.spherical_jn, z=z, dz=d)
    return jz


def i_sph(l, d=0):
    def iz(z):
        return _spherical_bessel1(l=l, func=sp.spherical_in, z=z, dz=d)
    return iz


def g_sph(l, d=0):
    def fn_g(z):
        z = complex(z)
        if (z**2).real >= 0:
            j = j_sph(l, d)(z)
            return j
        else:
            i = i_sph(l, d)(z)
            return i
    return fn_g


def levi_civita(a, b, c):
    if a < b < c or b < c < a or c < a < b:
        return 1
    elif a < c < b or b < a < c or c < b < a:
        return -1
    else:
        return 0


def basis_cartesian():
    return np.eye(3, 3)


def basis_pmz():
    isq2 = 1/np.sqrt(2)
    return np.array([[isq2,  1j*isq2, 0],
                     [isq2, -1j*isq2, 0],
                     [0,     0,       1]])


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


def threej(j1, j2, j, m1, m2, m):
    return (-1)**(j1-j2-m)/np.sqrt(2*j + 1) * qt.clebsch(j1, j2, j, m1, m2, -m)
