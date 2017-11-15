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


def _spherical_bessel1(l, func, z, dz=0, kind=0):
    """Compute the spherical bessel given by func_l at z, where dz specifies
    the number of derivatives.
    :param kind: integer specifying the type of bessel function
      0 = The derivative formulas are valid for functions j, y, h1, and h2
      1 = The derivative formulas are valid for functions k
    """
    z = complex(z)
    if dz == 0:
        return func(n=l, z=z)
    elif l == 0:
        return -_spherical_bessel1(l=1, func=func, z=z, dz=dz-1, kind=kind)
    else:
        jk = _spherical_bessel1(l=l-1, func=func, z=z, dz=dz-1, kind=kind)
        rest = 0
        for m in range(dz):
            rest += (
                (-1)**(m % 2) * sp.factorial(m) * z**(-m-1) *
                sp.binom(dz-1, m) *
                _spherical_bessel1(l=l, func=func, z=z, dz=dz-1-m, kind=kind)
            )
        if kind == 0:
            return jk - (l + 1) * rest
        else:
            return -jk - (l + 1) * rest


def j_sph(l, d=0):
    def jz(z):
        return _spherical_bessel1(l=l, func=sp.spherical_jn, z=z, dz=d,
                                  kind=0)
    return jz


def i_sph(l, d=0):
    def iz(z):
        return _spherical_bessel1(l=l, func=sp.spherical_in, z=z, dz=d,
                                  kind=0)
    return iz


def k_sph(l, d=0):
    def kz(z):
        return _spherical_bessel1(l=l, func=sp.spherical_kn, z=z, dz=d,
                                  kind=1)
    return kz


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
    return np.array([[isq2,    isq2,     0],
                     [1j*isq2, -1j*isq2, 0],
                     [0,       0,        1]])


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
    return np.array([[xr,  yr,  zr],
                     [xth, yth, zth],
                     [xph, yph, zph]])


def basis_roca(theta, phi, l, m):
    er, eth, eph = basis_spherical(theta, phi)
    if l == 0:
        xlm = np.zeros_like(er)
    else:
        xlm = m*(2*l+1)/l/(l+1) * 1j * Y_lm(l, m)(theta, phi) * eth
        xlm += -(l-m+1)/(l+1) * Y_lm(l+1, m)(theta, phi) * eph
        if m != -l:
            xlm += (l+m)/l * Y_lm(l-1, m)(theta, phi) * eph
        xlm *= np.sqrt(l * (l + 1)) / (2 * l + 1) / np.sin(theta)
    er_xlm = np.cross(er, xlm)
    return np.vstack((er, xlm, er_xlm))


def threej(j1, j2, j, m1, m2, m):
    return (-1)**(j1-j2-m)/np.sqrt(2*j + 1) * qt.clebsch(j1, j2, j, m1, m2, -m)
