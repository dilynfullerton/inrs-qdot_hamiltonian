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
            j = _j(l, d)(z)
            return j
        else:
            i = _i(l, d)(z)
            return i
    return fn_g