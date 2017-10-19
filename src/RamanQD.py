"""RamanQD.py
Definitions for creating a Raman spectra for a quantum dot according to
the definitions in Riera
"""
import numpy as np
import itertools as it
from scipy import integrate as integ
from scipy import optimize as opt
from scipy import linalg as lin
from HamiltonianQD import J, K
from HamiltonianQD import plot_dispersion_function
from matplotlib import pyplot as plt
import qutip as qt
from collections import namedtuple


__all__ = ['ModelSpaceElectronHolePair', 'RamanQD', 'PhysicalConstants']


def _lc_tensor(a, b, c):
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


def Pi(k, p, l, m):
    if k == 1 and p == 1:
        return -np.sqrt(
            (l + m + 2) * (l + m + 1) / 2 / (2 * l + 1) / (2 * l + 3))
    elif k == 1 and p == 2:
        return -Pi(k=1, p=1, l=l, m=-m)
    elif k == 1 and p == 3:
        return np.sqrt(
            (l - m + 1) * (l + m + 1) / (2 * l + 1) / (2 * l + 3)
        )
    elif k == 2 and p == 1:
        return -Pi(k=1, p=1, l=l-1, m=-m-1)
    elif k == 2 and p == 2:
        return -Pi(k=2, p=1, l=l, m=-m)
    elif k == 2 and p == 3:
        return Pi(k=1, p=3, l=l-1, m=m)
    else:
        raise RuntimeError


PhysicalConstants = namedtuple('PhysicalConstants', ['e'])


class ModelSpaceElectronHolePair:
    def __init__(self, num_n):
        self.dim_n = num_n

    def iter_states_SP(self, num_l):
        for l, n in it.product(range(num_l), range(self.dim_n)):
            for m in range(-l, l+1):
                yield (l, m, n), self.make_state_SP(l=l, m=m, n=n)

    def iter_states_EHP(self, num_l, num_l2=None):
        if num_l2 is None:
            num_l2 = num_l
        for l1, l2 in it.product(range(num_l), range(num_l2)):
            for m1, m2 in it.product(range(-l1, l1+1), range(-l2, l2+1)):
                for n1, n2 in it.product(self.dim_n, repeat=2):
                    yield (
                        (l1, m1, n1, l2, m2, n2),
                        self.make_state_EHP(l1, m1, n1, l2, m2, n2)
                    )

    def make_state_SP(self, l, m, n):
        """Make single-particle state
        """
        return qt.tensor(
            qt.fock_dm(self.dim_n, n),
            qt.spin_state(j=l, m=m, type='dm')
        )

    def make_state_EHP(self, l1, m1, n1, l2, m2, n2):
        return qt.tensor(
            self.make_state_SP(l1, m1, n1),
            self.make_state_SP(l2, m2, n2)
        )

    def zero_SP(self, l):
        return qt.qzero([self.dim_n, 2*l+1])

    def zero_EHP(self, l1, l2):
        return qt.tensor(self.zero_SP(l1), self.zero_SP(l2))

    def one_SP(self, l):
        return qt.qeye([self.dim_n, 2*l+1])

    def one_EHP(self, l1, l2):
        return qt.tensor(self.one_SP(l1), self.one_SP(l2))

    def am_SP(self, l):
        """Single particle annihilation operator
        """
        return qt.tensor(qt.destroy(self.dim_n), qt.qeye(2*l+1))

    def am_EHP(self, j, l1, l2=None):
        if j == 1:  # electron state
            return qt.tensor(
                self.am_SP(l1), self.one_SP(l1)
            )
        else:  # hole state
            if l2 is None:
                l2 = l1
            return qt.tensor(
                self.one_SP(l2), self.am_SP(l2)
            )

    def Lm_SP(self, l):
        return qt.tensor(qt.qeye(self.dim_n), qt.spin_Jm(j=l))

    def Lm_EHP(self, j, l1, l2=None):
        if j == 1:  # electron state
            return qt.tensor(
                self.Lm_SP(l1), self.one_SP(l1)
            )
        else:  # hole state
            if l2 is None:
                l2 = l1
            return qt.tensor(
                self.one_SP(l2), self.Lm_SP(l2)
            )


class RamanQD:
    def __init__(self, nmax, lmax, unit_cell, refidx, radius,
                 V1, V2, E_g, mu_i1, mu_i2, mu_01, mu_02,
                 free_electron_mass, electron_charge,
                 Gamma_f, Gamma_a1, Gamma_a2, expected_roots_x_lj,
                 verbose=False):
        # Model constants
        self._num_n = nmax + 1
        self._num_l = lmax + 1
        self.ms = ModelSpaceElectronHolePair(num_n=self._num_n)

        # j-dependent physical constants
        self._V_j = [V1, V2]
        self._Gamma_aj = [Gamma_a1, Gamma_a2]
        self._mu_ij = [mu_i1, mu_i2]  # QDOT effective mass?
        self._mu_0j = [mu_01, mu_02]  # Environment effective mass

        # Physical constants
        self._a_matrix = unit_cell
        self.E_g = E_g  # Gap energy
        self.r_0 = radius
        self.mu_0 = free_electron_mass  # Free electron mass
        self.e = abs(electron_charge)
        self.Gamma_f = Gamma_f

        # Functions
        self.eta = refidx  # omega -> refractive index

        # Derived physical constants
        self.mu_r = 1/(1 / self.mu_ij(1) + 1 / self.mu_ij(2))  # Riera (92)
        self.E_0 = 1 / 2 / self.mu_r / self.r_0**2  # Riera (151)
        self.delta_f = self.Gamma_f / self.E_0  # Riera (152)
        self.V = 4/3 * np.pi * self.r_0**2  # QD Volume
        self._p_cv0 = None

        # Root finding
        self._x = dict()  # l, n, j -> x
        self._y = dict()  # l, n, j -> y
        self._expected_roots_x = expected_roots_x_lj
        self._fill_roots_xy()

        # Matrix elements
        self._m_jpeh = dict()

        # Other
        self.verbose = verbose

    def _print(self, string):
        if self.verbose:
            print('  >> {}'.format(string))

    # -- Raman cross section --
    def cross_section(self, omega_l, e_l, omega_s, e_s):
        """Differential Raman scattering cross section. See Riera (201).
        Note: I have removed the factor of sigma_0 from this calculation.
        If the user wants to see the true cross section, the same should
        multilply the result of this function with that of 'sigma_0'
        """
        self._print('Obtaining cross section for')
        self._print('  omega_l = {}'.format(omega_l))
        self._print('  omega_s = {}'.format(omega_s))
        self._print('  e_l = {}'.format(e_l))
        self._print('  e_s = {}'.format(e_s))
        cs = 0
        for p in range(4):
            self._print('Calculating cross section for p = {}'.format(p))
            cs += self._cross_section_p(omega_l=omega_l, omega_s=omega_s,
                                        e_s=e_s, p=p)
        t0 = (27/4 * (omega_s/self.E_0)**2 * self.delta_f)
        ans = t0 * cs
        self._print('Cross section = {}'.format(ans))
        self._print('Done\n')
        return ans

    def _cross_section_p(self, omega_l, omega_s, e_s, p):
        """See Riera (202, 203). Note that the computation of the common
        prefactor containing sigma_0, which is present in Riera (202, 203)
        has been moved to cross_section to avoid multiple computations
        """
        t1 = 0
        for se, sh in it.product(self.states_SP(), repeat=2):
            g2 = self.g2(x1=self.x(j=1, state=se),
                         x2=self.x(j=2, state=sh),
                         omega_l=omega_l, omega_s=omega_s)
            t2 = self.S(p=p, e_s=e_s) / (g2**2 + self.delta_f**2)
            if p == 0:
                M11 = self.Mj(1, 1, estate=se, hstate=sh, omega_s=omega_s)
                M12 = self.Mj(1, 2, estate=se, hstate=sh, omega_s=omega_s)
                M21 = self.Mj(2, 1, estate=se, hstate=sh, omega_s=omega_s)
                M22 = self.Mj(2, 2, estate=se, hstate=sh, omega_s=omega_s)
                t1 += t2 * (
                    M11.real * M22.real + M12.real * M21.real +
                    M11.imag * M22.imag + M12.imag * M21.imag
                )
            else:
                M1p = self.Mj(1, p, estate=se, hstate=sh, omega_s=omega_s)
                M2p = self.Mj(2, p, estate=se, hstate=sh, omega_s=omega_s)
                t1 += abs(M1p + M2p)**2 * t2
        return t1

    def Mj(self, j, p, estate, hstate, omega_s):
        """See Riera (204) and (205)
        """
        enums = self.get_numbers(state=estate)
        hnums = self.get_numbers(state=hstate)
        if (j, p, enums, hnums) in self._m_jpeh:
            mm = 0
            for num, denom in self._m_jpeh[j, p, enums, hnums]:
                mm += num / (denom + omega_s/self.E_0)
            return mm
        if j == 1:
            l1i, m1i, n1i = enums
            l2i, m2i, n2i = hnums
        else:
            l1i, m1i, n1i = hnums
            l2i, m2i, n2i = enums
        fracs = list()
        for festate in self.states_SP():
            l1f, m1f, n1f = self.get_numbers(state=festate)
            # Continue if zero (check delta functions)
            if m1f != m2i or l1f != l2i:
                continue
            elif p == 1 and m1f != m1i + 1:
                continue
            elif p == 2 and m1f != m1i - 1:
                continue
            elif p == 3 and m1f != m1i:
                continue
            # Evaluate terms
            if l1f == l1i + 1:
                t11_num = Pi(1, p, l=l1i, m=m1i)  # TODO: verify correctness
            elif l1f == l1i - 1:
                t11_num = Pi(2, p, l=l1i, m=m1i)  # TODO: verify correctness
            else:
                continue
            t11_denom = (
                self.beta(j) *
                (self.x(l=l1i, n=n1i, j=j)**2 - self.x(l=l1f, n=n1f, j=j)**2)
                + 1j * self.Gamma_aj(j) / self.E_0
            )
            t12 = -self.beta(j) * (self.I(l1f, n1f, l1i, n1i, j=j) *
                                   self.T(l1f, n1f, n2i, j=j, nu=1/2))
            fracs.append((t11_num * t12, t11_denom))
        self._m_jpeh[j, p, enums, hnums] = fracs
        return self.Mj(j=j, p=p, estate=estate, hstate=hstate, omega_s=omega_s)

    # -- States --
    def states_SP(self):
        return self.ms.iter_states_SP(num_l=self._num_l)

    def states_EHP(self):
        return self.ms.iter_states_EHP(num_l=self._num_l)

    def get_numbers(self, state):
        return state[0]

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

    def g2(self, x1, x2, omega_l, omega_s):
        """See Riera (150)
        """
        return (
            (omega_l - self.E_g) / self.E_0 - omega_s / self.E_0 -
            self.beta(1) * x1**2 - self.beta(2) * x2**2
        )

    def p_cv0(self):
        """Momentum between valence and conduction bands, at k=0.
         See Riera (78).
        """
        if self._p_cv0 is not None:
            return self._p_cv0

        # The unit cell is defined by 3 vectors: a1, a2, a3, which
        # are the columns of...
        a_matrix = self._a_matrix

        # The function to integrate (in terms of the unit cell basis) is
        def intfn(c1, c2, c3, i):
            # Get position in cartesian basis
            x = np.dot(a_matrix, np.array([c1, c2, c3]))
            r = lin.norm(x, ord=2)
            # Evaluate u1 and u2 at r
            # TODO In Riera, this is written conj(u1'), but I don't know
            # what the prime is for
            u1_conj = np.conj(self.u_j(j=1)(r=r))
            du2 = self.u_j(j=2)(r=r, d_r=1)
            pu2 = x / r * du2
            return (u1_conj * pu2)[i]

        # Integrate real and imaginary parts separately
        def ifnr(c1, c2, c3, xi):
            return np.real(intfn(c1, c2, c3, i=xi))

        def ifni(c1, c2, c3, xi):
            return np.imag(intfn(c1, c2, c3, i=xi))

        # The integral is performed over the unit cell volume, 0 to 1 in
        # each coordinate
        x_re = integ.nquad(func=ifnr, ranges=[(0, 1)]*3, args=(0,))[0]
        x_im = integ.nquad(func=ifni, ranges=[(0, 1)]*3, args=(0,))[0]
        y_re = integ.nquad(func=ifnr, ranges=[(0, 1)]*3, args=(1,))[0]
        y_im = integ.nquad(func=ifni, ranges=[(0, 1)]*3, args=(1,))[0]
        z_re = integ.nquad(func=ifnr, ranges=[(0, 1)]*3, args=(2,))[0]
        z_im = integ.nquad(func=ifni, ranges=[(0, 1)]*3, args=(2,))[0]
        re = np.array([x_re, y_re, z_re])
        im = np.array([x_im, y_im, z_im])
        self._p_cv0 = re + 1j * im
        return self.p_cv0()

    def u_j(self, j):
        """Bloch function. I have not found the definition for this in
        Riera
        """
        def ujfn(r, d_r=0):
            return 1  # TODO
        return ujfn

    def sigma_0(self, omega_s, omega_l, e_l):
        """See Riera (104)
        """
        return (
            (4*np.sqrt(2) * self.V * self.e**4 *
             abs(np.dot(e_l, self.p_cv0()).sum())**2 * self.eta(omega_s) *
             self.mu_r**(1/2) * self.E_0**(3/2)) /
            (9*np.pi**2 * self.mu_0**2 * self.eta(omega_l) *
             omega_l * omega_s)
        )

    def Gamma_aj(self, j):
        return self._Gamma_aj[j-1]

    def V_j(self, j):
        return self._V_j[j-1]

    def mu_ij(self, j):
        return self._mu_ij[j-1]

    def mu_0j(self, j):
        return self._mu_0j[j-1]

    def beta(self, j):
        """See Riera (92)
        """
        return self.mu_r / self.mu_ij(j)

    # -- Coupling tensors --
    def I(self, lf, nf, li, ni, j, nu=1/2):
        r0 = self.r_0
        xi = self.x(l=li, n=ni, j=j)
        xf = self.x(l=lf, n=nf, j=j)
        yi = self.y(l=li, n=ni, j=j)
        yf = self.y(l=lf, n=nf, j=j)
        Jli = J(l=li+nu)
        Jlf = J(l=lf+nu)
        Kli = K(l=li+nu)
        Klf = K(l=lf+nu)

        def ifn_j(r):
            return Jli(z=xi*r/r0) * Jlf(z=xf*r/r0) * r
        ans_j_re = integ.quad(lambda x: ifn_j(x).real, a=0, b=self.r_0)[0]
        ans_j_im = integ.quad(lambda x: ifn_j(x).imag, a=0, b=self.r_0)[0]
        ans_j = ans_j_re + 1j * ans_j_im

        def ifn_k(r):
            return Kli(z=yi*r/r0) * Klf(z=yf*r/r0) * r
        ans_k_re = integ.quad(lambda x: ifn_k(x).real, a=self.r_0, b=np.inf)[0]
        ans_k_im = integ.quad(lambda x: ifn_k(x).imag, a=self.r_0, b=np.inf)[0]
        ans_k = ans_k_re + 1j * ans_k_im

        return (
            self.A(l=li, n=ni, j=j) * self.A(l=lf, n=nf, j=j) * ans_j +
            self.B(l=li, n=ni, j=j) * self.B(l=lf, n=nf, j=j) * ans_k
        )

    def T(self, l, na, nb, j, nu=1/2):
        return self.I(lf=l, nf=nb, li=l, ni=na, j=j, nu=nu)

    def _Jl_Kl(self, l, n, j):
        return (J(l=l+1/2)(self.x(l=l, n=n, j=j)),
                K(l=l+1/2)(self.y(l=l, n=n, j=j)))

    def A(self, l, n, j):
        Jl, Kl = self._Jl_Kl(l=l, n=n, j=j)
        return 1/np.sqrt(abs(Jl)**2 + (Jl/Kl)**2 * abs(Kl)**2)

    def B(self, l, n, j):
        Jl, Kl = self._Jl_Kl(l=l, n=n, j=j)
        return Jl/Kl * self.A(l=l, n=n, j=j)

    # -- Obtaining roots --
    def x(self, j, l=None, n=None, state=None):
        if l is not None and n is not None and (l, n, j) in self._x:
            return self._x[l, n, j]
        elif state is not None:
            l, m, n = self.get_numbers(state=state)
            return self.x(j=j, l=l, n=n)
        else:
            raise RuntimeError  # TODO

    def y(self, l, n, j):
        if (l, n, j) in self._y:
            return self._y[l, n, j]
        else:
            raise RuntimeError  # TODO

    def _fill_roots_xy(self):
        for j, l in it.product([1, 2], range(self._num_l)):
            for root, n in zip(self._solve_roots_xy(l=l, j=j),
                               range(self._num_n)):
                x, y = root
                assert not np.isnan(x)
                assert not np.isnan(y)
                self._x[l, n, j] = x
                self._y[l, n, j] = y

    def _solve_roots_xy(self, l, j):
        fun, jac = self.get_rootf_rootdf_xy(l=l, j=j)
        for x0y0 in self.expected_roots_xy(l=l, j=j):
            # assert opt.check_grad(func=fun, grad=jac, x0=x0y0)
            result = opt.root(fun=fun, jac=jac, x0=x0y0)
            if not result.success:
                print('FAILED')  # TODO
            else:
                xr, xi, yr, yi = result.x
                yield np.array([xr + 1j * xi, yr + 1j * yi])

    def _y2(self, x2, j):
        return (
            2 * self.mu_0j(j) * self.r_0 ** 2 * self.V_j(j) -
            self.mu_0j(j) / self.mu_ij(j) * x2
        )

    def plot_root_fn_x(self, l, j, xdat):
        fn = self.get_rootf_rootdf_xy(l=l, j=j)[0]

        def rootfn(xy):
            x, y = xy
            fr, fi, gr, gi = fn(np.array([x.real, x.imag, y.real, y.imag]))
            return np.array([fr + 1j * fi, gr + 1j * gi])

        def fpr(x):
            y = np.sqrt(self._y2(x2=x**2, j=j))
            return np.real(rootfn(np.array([x, y])))[0]

        def fmr(x):
            y = -np.sqrt(self._y2(x2=x**2, j=j))
            return np.real(rootfn(np.array([x, y])))[0]

        def fpi(x):
            y = np.sqrt(self._y2(x2=x**2, j=j))
            return np.imag(rootfn(np.array([x, y])))[0]

        def fmi(x):
            y = -np.sqrt(self._y2(x2=x**2, j=j))
            return np.imag(rootfn(np.array([x, y])))[0]

        def ix():
            for n in range(self._num_n):
                return ()
                # yield self.x(l, n, j)

        fig, ax = plt.subplots(1, 1)
        # plot_dispersion_function(
        #     xdat=xdat, rootfn=fpr, iterfn=ix(), fig=fig, ax=ax, show=False)
        # plot_dispersion_function(
        #     xdat=xdat, rootfn=fpi, iterfn=ix(), fig=fig, ax=ax, show=False)
        # plot_dispersion_function(
        #     xdat=xdat, rootfn=fmr, iterfn=ix(), fig=fig, ax=ax2, show=False)
        # plot_dispersion_function(
        #     xdat=xdat, rootfn=fmi, iterfn=ix(), fig=fig, ax=ax2, show=False)

        ax.axhline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(0, color='gray', lw=1, alpha=.5)

        for n in range(self._num_n):
            try:
                x = self.x(j=j, l=l, n=n)
                ax.axvline(x.real, ls='--', color='green', lw=1, alpha=.5)
            except RuntimeError:
                print('crap')

        ydat = np.array([fpr(x) for x in xdat])
        ydat /= lin.norm(ydat, ord=2)
        ax.plot(xdat, ydat, '-', color='red')

        ylog = np.log(np.abs(ydat))
        ylog /= lin.norm(ylog, ord=2)
        ax.plot(xdat, ylog, '--', color='red')

        ypr = np.real(np.sqrt([self._y2(x2=x**2, j=j) for x in xdat]))
        ypr /= lin.norm(ypr, ord=2)
        ax.plot(xdat, ypr, '-', color='blue')
        # ax.set_ylim(bottom=0)

        plt.show()
        return fig, ax

    def expected_roots_xy(self, l, j):
        for x0 in self._expected_roots_x[l, j]:
            x0 = complex(x0)
            y0 = np.sqrt(self._y2(x2=x0**2, j=j))
            yield np.array([x0.real, x0.imag, y0.real, y0.imag])
            # yield np.array([x0.real, x0.imag, -y0.real, -y0.imag])

    def get_rootf_rootdf_xy(self, l, j):
        fun_f = self._get_rootf1(l=l, j=j)
        fun_g = self._get_rootf2(j=j)

        def pi(zz):
            z1, z2 = zz
            return np.array([z1.real, z1.imag, z2.real, z2.imag])

        def ipi(xxyy):
            xr, xi, yr, yi = xxyy
            return np.array([xr+1j*xi, yr+1j*yi])

        dipi = np.array([[1, 1j, 0, 0],
                         [0, 0, 1, 1j]])
        dpi = dipi.transpose().conjugate()

        def psi(zz):
            z1, z2, = zz
            f = complex(fun_f(z1, z2))
            g = complex(fun_g(z1, z2))
            return np.array([f, g])

        def dpsi(zz):
            z1, z2 = zz
            fz1 = complex(fun_f(z1, z2, partial_x=1))
            fz2 = complex(fun_f(z1, z2, partial_y=1))
            gz1 = complex(fun_g(z1, z2, partial_x=1))
            gz2 = complex(fun_g(z1, z2, partial_y=1))
            return np.array([[fz1, fz2],
                             [gz1, gz2]])

        def rf(xxyy, *args, **kwargs):
            return pi(psi(ipi(xxyy)))

        def rdf(xxyy, *args, **kwargs):
            return dpi.dot(dpsi(ipi(xxyy)).dot(dipi))

        return rf, None

    def _get_rootf1(self, l, j):
        l = l+1/2
        mu0 = self.mu_0j(j)
        mui = self.mu_ij(j)

        def rf1(x, y, partial_x=0, partial_y=0):
            J_x = J(l)(x)
            dJ_x = J(l, d=1)(x)
            K_y = K(l)(y)
            dK_y = K(l, d=1)(y)
            if partial_x == 0 and partial_y == 0:
                return (
                    mu0 * x * dJ_x * K_y - mui * y * J_x * dK_y
                )
            elif (partial_x, partial_y) == (1, 0):
                d2J_x = J(l, d=2)(x)
                return (
                    mu0 * (dJ_x + x * d2J_x) * K_y - mui * y * dJ_x + dK_y
                )
            elif (partial_x, partial_y) == (0, 1):
                d2K_y = K(l, d=2)(y)
                return (
                    mu0 * x * dJ_x * dK_y - mui * (dK_y + y * d2K_y) * J_x
                )
            else:
                raise RuntimeError  # TODO
        return rf1

    def _get_rootf2(self, j):
        mu0 = self.mu_0j(j)
        mui = self.mu_ij(j)
        vj = self.V_j(j)

        def _fx2(x, deriv=0):
            if deriv == 0:
                return x**2
            elif deriv == 1:
                return 2*x
            elif deriv == 2:
                return 2
            elif deriv > 2:
                return 0

        def rf2(x, y, partial_x=0, partial_y=0):
            if partial_x == 0 and partial_y == 0:
                tc = 2 * mu0 * self.r_0**2 * vj
            else:
                tc = 0
            # TODO: verify formula
            return (
                _fx2(y, deriv=partial_y) +
                mu0 / mui * _fx2(x, deriv=partial_x) - tc)

        return rf2

