import itertools as it
import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from scipy import integrate as integ
from scipy import linalg
from helper_functions import Y_lm, j_sph, k_sph
from root_solver_2d import RootSolverComplex2d
from ModelSpace import ModelSpace


class ElectronHoleModelSpace(ModelSpace):
    def __init__(self, nmax, lmax, wavefunctions):
        self.nmax = nmax
        self.lmax = lmax
        self.nlen = self.nmax + 1
        self.llen = self.lmax + 1
        self.wavefunctions = wavefunctions

    def nfock(self, mode):
        return 2

    def modes(self):
        for band in range(2):
            for l in range(self.llen):
                for m in range(-l, l+1):
                    for n in range(self.nlen):
                        yield (band, l, m, n)

    def states(self):
        return self.wavefunctions.iter_states()

    def destroy(self, mode):
        """Returns a destruction operator for the given mode
        """
        ops = []
        for m, i in zip(self.modes(), it.count()):
            if m == mode:
                ops.append(qt.sigmam())
            else:
                ops.append(qt.qeye(self.nfock(mode)))
        return qt.tensor(ops)

    def get_nums(self, state):
        return state

    def get_ket(self, state):
        return self.create(mode=state) * self.vacuum()

    def get_omega(self, state):
        # TODO: verify
        band, l, m, n = self.get_nums(state)
        mu_in = self.wavefunctions.mu_in(band)
        r0 = self.wavefunctions.r_0
        x = self.wavefunctions.x(state)
        e = 1/2 / mu_in * x**2 / r0**2
        return complex(e).real

    def get_wavefunction_rad(self, state):
        return self.wavefunctions.wavefunction_env_radial(state)

    def get_wavefunction_ang(self, state):
        return self.wavefunctions.wavefunction_env_angular(state)


class ElectronModelSpace(ElectronHoleModelSpace):
    def modes(self):
        for l in range(self.llen):
            for m in range(-l, l+1):
                for n in range(self.nlen):
                    yield (0, l, m, n)


class HoleModelSpace(ElectronHoleModelSpace):
    def modes(self):
        for l in range(self.llen):
            for m in range(-l, l+1):
                for n in range(self.nlen):
                    yield (1, l, m, n)


class ElectronHoleWavefunctions(RootSolverComplex2d):
    def __init__(
            self, r_0, V_v, V_c, me_eff_in, mh_eff_in, me_eff_out, mh_eff_out,
            expected_roots_x_lj, n_max, l_max
    ):
        """
        :param V_v: Effective potential (?) of the valence band
        :param V_c: Effective potential (?) of the conduction band
        :param E_gap: Gap energy between conduction and valence bands
        :param me_eff_in: Effective electron mass in nanostructures
        :param mh_eff_in: Effective hole mass in nanostructures
        :param me_eff_out: Free electron mass
        :param mh_eff_out: Free hole mass
        :param expected_roots_x_lj: A dictionary matching (l, j) to an
        ordered list of expected x_lnj, where j is 1 (electron) or 2 (hole)
        :param num_n: Number of modes to solve for and include in calculations.
        :param num_l: Number of angular momentum states to solve for an
        include in calcuations
        """
        # Model constants
        self.num_n = n_max + 1
        self.num_l = l_max + 1
        self.ms = ElectronHoleModelSpace(nmax=self.num_n-1, lmax=self.num_l-1,
                                         wavefunctions=self)

        # Band-dependent physical constants (1: conduction, 2: valence)
        self._Veff = [V_c, V_v]
        self._mu_in = [me_eff_in, mh_eff_in]
        self._mu_out = [me_eff_out, mh_eff_out]

        # Derived/inherited physical constants
        self.r_0 = r_0

        # Root finding
        self._x = dict()  # l, n, band -> x
        self._y = dict()  # l, n, band -> y
        self._expected_roots_x = expected_roots_x_lj  # l, band -> x0
        self._fill_roots_xy()

        self._psi_rad_norm_dict = dict()  # l, n, band -> psi_rad_norm

    # -- States --
    def iter_states(self):
        """Returns an iterator for the set of basis states for which
        the boundary value problem has solutions
        """
        for state in self.ms.modes():
            if self.x(state) is not None:
                yield state

    def iter_x_n(self, l, band):
        for state in self.iter_states():
            band0, l0, m0, n0 = self.ms.get_nums(state)
            if band0 == band and l0 == l and m0 == 0:
                yield self.x(state), n0

    # -- Wavefunction --
    def _Psi_rad_unnormalized_lnj(self, state):
        A, B = self._A_B(state)
        x = self.x(state)
        y = self.y(state)
        band, l, m, n = self.ms.get_nums(state)

        def psifn(r, d_r=0):
            if r <= self.r_0:
                return A * j_sph(l, d=d_r)(x*r/self.r_0) * (x/self.r_0)**d_r
            else:
                return B * k_sph(l, d=d_r)(y*r/self.r_0) * (y/self.r_0)**d_r
        return psifn

    def _Psi_rad_norm_lnj(self, state):
        """Return the L2 norm of the radial part of psi
        """
        def intfn(r):
            psi = self._Psi_rad_unnormalized_lnj(state)(r)
            return abs(psi)**2 * r**2
        return np.sqrt(integ.quad(intfn, 0, np.inf)[0])

    def wavefunction_env_radial(self, state):
        def psifn(r, d_r=0):
            psi = self._Psi_rad_unnormalized_lnj(state)
            norm = self._Psi_rad_norm_lnj(state)
            return psi(r, d_r=d_r) / norm
        return psifn

    def wavefunction_env_angular(self, state):
        band, l, n, m = self.ms.get_nums(state)
        return Y_lm(l=l, m=m)

    def wavefunction_env(self, state):
        def psifn(r, theta, phi):
            return (self.wavefunction_env_radial(state)(r) *
                    self.wavefunction_env_angular(state)(theta, phi))
        return psifn

    def wavefunction_bloch(self, band):
        """Bloch function. I have not found the definition for this in
        Riera
        """
        def ujfn(r, theta, phi):
            return 1  # TODO
        return ujfn

    # -- Getters --
    def V_eff(self, band):
        return self._Veff[band]

    def mu_in(self, band):
        return self._mu_in[band]

    def mu_out(self, band):
        return self._mu_out[band]

    def _A_B(self, state):
        """I have redefined these to be unnormalized coefficients
        """
        band, l, m, n = self.ms.get_nums(state)
        jln = j_sph(l=l)(self.x(state))
        kln = k_sph(l=l)(self.y(state))
        return kln, jln

    # -- Obtaining roots --
    def x(self, state):
        band, l, m, n = self.ms.get_nums(state)
        nums = l, n, band
        if nums in self._x:
            return self._x[nums]
        else:
            return None  # TODO

    def y(self, state):
        band, l, m, n = self.ms.get_nums(state)
        nums = l, n, band
        if nums in self._y:
            return self._y[nums]
        else:
            return None  # TODO

    def _get_y(self, x, band):
        x = complex(x)
        return np.sqrt(
            2 * self.mu_out(band) * self.r_0**2 * self.V_eff(band) -
            self.mu_out(band) / self.mu_in(band) * x**2
        )

    def plot_root_fn_x(self, l, j, xdat, show=True):
        fn = self._get_root_function(l=l, j=j)

        def fpr(x):
            x = complex(x)
            y = self._get_y(x=x, band=j)
            result = fn(np.array([x, y]))[0]
            return np.real(result)

        fig, ax = plt.subplots(1, 1)
        ax.axhline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(0, color='gray', lw=1, alpha=.5)

        for x, n in self.iter_x_n(l=l, band=j):
            ax.axvline(x.real, ls='--', color='green', lw=1, alpha=.5)

        ydat = np.array([fpr(x) for x in xdat])
        ydat /= linalg.norm(ydat, ord=2)
        liner, = ax.plot(xdat, ydat, '-', color='red')

        ylog = np.log(np.abs(ydat))
        ylog /= linalg.norm(ylog, ord=2)
        ax.plot(xdat, ylog, '--', color='red')

        ypr = np.real(np.sqrt([self._get_y(x=x, band=j) for x in xdat]))
        ypr /= linalg.norm(ypr, ord=2)
        lineb, = ax.plot(xdat, ypr, '-', color='blue')
        ax.plot(xdat, -ypr, '-', color='blue')
        # ax.set_ylim(bottom=0)

        ax.set_ylim(bottom=-2*max(ypr), top=2*max(ypr))
        ax.set_title('Electron/hole root function for l = {} '
                     'in the band {}'.format(l, j))
        ax.legend(
            (liner, lineb),
            ('root equation', 'roots relation')
        )

        if show:
            plt.show()
        return fig, ax

    def _fill_roots_xy(self):
        for j, l in it.product(range(2), range(self.num_l)):
            for root, n in self._solve_roots_xy(j=j, l=l):
                x, y = root
                assert not np.isnan(x)  # TODO
                assert not np.isnan(y)  # TODO
                self._x[l, n, j] = x
                self._y[l, n, j] = y

    def _get_expected_roots_xy(self, j, l, *args, **kwargs):
        start, roots_x = self._expected_roots_x[l, j]
        for x0, n in zip(roots_x, range(start, self.num_n)):
            x0 = complex(x0)
            y0 = self._get_y(x=x0, band=j)
            yield np.array([x0, y0], dtype=complex), n

    def _get_root_func1(self, j, l, *args, **kwargs):
        """Re-expressed this in terms of spherical Bessel functions
        """
        def rf1(x, y):
            jx = j_sph(l)(x)
            ky = k_sph(l)(y)
            djx = j_sph(l, d=1)(x)
            dky = k_sph(l, d=1)(y)
            ans = self.mu_out(j) * x * djx * ky - self.mu_in(j) * y * dky * jx
            return ans
        return rf1

    def _get_root_func2(self, j, *args, **kwargs):
        def rf2(x, y):
            mu0 = self.mu_out(j)
            mui = self.mu_in(j)
            vj = self.V_eff(j)
            r0 = self.r_0
            ans = mui * y**2 + mu0 * x**2 - 2 * mu0 * mui * r0**2 * vj
            return ans
        return rf2