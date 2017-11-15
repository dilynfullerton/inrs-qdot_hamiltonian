import itertools as it
import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from scipy import integrate as integ
from scipy import linalg
from helper_functions import Y_lm, j_sph, k_sph, J, K
from root_solver_2d import RootSolverComplex2d
from ModelSpace import ModelSpace
from collections import namedtuple


ExitonMode = namedtuple('ExitonMode', ['band', 'l', 'm', 'n'])


class ExitonModelSpace(ModelSpace, RootSolverComplex2d):
    BAND_VAL = 0
    BAND_COND = 1

    def __init__(
            self, nmax, lmax,
            radius, V_v, V_c, me_eff_in, mh_eff_in, me_eff_out, mh_eff_out,
            free_electron_mass, expected_roots_x_elec, expected_roots_x_hole,
            E_gap,
    ):
        """
        :param V_v: Effective potential (?) of the valence band
        :param V_c: Effective potential (?) of the conduction band
        :param me_eff_in: Effective electron mass in nanostructure,
        relative to the free electron mass
        :param mh_eff_in: Effective hole mass in nanostructure,
        relative to the free electron mass
        :param me_eff_out: Effective electron mass outside of the
        nanostructure, relative to the free electron mass
        :param mh_eff_out: Effective hole mass outside of the
        nanostructure, relative to the free electron mass
        ordered list of expected x_lnj, where j is 1 (electron) or 2 (hole)
        include in calcuations
        :param free_electron_mass: Mass of a free electron
        """
        # Model space constants
        self.nmax = nmax
        self.lmax = lmax
        self.num_n = self.nmax + 1
        self.num_l = self.lmax + 1

        # Band-dependent physical constants (1: conduction, 2: valence)
        self._Veff_c = V_c
        self._Veff_v = V_v
        self._me_eff_in = me_eff_in
        self._me_eff_out = me_eff_out
        self._mh_eff_in = mh_eff_in
        self._mh_eff_out = mh_eff_out
        self.free_electron_mass = free_electron_mass

        # Derived/inherited physical constants
        self.r_0 = radius
        self.meff_reduced = self.free_electron_mass / (
            1/self.meff_in(band=ExitonModelSpace.BAND_COND) +
            1/self.meff_in(band=ExitonModelSpace.BAND_VAL)
        )
        self.E_0 = 1/2/self.meff_reduced/self.r_0**2
        self.E_gap = E_gap

        # Root finding
        self._roots_x = dict()  # l, n, band -> x
        self._roots_y = dict()  # l, n, band -> y
        self._expected_roots_x_elec = expected_roots_x_elec
        self._expected_roots_x_hole = expected_roots_x_hole
        self._fill_roots_xy()

        self._psi_rad_norm_dict = dict()  # l, n, band -> psi_rad_norm

    def nfock(self, mode):
        return 2

    def modes(self):
        for band in [ExitonModelSpace.BAND_COND, ExitonModelSpace.BAND_VAL]:
            for l in range(self.num_l):
                for m in range(-l, l+1):
                    for n in range(self.num_n):
                        yield ExitonMode(band=band, l=l, m=m, n=n)

    def states(self):
        """Returns an iterator for the set of basis states for which
        the boundary value problem has solutions
        """
        for state in self.modes():
            if self.x(state) is not None:
                yield state

    def electron_states(self):
        for s in self.states():
            if s.band == ExitonModelSpace.BAND_COND:
                yield s

    def hole_states(self):
        for s in self.states():
            if s.band == ExitonModelSpace.BAND_VAL:
                yield s

    def electron_hole_states(self):
        for e_state in self.electron_states():
            for h_state in self.hole_states():
                yield e_state, h_state

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

    def get_nums(self, mode):
        return mode.band, mode.l, mode.m, mode.n

    def get_ket(self, mode):
        return self.create(mode=mode) * self.vacuum_ket()

    def get_omega(self, state):
        """Returns the frequency associated with the first exitation of the
        given single-particle mode
        """
        # TODO: verify
        e = self.E_0 * self.get_omega_rel(state)
        return e

    def get_omega_ehp(self, ehp):
        elec, hole = ehp
        return self.get_omega(elec) + self.get_omega(hole) + self.E_gap

    def get_omega_rel(self, state):
        return self.beta(state.band) * self.x(state)**2

    # -- States --
    def iter_x_n(self, l, band):
        for state in self.states():
            if state.band == band and state.l == l and state.m == 0:
                yield self.x(state), state.n

    # -- Wavefunction --
    def _Psi_rad_unnormalized_lnj(self, state):
        x = self.x(state)
        y = self.y(state)
        l = state.l
        r0 = self.r_0

        def psifn(r, d_r=0):
            if r < r0:
                return k_sph(l)(y) * j_sph(l, d=d_r)(x*r/r0) * (x/r0)**d_r
            else:
                return j_sph(l)(x) * k_sph(l, d=d_r)(y*r/r0) * (y/r0)**d_r
        return psifn

    def _Psi_rad_norm_key(self, state):
        return state.l, state.n, state.band

    def _Psi_rad_norm_lnj(self, state):
        """Return the L2 norm of the radial part of psi
        """
        k = self._Psi_rad_norm_key(state)
        if k in self._psi_rad_norm_dict:
            return self._psi_rad_norm_dict[k]

        def intfn(r):
            psi = self._Psi_rad_unnormalized_lnj(state)(r)
            return abs(psi)**2 * r**2
        self._psi_rad_norm_dict[k] = np.sqrt(integ.quad(intfn, 0, np.inf)[0])
        return self._Psi_rad_norm_lnj(state)

    def wavefunction_envelope_radial(self, state):
        def psifn(r, d_r=0):
            psi = self._Psi_rad_unnormalized_lnj(state)
            norm = self._Psi_rad_norm_lnj(state)
            return psi(r, d_r=d_r) / norm
        return psifn

    def wavefunction_envelope_angular(self, state):
        return Y_lm(l=state.l, m=state.m)

    def wavefunction_envelope(self, state):
        def psifn(r, theta, phi):
            return (self.wavefunction_envelope_radial(state)(r) *
                    self.wavefunction_envelope_angular(state)(theta, phi))
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
        if band == ExitonModelSpace.BAND_COND:
            return self._Veff_c
        else:
            return self._Veff_v

    def meff_in(self, band):
        if band == ExitonModelSpace.BAND_COND:
            return self._me_eff_in
        else:
            return self._mh_eff_in

    def meff_out(self, band):
        if band == ExitonModelSpace.BAND_COND:
            return self._me_eff_out
        else:
            return self._mh_eff_out

    def beta(self, j):
        """See Riera (92)
        """
        return self.meff_reduced / self.meff_in(j) / self.free_electron_mass

    # def _A_B(self, state):
    #     """I have redefined these to be unnormalized coefficients
    #     """
    #     jln = J(l=state.l+1/2)(self.x(state))
    #     kln = K(l=state.l+1/2)(self.y(state))
    #     return kln, jln

    # -- Obtaining roots --
    def _root_dict_key(self, state):
        return state.l, state.n, state.band

    def x(self, state):
        k = self._root_dict_key(state)
        if k in self._roots_x:
            return self._roots_x[k]
        else:
            return None  # TODO

    def y(self, state):
        k = self._root_dict_key(state)
        if k in self._roots_y:
            return self._roots_y[k]
        else:
            return None  # TODO

    def _get_y_from_x(self, x, band):
        x = complex(x)
        return np.sqrt(
            2 * self.meff_out(band) *
            self.free_electron_mass * self.r_0**2 * self.V_eff(band) -
            (self.meff_out(band) / self.meff_in(band)) * x**2
        )

    def _high_x(self, band):
        return self.r_0 * np.sqrt(
            2 * self.meff_in(band) * self.free_electron_mass * self.V_eff(band)
        )

    def plot_root_fn_electrons(self, l, xdat, show=True):
        return self.plot_root_fn_x(l=l, xdat=xdat, show=show,
                                   j=ExitonModelSpace.BAND_COND)

    def plot_root_fn_holes(self, l, xdat, show=True):
        return self.plot_root_fn_x(l=l, xdat=xdat, show=show,
                                   j=ExitonModelSpace.BAND_VAL)

    def plot_root_fn_x(self, l, j, xdat, show=True):
        fn = self._get_root_function(l=l, j=j)

        def fpr(x):
            x = complex(x)
            y = self._get_y_from_x(x=x, band=j)
            return fn(np.array([x, y]))[0]

        fig, ax = plt.subplots(1, 1)

        # Plot axes
        ax.axhline(0, color='gray', lw=1, alpha=.5)
        ax.axvline(0, color='gray', lw=1, alpha=.5)

        # Plot high x boundaries
        high_x = self._high_x(j)
        if min(xdat) <= high_x <= max(xdat):
            ax.axvline(high_x, ls='-', color='black', lw=1, alpha=.5)
            ax.axvline(-high_x, ls='-', color='black', lw=1, alpha=.5)

        xdat = np.concatenate((-xdat, xdat))
        xdat.sort()

        # Plot real and imaginary parts of the root function
        ydat = np.array([fpr(x) for x in xdat])
        ydat /= linalg.norm(ydat, ord=2)
        linei, = ax.plot(xdat, np.imag(ydat), '-', color='blue')
        liner, = ax.plot(xdat, np.real(ydat), '-', color='red')
        ax.plot(xdat, np.abs(ydat), '-', color='purple')
        # Plot log function to aid in identifying roots
        ylog = np.log(np.abs(ydat))
        ylog /= linalg.norm(ylog, ord=2)
        ax.plot(xdat, ylog, '--', color='brown')

        # Regions of imaginary y
        x_min_imy = max(min(xdat), high_x)
        x_max_imy = max(xdat)
        x_min_imy = min(x_min_imy, x_max_imy)
        x_max_imy = max(x_min_imy, x_max_imy)
        span_imy = ax.axvspan(x_min_imy, x_max_imy, color='blue', alpha=.3)
        ax.axvspan(-x_max_imy, -x_min_imy, color='blue', alpha=.3)

        # Region of real y
        x_min_rey = max(min(xdat), -high_x)
        x_max_rey = min(max(xdat), high_x)
        span_rey = ax.axvspan(x_min_rey, x_max_rey, color='red', alpha=.3)

        # Roots
        for x, n in self.iter_x_n(l=l, band=j):
            if min(xdat) <= x <= max(xdat):
                ax.axvline(x.real, ls='--', color='green', lw=1, alpha=.5)
            else:
                print('Zero out of range: x={}'.format(x))

        ypr = np.real(np.sqrt([self._get_y_from_x(x=x, band=j) for x in xdat]))
        ypr /= linalg.norm(ypr, ord=2)
        ax.plot(xdat, ypr, '-.', color='cyan')
        ax.plot(xdat, -ypr, '-.', color='cyan')
        # ax.set_ylim(bottom=0)
        ax.set_ylim(bottom=-2*max(ypr), top=2*max(ypr))

        # Title and legend
        ax.set_title('Electron/hole root function for l = {} '
                     'in the band {}'.format(l, j))
        ax.legend(
            (liner, linei, span_rey, span_imy),
            ('root equation (real)', 'root equation (imag)',
             'real y', 'imag y')
        )

        if show:
            plt.show()
        return fig, ax

    def _fill_roots_xy(self):
        for j, l in it.product(
                [ExitonModelSpace.BAND_VAL, ExitonModelSpace.BAND_COND],
                range(self.num_l)
        ):
            for root, n in self._solve_roots_xy(j=j, l=l):
                x, y = root
                assert not np.isnan(x)  # TODO
                assert not np.isnan(y)  # TODO
                self._roots_x[l, n, j] = x
                self._roots_y[l, n, j] = y

    def _get_expected_roots_xy(self, j, l, *args, **kwargs):
        if j == ExitonModelSpace.BAND_VAL:
            start, roots_x = self._expected_roots_x_hole[l]
        else:
            start, roots_x = self._expected_roots_x_elec[l]
        for x0, n in zip(roots_x, range(start, self.num_n)):
            x0 = complex(x0)
            y0 = self._get_y_from_x(x=x0, band=j)
            yield np.array([x0, y0], dtype=complex), n

    def _get_root_func1(self, j, l, *args, **kwargs):
        """Re-expressed this in terms of spherical Bessel functions
        """
        def rf1(x, y):
            jx = j_sph(l)(x)
            ky = k_sph(l)(y)
            djx = j_sph(l, d=1)(x)
            dky = k_sph(l, d=1)(y)
            ans = x*djx*ky - (self.meff_in(j)/self.meff_out(j))*y*dky*jx
            return ans
        return rf1

    def _get_root_func2(self, j, *args, **kwargs):
        def rf2(x, y):
            mu0 = self.meff_out(j)
            mui = self.meff_in(j)
            vj = self.V_eff(j)
            r0 = self.r_0
            m0 = self.free_electron_mass
            ans = y**2*(mui/mu0) + x**2 - 2 * mui * m0 * r0**2 * vj
            return ans
        return rf2
