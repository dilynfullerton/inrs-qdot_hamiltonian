import numpy as np
from scipy import optimize as opt


class RootSolverComplex2d:
    def _solve_roots_xy(self, *args, **kwargs):
        fun = self._get_root_function_cplx(*args, **kwargs)
        for x0y0 in self._get_expected_roots_xxyy(*args, **kwargs):
            result = opt.root(fun=fun, x0=x0y0)
            if not result.success:
                print('FAILED')  # TODO
            else:
                xr, xi, yr, yi = result.x
                yield np.array([xr + 1j * xi, yr + 1j * yi])

    def _get_expected_roots_xxyy(self, *args, **kwargs):
        for x0, y0 in self._get_expected_roots_xy(*args, **kwargs):
            x0 = complex(x0)
            y0 = complex(y0)
            yield np.array([x0.real, x0.imag, y0.real, y0.imag])

    def _get_expected_roots_xy(self, *args, **kwargs):
        raise NotImplementedError

    def _get_root_function_cplx(self, *args, **kwargs):
        def pi(zz):
            z1, z2 = zz
            return np.array([z1.real, z1.imag, z2.real, z2.imag])

        def ipi(xxyy):
            xr, xi, yr, yi = xxyy
            return np.array([xr+1j*xi, yr+1j*yi])

        psi = self._get_root_function(*args, **kwargs)

        def rf(xxyy):
            return pi(psi(ipi(xxyy)))

        return rf

    def _get_root_function(self, *args, **kwargs):
        fun_f = self._get_root_func1(*args, **kwargs)
        fun_g = self._get_root_func2(*args, **kwargs)

        def psi(zz):
            z1, z2, = zz
            f = complex(fun_f(z1, z2))
            g = complex(fun_g(z1, z2))
            return np.array([f, g])
        return psi

    def _get_root_func1(self, *args, **kwargs):
        raise NotImplementedError

    def _get_root_func2(self, *args, **kwargs):
        raise NotImplementedError


class RootSolverReal2d:
    def _solve_roots_xy(self, *args, **kwargs):
        fun = self._get_root_function(*args, **kwargs)
        for x0y0 in self._get_expected_roots_xxyy(*args, **kwargs):
            result = opt.root(fun=fun, x0=x0y0)
            if not result.success:
                print('FAILED')  # TODO
            else:
                x, y = result.x
                yield np.array([x, y])

    def _get_expected_roots_xxyy(self, *args, **kwargs):
        for x0, y0 in self._get_expected_roots_xy(*args, **kwargs):
            yield np.array([x0, y0])

    def _get_expected_roots_xy(self, *args, **kwargs):
        raise NotImplementedError

    def _get_root_function(self, *args, **kwargs):
        fun_f = self._get_root_func1(*args, **kwargs)
        fun_g = self._get_root_func2(*args, **kwargs)

        def psi(xy):
            x, y, = xy
            f = fun_f(x, y)
            g = fun_g(x, y)
            return np.array([f, g])
        return psi

    def _get_root_func1(self, *args, **kwargs):
        raise NotImplementedError

    def _get_root_func2(self, *args, **kwargs):
        raise NotImplementedError
