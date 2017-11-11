import numpy as np
from PhononsModelSpace import PhononModelSpace
from ExitonModelSpace import ExitonModelSpace
from RamanQD import RamanQD
# import warnings
# warnings.filterwarnings('error')

RADIUS = 2e-7  # [cm]

# Phonon parameters
NMAX_PHONON = 2
LMAX_PHONON = 1
OMEGA_L = 305  # [cm-1]
OMEGA_T = 238  # [cm-1]
BETA_L = 5.04e-6  # [1]
BETA_T = 1.58e-6  # [1]
EPS_INF_IN = 5.72
EPS_INF_OUT = 4.64
XDAT_MU = np.linspace(1e-6, 10, 10000)
EXPECTED_ROOTS_PHONON = {
    0: (0, [1.65, 4.04, 4.50]),
    1: (0, [3.18, 4.80, 5.26]),
}

# Exiton parameters
NMAX_EXITON = 2
LMAX_EXITON = 1
V0_VAL = 1.5324e4  * 1e4  # [cm-1]
V0_COND = 2.0164e4 * 1e4  # [cm-1]
MASS_E_IN = .18  * 4.12e9
MASS_H_IN = .51  * 4.12e9
MASS_E_OUT = 1  * 4.12e9
MASS_H_OUT = 1  * 4.12e9
XDAT_X = np.linspace(1e-10, 20, 10000)
EXPECTED_ROOTS_ELECTRONS = {
    0: (0, [3.1, 6.1, 9.2]),
    1: (0, [4.4, 7.6, 10.8])
}
EXPECTED_ROOTS_HOLES = {
    0: (0, [3.1, 6.1, 9.2]),
    1: (0, [4.40, 7.66, 10.81])
}

# Raman parameters
E_GAP = 2.097e4 * 1e4
GAMMA_A = 8.06554  # [cm-1]
GAMMA_B = GAMMA_A
GAMMA_F = 3 * GAMMA_A

OMEGA_LASER = 1000  # [cm-1]
OMEGA_SEC = 100  # [cm-1]
POLAR_LASER = [1., 0., 0.]
POLAR_SEC = [1., 0., 0.]
REFRACTION_IDX_LASER = 1
REFRACTION_IDX_SEC = 1


# Roots


# Get phonon space
phonon_space = PhononModelSpace(
    nmax=NMAX_PHONON, lmax=LMAX_PHONON,
    radius=RADIUS, beta_L=BETA_L, beta_T=BETA_T,
    omega_L=OMEGA_L, omega_T=OMEGA_T,
    epsilon_inf_qdot=EPS_INF_IN, epsilon_inf_env=EPS_INF_OUT,
    expected_roots_mu_l=EXPECTED_ROOTS_PHONON,
)

# Plot roots for phonon potential
# phonon_space.plot_root_function_mu(l=0, xdat=XDAT_MU, show=True)
# phonon_space.plot_root_function_mu(l=1, xdat=XDAT_MU, show=True)

# Get exiton space
exiton_space = ExitonModelSpace(
    nmax=NMAX_EXITON,
    lmax=LMAX_EXITON,
    radius=RADIUS,
    V_v=V0_VAL,
    V_c=V0_COND,
    me_eff_in=MASS_E_IN,
    me_eff_out=MASS_E_OUT,
    mh_eff_in=MASS_H_IN,
    mh_eff_out=MASS_H_OUT,
    expected_roots_x_elec=EXPECTED_ROOTS_ELECTRONS,
    expected_roots_x_hole=EXPECTED_ROOTS_HOLES,
)

# Plots roots: Electrons
# exiton_space.plot_root_fn_electrons(l=0, xdat=XDAT_X, show=True)
# exiton_space.plot_root_fn_electrons(l=1, xdat=XDAT_X, show=True)

# Plot roots: Holes
# exiton_space.plot_root_fn_holes(l=0, xdat=XDAT_X, show=True)
# exiton_space.plot_root_fn_holes(l=1, xdat=XDAT_X, show=True)


# Make system Hamiltonian
ham = RamanQD(
    Gamma_a=GAMMA_A,
    Gamma_b=GAMMA_B,
    Gamma_f=GAMMA_F,
    phonon_space=phonon_space,
    exiton_space=exiton_space,
    E_gap=E_GAP,
)

print(
    ham.differential_raman_efficiency(
        omega_l=OMEGA_LASER, e_l=POLAR_LASER, omega_s=OMEGA_SEC, e_s=POLAR_SEC)
)
