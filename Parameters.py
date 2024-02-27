from dataclasses import dataclass


@dataclass
class Parameters:
    """All model parameters. Default values are those of DICE 2016"""

    # model setup
    timestep: int = 5
    start_year: int = 2010
    n_timesteps: int = 100

    # optimization settings
    max_iterations: int = 500
    tolerance: float = 1e-4

    # logistic growth equation parameters
    L_initial: float = 7.403  # Initial world population (billions)
    g_L_initial: float = 0.134  # per period
    L_asymptotic: float = 11.5  # Asymptotic population (billions)

    A_initial: float = 5.115  # Initial level of total factor productivity
    g_A_initial: float = 0.076  # Initial growth rate of TFP per 5 years
    delta_A: float = 0.005  # Decline rate of TFP per 5 years

    Y_initial: float = 105.5  # Initial world gross output (trill 2010 USD)
    E_initial: float = 35.85  # Industrial emissions 2010 (GtCO2 per year)

    mu_initial: float = 0.03  # Initial emissions control rate for base case

    g_sigma_initial: float = -0.0152  # Initial growth of sigma (per year)

    delta_sigma: float = -0.001  # Decline rate of decarbonization (per period)

    delta_K: float = 0.100  # Deprecetaion rate on capital (per year)
    K_initial: float = 223.0  # Initial capital value (trillion 2010 USD)

    alpha: float = 1.45  # Elasticity of marginal utility of consumption
    rho: float = 0.015  # Initial rate of social time preference per year

    gamma: float = 0.300  # Capital elasticity in production function

    M_AT_initial: float = 851.0  # Initial concentration in atmosphere 2010 (GtC)
    M_UP_initial: float = 460.0  # Initial Concentration in upper strata 2010 (GtC)
    M_LO_initial: float = 1740.0  # Initial Concentration in lower strata 2010 (GtC)

    psi_1: float = 0.0
    psi_2: float = 0.00267  # Damage quadratic term /0.00267 /
    psi_3: float = 0.0  # Damage term, 0.1644 for Weitzman damage function

    phi_12: float = 0.12
    phi_23: float = 0.007
    M_AT_eq: float = 588.0  # Equilibrium concentration atmosphere (GtC)
    M_UP_eq: float = 360.0  # Equilibrium concentration in upper strata (GtC)
    M_LO_eq: float = 1720.0  # Equilibrium concentration in lower strata (GtC)

    eta: float = 3.6813  # fco22x Forcings of equilibrium CO2 doubling (Wm-2) /3.8 /
    F_EX_initial: float = 0.5  # 2010 forcings of non-CO2 GHG (Wm-2)
    F_EX_2100: float = 1.0  # 2100 forcings of non-CO2 GHG (Wm-2)

    xi_1: float = 0.1005
    xi_3: float = 0.088
    xi_4: float = 0.025
    t2xco2: float = 3.1  # Equilibrium temp impact (oC per doubling CO2) / 2.9 /

    T_AT_initial: float = 0.85  # Initial atmospheric temp change (C from 1900)
    T_LO_initial: float = 0.0068  # Initial lower stratum temp change (C from 1900)

    E_Land_initial: float = 2.6  # Carbon emissions from land 2010 (GtCO2 per period)
    delta_E_Land: float = 0.115  # Decline rate of land emissions (per period)

    theta_2: float = 2.8  # expcost2 Exponent of control cost function
    B_initial: float = 550  # Cost of backstop 2010$ per tCO2 2010 (pback).
    # I.e. marginal cost of emissions at 100% abatement

    delta_B: float = 0.025  # Initial cost decline backstop cost per period (gback)

    # limits
    mu_upper_limit_late: float = (
        1.2  # limmiu Upper limit on control rate after 2150 / 1.2 /
    )
    mu_upper_limit_early: float = 1.0
    mu_upper_limit_change_year: int = 2150
    mu_lower_limit: float = 0.01

    S_lower_limit: float = 0.1
    S_upper_limit: float = 0.9
    n_periods_at_opt_long_run_S: int = 10

    # used for TCRE climate module only
    TCRE = 1.6  # Transient climate response to cumulative carbon emissions (degrees C per Tt carbon)

    # settings
    climate_module: str = "original"  # or "TCRE"

    def update(self, parameter_updates: dict):
        """Update selected parameters from a dictionary
        
        Input:
            parameter_updates: a dictionary of paramaters of the form {"parameter name":value}"""

        for key, value in parameter_updates.items():
            if hasattr(self, key):
                setattr(self, key, value)


updated_parameters = {
    "start_year": 2025,
    "L_initial": 7.888,
    "g_A_initial": 0.079,
    "delta_A": 0.006,
    "E_initial": 40,
    "K_initial": 400,
    "M_AT_initial": 879,
    "M_UP_eq": 460,
    "M_LO_eq": 1740,
    "eta": 3.8,
    "T_AT_initial": 1.3,
    "E_Land_initial": 3.9,
    "delta_E_land": 0.2,
    "M_AT_eq": 596.4,
    "alpha": 1.15,
    "A_initial": 3.8,
    "g_sigma_initial": -0.01,
    "t2xco2": 2.9,
    "F_EX_initial": 0.22,
    "F_EX_2100": 0.22,  # "F_EX_2100": 0.7,
    "mu_initial": 0.039,
    "mu_upper_limit_change_year": 2060,
}
