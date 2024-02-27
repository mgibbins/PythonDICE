import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from Parameters import Parameters, updated_parameters
from PlotResults import plot_results
import time
import logging


# set logger to display info messages
logging.getLogger().setLevel(logging.INFO)


@dataclass
class ControlVariables:
    """Control variables in the optimization"""
    S: np.array
    mu: np.array


class Setup:
    """Initial setup: results calculated once at the start directly from parameters"""

    def __init__(self, parameters):
        
        # check settings are allowable
        assert parameters.climate_module in [
            "original",
            "TCRE",
        ], "climate_module must be 'original' or 'TCRE'"

        self.T = self.calc_T(
            parameters.start_year, parameters.timestep, parameters.n_timesteps
        )

        self.sigma_initial = self.calc_sigma_initial(
            parameters.E_initial, parameters.Y_initial, parameters.mu_initial
        )

        self.sigma = self.calc_sigma(
            self.T,
            parameters.timestep,
            self.sigma_initial,
            parameters.g_sigma_initial,
            parameters.delta_sigma,
        )

        self.L = self.calc_L(
            self.T,
            parameters.L_initial,
            parameters.g_L_initial,
            parameters.L_asymptotic,
        )

        self.A = self.calc_A(
            self.T,
            parameters.A_initial,
            parameters.g_A_initial,
            parameters.delta_A,
            parameters.timestep,
        )

        self.E_Land = self.calc_E_Land(
            self.T, parameters.E_Land_initial, parameters.delta_E_Land
        )

        self.R = self.calc_R(self.T, parameters.rho, parameters.timestep)

        self.B = self.calc_B(self.T, parameters.B_initial, parameters.delta_B)
        self.theta_1 = self.calc_theta_1(self.B, self.sigma, parameters.theta_2)

        self.optimal_long_run_S = self.calc_optimal_long_run_S(
            parameters.delta_K, parameters.alpha, parameters.gamma, parameters.rho
        )

        # constraints
        self.mu_limits = self.calc_mu_limits(
            self.T,
            parameters.mu_upper_limit_early,
            parameters.mu_upper_limit_late,
            parameters.mu_upper_limit_change_year,
            parameters.mu_lower_limit,
        )

        self.S_limits = self.calc_S_limits(
            self.T,
            parameters.S_lower_limit,
            parameters.S_upper_limit,
            parameters.n_periods_at_opt_long_run_S,
            self.optimal_long_run_S,
        )

        # initialize control variables
        self.initial_control_variables = self.calc_initial_control_variables(
            self.T, self.mu_limits, parameters.S_lower_limit, parameters.S_upper_limit
        )

        if parameters.climate_module == "original":
            (
                self.phi_11,
                self.phi_21,
                self.phi_22,
                self.phi_32,
                self.phi_33,
            ) = self.calc_phis(
                parameters.phi_12,
                parameters.phi_23,
                parameters.M_AT_eq,
                parameters.M_UP_eq,
                parameters.M_LO_eq,
            )

            self.F_EX = self.calc_F_EX(
                self.T,
                parameters.F_EX_initial,
                parameters.F_EX_2100,
            )
            self.xi_2 = self.calc_xi_2(parameters.eta, parameters.t2xco2)

    @staticmethod
    def calc_sigma_initial(E_initial, Y_initial, mu_initial):
        return E_initial / (Y_initial * (1 - mu_initial))

    @staticmethod
    def calc_sigma(T, timestep, sigma_initial, g_sigma_initial, delta_sigma):
        g_sigma = g_sigma_initial * (1 + delta_sigma) ** (
            np.arange(0, len(T)) * timestep
        )
        sigma = np.ones(len(T)) * sigma_initial
        for t in np.arange(1, len(T)):
            sigma[t] = sigma[t - 1] * np.exp(g_sigma[t - 1] * timestep)
        return sigma

    @staticmethod
    def calc_T(start_year, timestep, n_timesteps):
        end_year = start_year + timestep * (n_timesteps)
        return np.arange(start_year, end_year, timestep)

    @staticmethod
    def calc_R(T, rho, timestep):
        """equation 3"""
        return (1 + rho) ** (-timestep * np.arange(0, len(T)))

    @staticmethod
    def calc_A(T, A_initial, g_A_initial, delta_A, timestep):
        """p.9"""
        g_A = g_A_initial * np.exp(-delta_A * timestep * np.arange(0, len(T)))
        A = np.ones(len(T)) * A_initial
        for t in np.arange(1, len(T)):
            A[t] = A[t - 1] / (1 - g_A[t - 1])
        return A

    @staticmethod
    def calc_B(T, B_initial, delta_B):
        return B_initial * ((1 - delta_B) ** np.arange(0, len(T)))

    @staticmethod
    def calc_theta_1(B, sigma, theta_2):
        return B * (sigma / 1000) / theta_2

    @staticmethod
    def calc_L(T, L_initial, g_L_initial, L_asymptotic):
        L = np.zeros(len(T))
        L[0] = L_initial
        for t in np.arange(1, len(T)):
            L[t] = L[t - 1] * (L_asymptotic / L[t - 1]) ** g_L_initial
        return L

    @staticmethod
    def calc_E_Land(T, E_Land_initial, delta_E_Land):
        return E_Land_initial * ((1 - delta_E_Land) ** np.arange(0, len(T)))

    @staticmethod
    def calc_optimal_long_run_S(delta_K, alpha, gamma, rho):
        """p.98 optlrsav = (dk + .004)/(dk + .004*elasmu + prstp)*gama;"""
        return (delta_K + 0.004) / (delta_K + 0.004 * alpha + rho) * gamma

    @staticmethod
    # constraints
    def calc_mu_limits(
        T,
        mu_upper_limit_early,
        mu_upper_limit_late,
        mu_upper_limit_change_year,
        mu_lower_limit,
    ):
        mu_lower_limit_series = np.full(len(T), mu_lower_limit)
        mu_upper_limit_series = np.full(len(T), mu_upper_limit_early)
        mu_upper_limit_series[T > mu_upper_limit_change_year] = mu_upper_limit_late

        mu_limits = []
        for i in range(len(T)):
            mu_limits.append((mu_lower_limit_series[i], mu_upper_limit_series[i]))
        return mu_limits

    @staticmethod
    def calc_S_limits(
        T, S_lower_limit, S_upper_limit, n_periods_at_opt_long_run_S, optimal_long_run_S
    ):
        S_lower_limit_series = np.full(len(T), S_lower_limit)
        S_upper_limit_series = np.full(len(T), S_upper_limit)

        S_upper_limit_series[-n_periods_at_opt_long_run_S:] = optimal_long_run_S
        S_lower_limit_series[-n_periods_at_opt_long_run_S:] = (
            optimal_long_run_S * 0.99999
        )

        S_limits = []
        for i in range(len(T)):
            S_limits.append((S_lower_limit_series[i], S_upper_limit_series[i]))

        return S_limits

    @staticmethod
    def calc_initial_control_variables(T, mu_limits, S_lower_limit, S_upper_limit):
        # initial mu as halfway between bounds
        mu0 = [(x[0] + x[1]) / 2 for x in mu_limits]

        # initial S as 0.2 (or upper/lower limit if outside of bounds)
        S0 = np.ones(len(T)) * min(max(0.2, S_lower_limit), S_upper_limit)
        return ControlVariables(S=S0, mu=mu0)

    @staticmethod
    def calc_phis(phi_12, phi_23, M_AT_eq, M_UP_eq, M_LO_eq):
        """only used in original climate module"""
        phi_11 = 1 - phi_12
        phi_21 = phi_12 * M_AT_eq / M_UP_eq
        phi_22 = 1 - phi_21 - phi_23
        phi_32 = phi_23 * M_UP_eq / M_LO_eq
        phi_33 = 1 - phi_32

        return phi_11, phi_21, phi_22, phi_32, phi_33

    @staticmethod
    def calc_F_EX(T, F_EX_initial, F_EX_2100):
        """only used in original climate module
        p.98 Calculate exogenous radiative forcing, i.e. that due to other GHGs not otherwise modelled.

        Modelled as linear increase from initial value to max value up to 2100, then maximum for the
        remaining time."""

        # set all to F_EX_2100
        F_EX = np.full(len(T), F_EX_2100)

        # before 2100, linear increase from F_EX_initial to F_EX_2100
        F_EX[T <= 2100] = np.linspace(F_EX_initial, F_EX_2100, (T <= 2100).sum())

        return F_EX

    @staticmethod
    def calc_xi_2(eta, t2xco2):
        """only used in original climate module"""
        return eta / t2xco2


class CoreCalculation:
    """Calculates a full model run for a given set of parameters and control variables"""

    def __init__(
        self, parameters: Parameters, setup: Setup, control_variables: ControlVariables
    ):
        # calculate first timestep values
        E_Ind_initial = self.calc_E_Ind(
            setup.sigma[0],
            control_variables.mu[0],
            setup.A[0],
            parameters.K_initial,
            parameters.gamma,
            setup.L[0],
        )
        E_initial = self.calc_E(E_Ind_initial, setup.E_Land[0])

        Lambda_initial = self.calc_Lambda(
            setup.theta_1[0], control_variables.mu[0], parameters.theta_2
        )

        Omega_initial = self.calc_Omega(
            parameters.psi_1,
            parameters.psi_2,
            parameters.psi_3,
            parameters.T_AT_initial,
        )

        Q_initial = self.calc_Q(
            Lambda_initial,
            Omega_initial,
            parameters.K_initial,
            parameters.gamma,
            setup.L[0],
            setup.A[0],
        )

        I_initial = self.calc_I(control_variables.S[0], Q_initial)

        # initialize variables
        self.K = self.initialize(parameters.K_initial, setup.T)
        self.Lambda = self.initialize(Lambda_initial, setup.T)
        self.Omega = self.initialize(Omega_initial, setup.T)
        self.E_Ind = self.initialize(E_Ind_initial, setup.T)
        self.E = self.initialize(E_initial, setup.T)
        self.Q = self.initialize(Q_initial, setup.T)
        self.I = self.initialize(I_initial, setup.T)
        self.T_AT = self.initialize(parameters.T_AT_initial, setup.T)

        # initialize additional variables only in the original climate module
        if parameters.climate_module == "original":
            F_initial = self.calc_F(
                parameters.M_AT_initial,
                setup.F_EX[0],
                parameters.eta,
                parameters.M_AT_eq,
            )
            self.T_LO = self.initialize(parameters.T_LO_initial, setup.T)
            self.M_AT = self.initialize(parameters.M_AT_initial, setup.T)
            self.M_UP = self.initialize(parameters.M_UP_initial, setup.T)
            self.M_LO = self.initialize(parameters.M_LO_initial, setup.T)
            self.F = self.initialize(F_initial, setup.T)

        # calculate iteratively for remaining timesteps
        for t in np.arange(1, len(setup.T)):
            self.K[t] = self.calc_K(
                self.I[t - 1], parameters.delta_K, self.K[t - 1], parameters.timestep
            )
            self.Lambda[t] = self.calc_Lambda(
                setup.theta_1[t], control_variables.mu[t], parameters.theta_2
            )
            self.E_Ind[t] = self.calc_E_Ind(
                setup.sigma[t],
                control_variables.mu[t],
                setup.A[t],
                self.K[t],
                parameters.gamma,
                setup.L[t],
            )
            self.E[t] = self.calc_E(self.E_Ind[t], setup.E_Land[t])

            if parameters.climate_module == "original":
                self.M_AT[t], self.M_UP[t], self.M_LO[t] = self.calc_Ms(
                    parameters.timestep,
                    self.E[t - 1],
                    self.M_AT[t - 1],
                    self.M_UP[t - 1],
                    self.M_LO[t - 1],
                    setup.phi_11,
                    setup.phi_21,
                    parameters.phi_12,
                    setup.phi_22,
                    setup.phi_32,
                    parameters.phi_23,
                    setup.phi_33,
                )
                self.F[t] = self.calc_F(
                    self.M_AT[t], setup.F_EX[t], parameters.eta, parameters.M_AT_eq
                )
                self.T_AT[t], self.T_LO[t] = self.calc_Ts(
                    self.F[t],
                    self.T_AT[t - 1],
                    self.T_LO[t - 1],
                    parameters.xi_1,
                    setup.xi_2,
                    parameters.xi_3,
                    parameters.xi_4,
                )
            if parameters.climate_module == "TCRE":
                self.T_AT[t] = self.calc_T_AT_TCRE(
                    self.T_AT[t - 1],
                    parameters.TCRE,
                    parameters.timestep,
                    self.E[t - 1],
                )
            self.Omega[t] = self.calc_Omega(
                parameters.psi_1, parameters.psi_2, parameters.psi_3, self.T_AT[t]
            )
            self.Q[t] = self.calc_Q(
                self.Lambda[t],
                self.Omega[t],
                self.K[t],
                parameters.gamma,
                setup.L[t],
                setup.A[t],
            )
            self.I[t] = self.calc_I(control_variables.S[t], self.Q[t])

    @staticmethod
    def initialize(initial_value, T):
        series = np.zeros(len(T))
        series[0] = initial_value
        return series

    @staticmethod
    def calc_Lambda(theta_1, mu: float, theta_2):
        """equation 6"""
        return theta_1 * mu**theta_2

    @staticmethod
    def calc_Q(Lambda, Omega, K, gamma, L, A):
        """equation 4"""
        return ((1 - Lambda) * A * K**gamma * L ** (1 - gamma)) / (1 + Omega)

    @staticmethod
    def calc_K(I_prev, delta_K, K_prev, timestep):
        """equation 9.
        Except change to I_prev instead of I,adjust for 5 year timestep,
        and change from -delta_K to +(1-delta_K)"""
        # (1 - dk) ** tstep * iK[index - 1] + tstep * iI[index - 1]
        return timestep * I_prev + ((1 - delta_K) ** (timestep)) * K_prev

    @staticmethod
    def calc_E_Ind(sigma, mu, A, K, gamma, L):
        """equation 10"""
        return sigma * (1 - mu) * A * K**gamma * (L) ** (1 - gamma)

    @staticmethod
    def calc_E(E_Ind, E_Land):
        """equation 12"""
        return E_Ind + E_Land

    @staticmethod
    def calc_Ms(
        timestep,
        E_prev,
        M_AT_prev,
        M_UP_prev,
        M_LO_prev,
        phi_11,
        phi_21,
        phi_12,
        phi_22,
        phi_32,
        phi_23,
        phi_33,
    ):
        """equations 13, 14, 15"""

        M_AT = timestep * E_prev / 3.666 + phi_11 * M_AT_prev + phi_21 * M_UP_prev
        M_UP = phi_12 * M_AT_prev + phi_22 * M_UP_prev + phi_32 * M_LO_prev
        M_LO = phi_23 * M_UP_prev + phi_33 * M_LO_prev

        return M_AT, M_UP, M_LO

    @staticmethod
    def calc_F(M_AT, F_EX, eta, M_AT_eq):
        """equation 16"""
        return eta * np.log2(M_AT / M_AT_eq) + F_EX

    @staticmethod
    def calc_Ts(F, T_AT_prev, T_LO_prev, xi_1, xi_2, xi_3, xi_4):
        """equation 17, 18"""

        T_AT = T_AT_prev + xi_1 * (
            F - xi_2 * T_AT_prev - xi_3 * (T_AT_prev - T_LO_prev)
        )
        T_LO = T_LO_prev + xi_4 * (T_AT_prev - T_LO_prev)

        return max(T_AT, 0), T_LO

    @staticmethod
    def calc_T_AT_TCRE(T_AT_prev, TCRE, timestep, E_prev):
        # TCRE in units degrees C per Tt carbon
        CDCR = TCRE / 3.67  # convert from tons C to tons carbon dioxide

        return T_AT_prev + E_prev * timestep * CDCR / 10**3  # degrees C

    @staticmethod
    def calc_Omega(psi_1, psi_2, psi_3, T_AT):
        """equation 5

        Modified to allow a Weitzman (2010) damage function with psi_3"""

        return psi_1 * T_AT + psi_2 * T_AT**2 + (psi_3 * T_AT) ** 6.754

    @staticmethod
    def calc_I(S, Q):
        return S * Q


class Objective:
    """Evaluation of the objective function from the results of a model run"""

    def __init__(self, core_calculation, setup, parameters):
        self.C = self.calc_C(core_calculation.Q, core_calculation.I)

        self.U = self.calc_U(self.C, setup.L, parameters.alpha)

        self.W = self.calc_W(self.U, setup.R)

    @staticmethod
    def calc_C(Q, I):
        """equation 7"""
        return Q - I

    @staticmethod
    def calc_U(C, L, alpha):
        """equations 2 and 8"""

        return 1000 * L * (((C / L) ** (1 - alpha) - 1) / (1 - alpha) - 1)

    @staticmethod
    def calc_W(U, R):
        """equation 1"""

        W = (U * R).sum()
        return W


class Optimization:
    def __init__(self, parameters, setup):
        x0 = self.combine_control_variables(setup.initial_control_variables)

        self.result = minimize(
            self.function_to_minimize,
            x0,
            args=(parameters, setup),
            method="SLSQP",
            bounds=tuple(setup.S_limits + setup.mu_limits),
            options={
                "disp": True,
                "maxiter": parameters.max_iterations,
                "ftol": parameters.tolerance,
            },
        )

        self.result_control_variables = self.split_control_variables(self.result.x)

    @staticmethod
    def combine_control_variables(control_variables: ControlVariables):
        """Combine the two control variables (S and mu) into a single array"""
        return np.append(control_variables.S, control_variables.mu)

    @staticmethod
    def split_control_variables(x):
        """Split the two control variables (S and mu) back into separate objects"""
        S = np.array(x[0:int(len(x) / 2)])
        mu = np.array(x[int(len(x) / 2):])
        return ControlVariables(S=S, mu=mu)

    def function_to_minimize(self, x, parameters, setup):
        # split input x back out into S and mu
        current_control_variables = self.split_control_variables(x)

        core_calculation = CoreCalculation(
            parameters=parameters,
            setup=setup,
            control_variables=current_control_variables,
        )
        objective = Objective(core_calculation, setup, parameters)

        return -objective.W  # objective is to maximize W, i.e. minimize -W


class DICE:
    """Initialize and run the DICE model
    
    Inputs:
        use_updated_parameters: bool (opt) - whether or not to use suggested parameter updates
            (otherwise defaults to DICE 2016 parameters)
         user_parameters: dict (opt) - optional dictionary of individual parameters to update
         optimize: bool (opt) - whether to run the optimization. If False, model will be initalized
            but the optimization will not be run
    
    Attributes:
        parameters: Parameters - the parameters used
        setup: Setup - all data calculated with initial model setup
        result_control_variables: The control variables after optimization
        result_calc: The other variables based on a model run with the optimal control variables
        objective: the value of the objective function after optimization
        SCC: the social cost of carbon calculated based on the variables of the result_calc
    """

    def __init__(
        self,
        use_updated_parameters: bool = False,
        user_parameters: dict = None,
        optimize: bool = True,
    ):
        # parameters
        parameters = Parameters()

        # update parameters from dictionaries
        if use_updated_parameters:
            parameters.update(updated_parameters)

        if user_parameters:
            parameters.update(user_parameters)

        self.parameters = parameters

        # initialize
        self.setup = Setup(parameters=parameters)

        self.initial_calc = CoreCalculation(
            parameters=self.parameters,
            setup=self.setup,
            control_variables=self.setup.initial_control_variables,
        )
        self.initial_objective = Objective(
            self.initial_calc, self.setup, self.parameters
        )

        if optimize:
            self.run_optimization()

    def run_optimization(self):
        """Run the optimization. Saves results of optimal run to self.result_calc"""

        start_time = time.time()
        logging.info("Running model optimization...")
        self.optimization = Optimization(parameters=self.parameters, setup=self.setup)
        logging.info(
            f"Done! Optimization run time: {(time.time()-start_time):.1f} seconds"
        )

        # calculate full results for the final control variables
        self.result_calc = CoreCalculation(
            parameters=self.parameters,
            setup=self.setup,
            control_variables=self.optimization.result_control_variables,
        )

        self.objective = Objective(self.result_calc, self.setup, self.parameters)

        self.SCC = self.setup.B * self.optimization.result_control_variables.mu ** (
            self.parameters.theta_2 - 1
        )

    def plot_results(self, plot_to_year=None):
        """Plot the results of this model, return a figure"""

        fig = plot_results({self: "model"}, plot_to_year=plot_to_year)
        return fig
