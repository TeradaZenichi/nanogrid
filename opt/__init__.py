import json
import pyomo.environ as pyo
from datetime import datetime

class BatteryEnergyManagement:
    def __init__(self, horizon, start_date, end_date, data_path):
        """
        Initializes the problem with a defined time horizon, optimization start and end dates,
        and loads parameters from a JSON file.

        Parameters:
            horizon (int): Number of time periods.
            start_date (datetime): Optimization start date.
            end_date (datetime): Optimization end date.
            data_path (str): Path to the JSON file containing the problem parameters.
        """
        self.horizon = horizon
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path
        self.model = None
        self._create_model(data_path)
    
    def _create_model(self, data_path):
        """
        Creates the Pyomo model using parameters loaded from a JSON file.

        The JSON file should contain keys such as:
            - "battery_capacity" (maximum energy capacity of the battery)
            - "Pmax"             (maximum power, i.e. battery_max_power)
            - "efficiency"       (charging/discharging efficiency)
            - "timestep1"        (Δt1, time step for the first SOC update)
            - "timestep2"        (Δt2, time step for the subsequent periods)
        """
        # Load problem parameters from the JSON file
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        battery_capacity = data.get("battery_capacity")
        battery_max_power = data.get("Pmax")
        efficiency = data.get("efficiency")
        dt1 = data.get("timestep1")
        dt2 = data.get("timestep2")
        
        m = pyo.ConcreteModel()
        
        # Set of time periods (e.g., 0 to horizon-1)
        m.T = pyo.RangeSet(0, self.horizon - 1)
        
        # Mutable parameters to be updated with forecasts
        m.pv = pyo.Param(m.T, mutable=True, initialize=0)         # PV generation forecast
        m.demand = pyo.Param(m.T, mutable=True, initialize=0)       # Demand forecast
        m.grid_status = pyo.Param(m.T, mutable=True, initialize=1)  # Grid status (1: available, 0: unavailable)
        
        # Battery system constants loaded from JSON
        m.battery_max_power = battery_max_power
        m.efficiency = efficiency
        
        # Scalar parameter for the initial state of charge (SOC) of the battery
        m.initial_soc = pyo.Param(initialize=0, mutable=True)
        
        # Decision variables
        m.EBESS = pyo.Var(m.T, bounds=(0, battery_capacity))       # Battery state of charge
        m.PBESS_c = pyo.Var(m.T, within=pyo.NonNegativeReals)        # Charging power
        m.PBESS_d = pyo.Var(m.T, within=pyo.NonNegativeReals)        # Discharging power
        m.PEDSp = pyo.Var(m.T, within=pyo.NonNegativeReals)          # Positive load dispatch (excess)
        m.PEDSn = pyo.Var(m.T, within=pyo.NonNegativeReals)          # Negative load dispatch (deficit)
        m.XLoad = pyo.Var(m.T, within=pyo.NonNegativeReals)          # Load curtailment
        
        # Constraint: Battery state of charge (SOC) balance with variable time steps
        def soc_balance_rule(m, t):
            if t == m.T.first():
                # For the first period, use the provided initial SOC and apply Δt1
                soc_prev = m.initial_soc
                dt = dt1
            else:
                # For subsequent periods, use the previous period's SOC and apply Δt2
                soc_prev = m.EBESS[t - 1]
                dt = dt2
            # SOC update: previous SOC + (charging minus discharging) * appropriate time step
            return m.EBESS[t] == soc_prev + dt * (m.PBESS_c[t] * m.efficiency - m.PBESS_d[t] / m.efficiency)
        
        m.soc_balance = pyo.Constraint(m.T, rule=soc_balance_rule)
        
        # Additional constraints (e.g., power balance, limits on charging/discharging) can be added here
        
        self.model = m
    
    def optimize(self, pv_forecast, demand_forecast, grid_status, initial_battery_state):
        """
        Updates the model parameters with the forecasts and current battery state, then solves the optimization problem.
        
        Parameters:
            pv_forecast (list or dict): PV generation forecast for each period.
            demand_forecast (list or dict): Demand forecast for each period.
            grid_status (list or dict): Grid status for each period.
            initial_battery_state (float): Current battery state of charge.
            
        Returns:
            dict: A dictionary with the solution (values for each variable per period) or None if an optimal solution is not found.
        """
        m = self.model
        
        # Update the initial battery state parameter
        m.initial_soc = initial_battery_state
        
        # Update model parameters for each time period t
        for t in m.T:
            m.pv[t] = pv_forecast[t]
            m.demand[t] = demand_forecast[t]
            m.grid_status[t] = grid_status[t]
        
        # Select and execute the solver (e.g., GLPK)
        solver = pyo.SolverFactory('glpk')
        results = solver.solve(m, tee=False)
        
        # Check if the solution is optimal
        if (results.solver.status == pyo.SolverStatus.ok) and \
           (results.solver.termination_condition == pyo.TerminationCondition.optimal):
            solution = {t: {
                'EBESS': pyo.value(m.EBESS[t]),
                'PBESS_c': pyo.value(m.PBESS_c[t]),
                'PBESS_d': pyo.value(m.PBESS_d[t]),
                'PEDSp': pyo.value(m.PEDSp[t]),
                'PEDSn': pyo.value(m.PEDSn[t]),
                'XLoad': pyo.value(m.XLoad[t])
            } for t in m.T}
            return solution
        else:
            print("The solver did not find an optimal solution.")
            return None


# if name main
if __name__ == "__main__":
    # Example usage
    horizon = 24
    start_date = datetime(2022, 1, 1, 0, 0, 0)
    end_date = datetime(2022, 1, 1, 23, 59, 59)
    data_path = "data/battery_parameters.json"
    
    # Create the battery energy management problem
    problem = BatteryEnergyManagement(horizon, start_date, end_date, data_path)
    
    # Example forecasts and initial battery state
    pv_forecast = {t: 0 for t in range(horizon)}
    demand_forecast = {t: 0 for t in range(horizon)}
    grid_status = {t: 1 for t in range(horizon)}
    initial_battery_state = 0
    
    # Solve the optimization problem
    solution = problem.optimize(pv_forecast, demand_forecast, grid_status, initial_battery_state)
    
    if solution is not None:
        print("Optimal solution found:")
        for t, values in solution.items():
            print(f"Period {t}: {values}")
    else:
        print("No optimal solution found.")