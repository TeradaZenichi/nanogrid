import pyomo.environ as pyo


class Parameters:
    def __init__(self, config):
        self.config = config
        self.Δt = self.config.get('time_step', 5)/60

    def build(self, model):
        self.Ωt = range(24)
        pass

 


class BESS:
    def __init__(self, config, parameters):
        self.param = parameters

    def build(self, model):
        model.BESS = pyo.Block()
        self.var = model.BESS
        self.var.Pbess = pyo.Var(self.param.Ωt, within=pyo.NonNegativeReals)

        

class PV:
    def __init__(self, config, parameters):
        self.param = parameters
    

class Load:
    def __init__(self, config, parameters):
        self.param = parameters
    
        
class Grid:
    def __init__(self, config, parameters):
        self.param = parameters
    

class EV:
    def __init__(self, config, parameters):
        self.param = parameters
    

class MicrogridDesign:
    def __init__(self, config, df_pv, df_load):
        self.param = Parameters(config.get('Parameters', {}))
        self.bess  = BESS(config.get('BESS', {}), self.param)
        self.load  = Load(config.get('Load', {}), self.param)
        self.grid  = Grid(config.get('Grid', {}), self.param)
        self.pv    = PV(config.get('PV', {}), self.param)

        pass

    def build(self):
        self.model = pyo.ConcreteModel()
        self.param.build(self.model)
        self.bess.build(self.model)
        self.load.build(self.model)
        self.grid.build(self.model)
        self.pv.build(self.model)

    def optimize(self):
        pass

    def get_results(self):
        pass


if __name__ == "__main__":
    home = SmartHome({})
    home.build()