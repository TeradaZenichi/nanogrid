from sizing import MicrogridDesign
import pandas as pd


if __name__ == "__main__":
    df_pv = pd.read_csv("data/sizing/pv_scenarios.csv")
    df_load = pd.read_csv("data/sizing/load_scenarios")


    design = MicrogridDesign('data/parameters.json', df_pv, df_load)
    design.build()
    design.optimize()
    