from opt.plot_results import plot_custom_dispatch_with_soc
import pandas as pd


df = pd.read_csv("outputs/operation_real.csv", index_col=0, parse_dates=True)
plot_custom_dispatch_with_soc(df, start_dt="2009-05-02 12:00:00", end_dt="2009-05-02 15:00:00")