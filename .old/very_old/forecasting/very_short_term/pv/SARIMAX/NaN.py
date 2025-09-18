import pandas as pd

def identificar_nans(csv_file):
    # Lê o CSV; caso exista uma coluna datetime, podemos usá-la como índice
    try:
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
    except Exception as e:
        print("Não foi possível usar a coluna 'datetime' como índice. Carregando sem parse_dates.")
        df = pd.read_csv(csv_file)
    
    # Verifica a quantidade de NaNs na coluna 'Ppower'
    n_nans = df['Ppower'].isna().sum()
    print(f"Número de valores NaN encontrados em 'Ppower': {n_nans}")
    
    if n_nans > 0:
        # Exibe as linhas que contém NaN na coluna 'Ppower'
        nan_rows = df[df['Ppower'].isna()]
        print("Linhas com valores NaN:")
        print(nan_rows)
    else:
        print("Nenhum valor NaN encontrado na coluna 'Ppower'.")

if __name__ == '__main__':
    csv_file = r'C:/Users/Lucas/Code/nanogrid/data/pv_5min.csv'
    identificar_nans(csv_file)
