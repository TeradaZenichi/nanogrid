import pandas as pd

# Carregar os dados
file_path = "data/pv_5min_cleaned.csv"  # Ajuste o caminho conforme necessário
df = pd.read_csv(file_path)

# Converter a coluna datetime para formato de data-hora, tratando erros como NaT
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Criar coluna de data para análise por dia
df["date"] = df["datetime"].dt.date

# Análise
report = []

for date, group in df.groupby("date"):
    total_entries = len(group)
    unique_values = group["Ppower"].nunique()
    all_zero = (group["Ppower"] == 0).all()
    contains_nan = group["Ppower"].isna().any()
    incomplete_day = total_entries < 288
    constant_values = unique_values == 1

    # Adiciona apenas os dias com alguma condição True
    if any([incomplete_day, constant_values, all_zero, contains_nan]):
        report.append({
            "Data": date,
            "Dias Incompletos": incomplete_day,
            "Valores Constantes": constant_values,
            "Totalmente Zerado": all_zero,
            "Contém NaN": contains_nan
        })

# Criar DataFrame do relatório
report_df = pd.DataFrame(report)

# Exibir apenas as colunas que têm valores True
if not report_df.empty:
    report_df = report_df.loc[:, (report_df != False).any(axis=0)]

# Exibir relatório no terminal
print(report_df)

# Salvar relatório em CSV
output_file = "data/relatorio_problemas.csv"
report_df.to_csv(output_file, index=False)
print(f"Relatório salvo em: {output_file}")
