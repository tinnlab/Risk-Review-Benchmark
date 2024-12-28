clin_df = train_dataframes_small['clinical']
nume_gol = []
for col in clin_df.columns:
    if clin_df[col].nunique() > 5:
        nume_gol.append(col)

clin_df_num = clin_df[nume_gol]