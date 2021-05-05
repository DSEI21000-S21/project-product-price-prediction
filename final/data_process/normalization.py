import numpy as np
from sklearn.preprocessing import normalize

def normalize_numeric_col(df, skip_cols = [], print_process=False):
    numeric_columns = df.select_dtypes([np.number]).columns
    for column in numeric_columns:
        if column in skip_cols:
            continue
        if print_process:
            print("Normalizing column: ", column)
        df[column] = normalize(df[column].values.reshape(1, -1)).reshape(-1, 1)
    return df, numeric_columns
