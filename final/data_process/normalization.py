import numpy as np
# from sklearn.preprocessing import normalize

def normalize_numeric_col(df, skip_cols = [], print_process=False):
    numeric_columns = df.select_dtypes([np.number]).columns.tolist()
    drop_col = []
    for column in numeric_columns:
        if column in skip_cols:
            continue

        min = df[column].values.min()
        max = df[column].values.max()

        if max-min > 0:
            if print_process:
                print("Normalizing column: ", column)
            df[column] = (df[column].values - min)/(max-min)
        else:
            drop_col.append(column)

    for column in drop_col:
        print("Drop Column with all Zeros: ", column)
        numeric_columns.remove(column)
        del df[column]

    return df, numeric_columns
