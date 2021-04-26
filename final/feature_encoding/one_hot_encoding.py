import pandas as pd

def one_hot_encode_feature(df, encode_column,drop_first=True):
    # using get_dummies, drop one column as it is correlated with other column
    encode_df = pd.get_dummies(df[[encode_column]], drop_first=drop_first)

    merge_df = pd.concat([df, encode_df], axis=1)
    merge_df.drop([encode_column], axis=1, inplace=True)

    return merge_df, encode_df.columns
