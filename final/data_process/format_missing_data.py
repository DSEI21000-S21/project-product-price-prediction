
def separate_categories_replace_nan_w_missing(df):
    cat_columns = df.category_name.str.split('/')
    for i in range(3):
        df['c%d'%(i+1)] = cat_columns.str.get(0)
    df.drop('category_name', axis=1, inplace=True)
    return df


def fill_na_with_missing(df):
    for column in df.columns:
        if column == "item_description":
            df[column].fillna('No description yet', inplace=True)
        else:
            df[column].fillna('missing', inplace=True)
    return df

