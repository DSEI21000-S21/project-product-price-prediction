
def separate_categories_replace_nan_w_missing(df):
    cat_columns = df.category_name.str.split('/')
    for i in range(3):
        df['c%d'%(i+1)] = cat_columns.str.get(0)
    df.drop('category_name', axis=1, inplace=True)
    return df

def fill_na_with_missing(df):
    for column in df.columns:
        df[column].fillna('missing', inplace=True)
    return df

def abc():
    pass

# df = pd.read_csv("../../data/train.tsv", sep="\t")
# df.dropna(inplace=True)
#
# df = pd.concat([df[['train_id', 'name', 'item_condition_id', 'brand_name','price', 'shipping', 'item_description']],
#                     df.category_name.apply(create_new_categories)], axis=1)
# df.dropna(inplace=True)
# df.to_csv("../../data/data_wo_missing_values_split_category.csv", index=False)

# df.shape  # (846982, 10)
