import pandas as pd

def create_new_categories(x):
    try:
        categories = x.split('/')
        if len(categories) == 3:
            return pd.Series({'c1':categories[0].lower().strip(),
                              'c2':categories[1].lower().strip(),
                              'c3':categories[2].lower().strip()})
        return pd.Series({'c1': None, 'c2':None, 'c3': None})
    except:
        return pd.Series({'c1': None, 'c2':None, 'c3': None})


df = pd.read_csv("../../data/train.tsv", sep="\t")
df.dropna(inplace=True)

df = pd.concat([df[['train_id', 'name', 'item_condition_id', 'brand_name','price', 'shipping', 'item_description']],
                    df.category_name.apply(create_new_categories)], axis=1)

df.to_csv("../../data/data_wo_missing_values_split_category.csv")

# df.shape  # (846982, 10)
