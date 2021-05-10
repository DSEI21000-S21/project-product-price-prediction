import pandas as pd
import numpy as np
from final.data_process.format_missing_data import separate_categories_replace_nan_w_missing, fill_na_with_missing
from final.data_process.visualization import visualization_str_column_attributes
from final.data_process.text_cleaning import cleaning_text
from final.feature_extraction.extract_text_information import get_word_count, get_special_char_count
from final.data_process.normalization import normalize_numeric_col


from final.random_sampling.even_sample_brand import stratified_sampling_by_brand
from final.random_sampling.even_sample_category import stratified_sampling_by_category
from final.random_sampling.even_sample_by_price_range import stratified_sampling_by_price

from final.feature_extraction.text_vectorization import encode_categories,encode_string_column

df = pd.read_csv("data/train.tsv",sep="\t")

# fill missing data
df = separate_categories_replace_nan_w_missing(df)
df = fill_na_with_missing(df)

# remove invalidate price -> price must greater than 0
original_num_item = len(df)
df = df[df.price > 0]
print("number of item with price less than or equal to 0: ", abs(len(df)-original_num_item))

# visualize distribution for each column
visualization_str_column_attributes(df)

# clean text
df['clean_name'] = cleaning_text(df['name'].values)
df['clean_description'] = cleaning_text(df['item_description'].values)
df['c1'] = cleaning_text(df['c1'].values)
df['c2'] = cleaning_text(df['c2'].values)
df['c3'] = cleaning_text(df['c3'].values)
df['brand_name'] = cleaning_text(df['brand_name'].values)

# extract features
df = get_word_count(df, columns = ["clean_name", "clean_description", 'c1', 'c2', 'c3','brand_name'], print_col=True)
error_item = df[df.isnull().any(axis=1)]
error_item.to_csv("data/error_data.csv", index=False)

df = df.dropna()
df = get_special_char_count(df, columns = ["name", "item_description"], print_col=True)

df.drop(["name", "item_description"], axis=1, inplace=True)

# normalize numeric data
df, numeric_column = normalize_numeric_col(df,skip_cols=['train_id','item_condition_id','price','shipping'],print_process=True)
df.to_csv("data/clean_data_with_text_features.csv", index=False)

# --------
df = pd.read_csv("data/clean_data_with_text_features.csv")

# visualize relation between different attribute with price
# to be add

# sampling data
price_sample_df = stratified_sampling_by_price(df, file_dir="data",number_samples = 100000,
                                               include_high_price = True, save_sample_df = True)

brand_sample_df = stratified_sampling_by_brand(df, file_dir="data",number_samples = 100000,
                                               replace = False, save_sample_df = True)

c1_sample_df = stratified_sampling_by_category(df, file_dir="data", category_name  = "c1",number_samples = 100000,
                                               replace = False, save_sample_df = True)

c2_sample_df = stratified_sampling_by_category(df, file_dir="data", category_name  = "c2",number_samples = 100000,
                                               replace = False, save_sample_df = True)

c3_sample_df = stratified_sampling_by_category(df, file_dir="data", category_name  = "c3",number_samples = 100000,
                                               replace = False, save_sample_df = True)


#-------------
sample_df = pd.read_csv("data/random_samples/stratified_sampling_clean_text_data_by_brand_name_sz93550_1620226553.csv")


# train test split


cat_features, cat_features_name = encode_categories(sample_df, columns = ['c1','c2','c3','brand_name'], min_df = 10, print_progress=True)
str_features, str_features_name = encode_string_column(sample_df, columns=['clean_name', 'clean_description'],
                                                       min_df=10, max_features=15000, print_progress=True)
