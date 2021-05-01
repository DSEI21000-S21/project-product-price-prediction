import pandas as pd
import timeit

from final.feature_extraction.clean_text import clean_text_with_extract_info
from final.data_process.format_data_wo_missing_values import create_new_categories

# process text cleaning
df = pd.read_csv("../data/train.tsv",sep="\t")
df = df[['train_id', 'name', 'item_description']]
df.dropna(inplace=True)

start = timeit.default_timer()

process_name_df = clean_text_with_extract_info(df, col_name="item_description",num_of_processes=12)
save_file_name="clean_item_description_with_features.csv"
process_name_df.to_csv("../data/%s" % save_file_name, index=False)

process_name_df = clean_text_with_extract_info(df, col_name="name",num_of_processes=12)
save_file_name="clean_item_name_with_features.csv"
process_name_df.to_csv("../data/%s" % save_file_name, index=False)

stop = timeit.default_timer()
print('Time 2: ', stop - start)


# merge files
item_description_df = pd.read_csv("data/clean_item_description_with_features.csv")
column = ['train_id', 'clean_item_description', 'bef_word_count',
       'bef_char_count', 'bef_avg_word_len', 'upper_word_count',
       'upper_char_count', 'stopword_count', 'punctuation_count',
       'number_count',  'after_word_count',
       'after_char_count', 'after_avg_word_len']
item_description_df = item_description_df[column]
column[2:] = list(map(lambda x: 'item_description_'+x, column[2:]))
item_description_df.columns = column



item_name_df = pd.read_csv("data/clean_item_name_with_features.csv")
column = ['train_id', 'clean_name', 'bef_word_count',
       'bef_char_count', 'bef_avg_word_len', 'upper_word_count',
       'upper_char_count', 'stopword_count', 'punctuation_count',
       'number_count',  'after_word_count',
       'after_char_count', 'after_avg_word_len']
item_name_df = item_name_df[column]
column[2:] = list(map(lambda x: 'item_name_'+x, column[2:]))
column[1] = 'clean_item_name'
item_name_df.columns = column

ori_df = pd.read_csv("../data/train.tsv",sep="\t")
ori_df = ori_df[['train_id', 'item_condition_id', 'category_name', 'brand_name','shipping','price']]

merge_df = pd.merge(item_description_df,item_name_df, on='train_id', how='outer')
merge_df = pd.merge(merge_df,ori_df, on='train_id', how='outer')
merge_df.to_csv("data/all_clean_text_with_features.csv" , index=False)


df = pd.concat([merge_df,
                merge_df.category_name.apply(create_new_categories)], axis=1)
df.to_csv("data/all_clean_text_with_features_and_split_category.csv", index=False)
