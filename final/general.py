import pandas as pd
from final.feature_extraction.clean_text import clean_text_with_extract_info

df = pd.read_csv("../data/train.tsv",sep="\t")

process_name_df = clean_text_with_extract_info(df, col_name="name", save_file_name="clean_item_name_with_features.csv")

print(process_name_df.head(5))
