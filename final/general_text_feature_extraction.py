import pandas as pd
from final.feature_extraction.clean_text import clean_text_with_extract_info

# df = pd.read_csv("../data/sample_data.csv")
df = pd.read_csv("../data/train.tsv",sep="\t")
col_name = 'name'
df = df[['train_id', col_name]]


process_name_df = clean_text_with_extract_info(df, col_name=col_name)


# save the result
save_file_name="clean_item_name_with_features.csv"
process_name_df.to_csv("/home/jchen/Other/ml/data/%s" % save_file_name, index=False)

