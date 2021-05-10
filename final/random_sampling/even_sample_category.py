import math
import pandas as pd
from datetime import datetime


def stratified_sampling_by_category(df, file_dir="data", category_name  = "c1", number_samples = 10000, replace = False, save_sample_df = True,file_name="stratified_sampling_clean_text_data_by"):
    random_num = int(datetime.now().timestamp())
    limit = 50
    add_sample = 0
    #C1
    # the c1 with minimum item is `handmade` with 70 item - total 10  category
    # if number_samples > 7000, should not include handmade category
    if category_name  == "c1" and number_samples > 7000:
        df = df.loc[df.c1 != "handmade", :]
    elif category_name  == "c2":
        c2_dist = df.c2.value_counts()

        c2_item_limit_dict = {7000: 80, 10000: 120, 30000: 300,40000: 450, 50000: 600}
        for need_sample, item_limit in c2_item_limit_dict.items():
            if number_samples >= need_sample:
                limit = item_limit
        if not replace:
            if number_samples >= 50000:
                add_sample = 2
            if number_samples >= 20000:
                add_sample = 4

        select_c2 = c2_dist[c2_dist > limit].index.to_list()
        df = df.loc[df.c2.isin(select_c2), :]

    elif category_name  == "c3":
        c3_dist = df.c3.value_counts()
        c3_item_limit_dict = {15000: 100, 30000: 150, 50000: 250}
        for need_sample, item_limit in c3_item_limit_dict.items():
            if number_samples >= need_sample:
                limit = item_limit
        if number_samples >= 50000 and not replace:
            add_sample = 1

        select_c3 = c3_dist[c3_dist > limit].index.to_list()
        df = df.loc[df.c3.isin(select_c3), :]
    else: # category_name not exit
        return None

    # random sampling
    number_sample_per_category = math.ceil(number_samples / len(df[category_name].value_counts()))
    number_sample_per_category += add_sample
    # get sample
    sampling_df = df.groupby(category_name).apply(lambda s: s.sample(min(len(s),number_sample_per_category),
                                                            replace = replace,random_state=random_num))
    # shuffle rows
    sampling_df = sampling_df.sample(frac=1)

    # check number of sample get
    if len(sampling_df) >  number_samples:
        sampling_df = sampling_df[:number_samples]
    print(sampling_df[category_name].value_counts())
    if save_sample_df:
        sampling_df.to_csv("%s/random_samples/%s_%s_sz%d_%d.csv"%(file_dir,file_name,category_name,len(sampling_df),random_num),
                           index=False)
    return sampling_df
