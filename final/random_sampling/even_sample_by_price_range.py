import math
import pandas as pd
from datetime import datetime


def stratified_sampling_by_price(file_dir="../../data",number_samples = 10000, include_high_price = False, save_sample_df = True):
    random_num = int(datetime.now().timestamp())
    # read data
    df = pd.read_csv("%s/data_wo_missing_values_split_category.csv"%file_dir) # result shape: (844460, 10)

    bins = [0, 10, 20, 30, 40,50,60,70,80,90,100, 200,500,1000,2500]
    df['price_bin'] = pd.cut(df['price'], bins)


    brand_dist = df.brand_name.value_counts()
    limit = 10
    brand_item_limit_dict = {15000: 50, 50000: 100}
    for need_sample, item_limit in brand_item_limit_dict.items():
        if number_samples >= need_sample:
            limit = item_limit
    brand_w_over_100_item  = brand_dist[brand_dist>limit].index.to_list()
    df = df.loc[df.brand_name.isin(brand_w_over_100_item), :] # result shape: (800066, 10)

    # random sampling
    number_sample_per_brand = math.ceil(number_samples / len(brand_w_over_100_item))
    if number_samples > 55000:
        number_sample_per_brand += 1
    # get sample
    sampling_df = df.groupby('brand_name').apply(lambda s: s.sample(min(len(s),number_sample_per_brand),
                                                                    random_state=random_num))

    # shuffle rows
    sampling_df = sampling_df.sample(frac=1)

    # check number of sample get
    if len(sampling_df) >  number_samples:
        sampling_df = sampling_df[:number_samples]

    print(sampling_df.brand_name.value_counts())

    if save_sample_df:
        sampling_df.to_csv("%s/random_samples/stratified_sampling_data_by_price_sz%d_%d.csv"%(file_dir,len(sampling_df),random_num), index=False)
    return sampling_df
