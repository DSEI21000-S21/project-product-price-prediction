import math
import pandas as pd
from datetime import datetime


def sampling_price(s,number_sample_per_price,replace,random_num, extract_num = 0):
    replace_item = False

    if s.iloc[0]['price_bin'] == pd.Interval(left=500, right=2500):
        replace_item = replace
    if replace:
        if s.iloc[0]['price_bin'] == pd.Interval(left=500, right=2500):
            number_sample_per_price -= 15 * extract_num
        else:
            number_sample_per_price +=  extract_num

    return s.sample(number_sample_per_price,
                    replace = replace_item,random_state=random_num)
def stratified_sampling_by_price(file_dir="../../data",number_samples = 10000, include_high_price = False, save_sample_df = True):
    random_num = int(datetime.now().timestamp())
    replace = False
    extract_num = 0
    # read data
    df = pd.read_csv("%s/data_wo_missing_values_split_category.csv"%file_dir) # result shape: (844460, 10)

    bins = [0,5, 10,15, 20, 25, 30, 40,50,60,70,80,90,100, 200,500, 2500] # 17  bins
    df['price_bin'] = pd.cut(df['price'], bins)
    print(df['price_bin'].value_counts())
    if not include_high_price:
        df = df.loc[df.price_bin != pd.Interval(left=500, right=2500), :] # result shape: (800066, 10)
    else:
        replace = True
        extract_num = (number_samples -  (17*1000)) // 5000

    # random sampling
    number_sample_per_price = math.ceil(number_samples / len(df['price_bin'].value_counts()))

    # get sample
    sampling_df = df.groupby('price_bin').apply(lambda s: sampling_price(s, number_sample_per_price,replace,random_num,extract_num))

    # shuffle rows
    sampling_df = sampling_df.sample(frac=1)

    # check number of sample get
    if len(sampling_df) >  number_samples:
        sampling_df = sampling_df[:number_samples]

    print(sampling_df['price_bin'].value_counts())

    if save_sample_df:
        file_name = '_w%shigh_sz%d_%d' % ('' if include_high_price else 'o',len(sampling_df),random_num)
        sampling_df.to_csv("%s/random_samples/stratified_sampling_data_by_price%s.csv"%(file_dir,file_name), index=False)
    return sampling_df
