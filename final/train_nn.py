import pandas as pd
import numpy as np
from final.model_evaluation.keras_model import neural_network
from sklearn.model_selection import train_test_split
from final.random_sampling.even_sample_brand import stratified_sampling_by_brand
from final.feature_extraction.text_vectorization import encode_categories,encode_string_column

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression # F-value between label/feature for regression tasks.
from final.helper.save_data import save_np_file

df = pd.read_csv("data/clean_data_with_text_features.csv")
# df = stratified_sampling_by_brand(df, file_dir="data",number_samples = 10000,
#                                                replace = False, save_sample_df = False)



Y = np.log1p(df['price'])
df.drop(['price'], axis=1, inplace=True)


train_df, test_df , y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=12342)
print('Train size: %s, Test size: %s'%(train_df.shape, test_df.shape))


# feature extraction
train_cat_features, test_cat_features, train_cat_features_name = encode_categories(train_df, test_df,
                                                                columns = ['c1','c2','c3','brand_name'],
                                                                min_df = 10, print_progress=True)
train_str_features, test_str_features, train_str_features_name = encode_string_column(train_df, test_df,
                                                                   columns=['clean_name', 'clean_description'],
                                                                   min_df=10, max_features=15000,
                                                                   print_progress=True)
other_columns = list(train_df.select_dtypes([np.number]).columns)
other_columns.remove('train_id')
train_other_features = train_df[other_columns].values
test_other_features = test_df[other_columns].values
all_train = np.hstack((train_cat_features, train_str_features, train_other_features))
all_test = np.hstack((test_cat_features, test_str_features, test_other_features))
print('Train features size: %s, Test features size: %s'%(all_train.shape,
                                                         all_test.shape))


del train_cat_features, train_str_features, train_other_features
del test_cat_features, test_str_features, test_other_features
del train_df, test_df

# select k best
skb = SelectKBest(f_regression, k=5000)
x_skb_select_train = skb.fit_transform(all_train, y_train)
x_skb_select_test = skb.transform(all_test)

save_np_file(dir = "data", filename="select_k_best_train.npy", data=x_skb_select_train)
save_np_file(dir = "data", filename="select_k_best_test.npy", data=x_skb_select_test)
save_np_file(dir = "data", filename="y_train.npy", data=y_train)
save_np_file(dir = "data", filename="y_test.npy", data=y_test)

model = neural_network(model_prefix="select_k_best")
model.fit(x_skb_select_train, y_train.values, x_skb_select_test, y_test.values,
          n_epoch=100,epoch=1, bs=128)

skb_select_train_pred = model.predict(x_skb_select_train)
skb_select_test_pred = model.predict(x_skb_select_test)
save_np_file(dir = model.model_name, filename="select_k_best_y_train.npy", data=skb_select_train_pred)
save_np_file(dir = model.model_name, filename="select_k_best_y_test.npy", data=skb_select_test_pred)
# model.evaluation(y_train, skb_select_train_pred, y_test, skb_select_test_pred, price_split=30)

all_model = neural_network(model_prefix="all_data")
all_model.fit(all_train, y_train.values, all_test, y_test.values,
              n_epoch=100,epoch=1, bs=128)

all_train_pred = all_model.predict(all_train)
all_test_pred = all_model.predict(all_test)
save_np_file(dir = all_model.model_name, filename="all_y_train.npy", data=all_train_pred)
save_np_file(dir = all_model.model_name, filename="all_y_test.npy", data=all_test_pred)
# all_model.evaluation(y_train, all_train_pred, y_test, all_test_pred, price_split=30)


# load data
# skb_select_loss = np.load("select_k_best_NN_256_dr1e-01_64_dr1e-01_16_dr1e-01_lr1e-03_loss.npy")
model_name = "all_data_NN_256_dr1e-01_64_dr1e-01_16_dr1e-01_lr1e-03"
all_loss = np.load("%s/%s_loss.npy"%(model_name,model_name))
