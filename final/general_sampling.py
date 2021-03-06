from final.random_sampling.even_sample_brand import stratified_sampling_by_brand
from final.random_sampling.even_sample_category import stratified_sampling_by_category
from final.random_sampling.even_sample_by_price_range import stratified_sampling_by_price


price_sample_df = stratified_sampling_by_price(file_dir="../data",number_samples = 50000,
                                               include_high_price = True, save_sample_df = True)

brand_sample_df = stratified_sampling_by_brand(file_dir="../data",number_samples = 50000,
                                               replace = False, save_sample_df = True)

c1_sample_df = stratified_sampling_by_category(file_dir="../data", category_name  = "c1",number_samples = 50000,
                                               replace = False, save_sample_df = True)

c2_sample_df = stratified_sampling_by_category(file_dir="../data", category_name  = "c2",number_samples = 50000,
                                               replace = False, save_sample_df = True)

c3_sample_df = stratified_sampling_by_category(file_dir="../data", category_name  = "c3",number_samples = 50000,
                                               replace = False, save_sample_df = True)
