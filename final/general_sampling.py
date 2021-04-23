from final.random_sampling.even_sample_brand import stratified_sampling_by_brand

brand_sample_df = stratified_sampling_by_brand(file_dir="data",number_samples = 50000,
                                               replace = False, save_sample_df = True)
