import string
from multiprocessing import  Pool
from functools import partial
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from final.feature_extraction.extract_text_info import extract_general_row_info, extract_row_word_counts
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
# If fail to download, run below two line before download
import ssl
try:
    ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = ssl._create_unverified_context


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def clean_row_text(x, col_name, clean_col_name, stop_words, lemmatizer):
    x[clean_col_name] = x[col_name].lower()
    x[clean_col_name] = x[clean_col_name].translate(str.maketrans('', '', string.punctuation))
    x[clean_col_name] = " ".join(x for x in x[clean_col_name].split() if x not in stop_words)
    x[clean_col_name] = str(TextBlob(x[clean_col_name]).correct())
    x[clean_col_name] = lemmatizer.lemmatize(x[clean_col_name])
    return x




def extract_info(data_subset,col_name,stop_words):
    return data_subset.progress_apply(extract_general_row_info, col_name=col_name, stop_words=stop_words, axis=1)


def extract_counts(data_subset,col_name,prefix):
    return data_subset.progress_apply(extract_row_word_counts, col_name=col_name, prefix=prefix, axis=1)

def clean_text(data_subset,col_name, clean_col_name, stop_words, lemmatizer):
    return data_subset.progress_apply(clean_row_text, col_name=col_name, clean_col_name=clean_col_name, stop_words=stop_words, lemmatizer=lemmatizer, axis=1)

# def parallelize_on_rows(data, func, num_of_processes=8):
#     return parallelize(data, partial(run_on_subset, func), num_of_processes)


def clean_text_with_extract_info(df, col_name, num_of_processes = 8):

    nltk.download('wordnet')
    nltk.download('stopwords')
    lemmatizer = WordNetLemmatizer()
    clean_col_name = "clean_%s" % col_name
    stop_words = stopwords.words('english')

    # extract text info
    df = parallelize(df, partial(extract_counts, col_name=col_name, prefix="bef"), num_of_processes)
    df = parallelize(df, partial(extract_info, col_name=col_name, stop_words=stop_words), num_of_processes)



    # clean text
    df = parallelize(df, partial(clean_text, col_name=col_name, stop_words=stop_words,
                                 clean_col_name=clean_col_name, lemmatizer=lemmatizer), num_of_processes)
    # extract text info after clean text
    df = parallelize(df, partial(extract_counts, col_name=clean_col_name, prefix="after"), num_of_processes)



    return df
