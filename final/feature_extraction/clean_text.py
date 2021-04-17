import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from final.feature_extraction.extract_text_info import extract_general_text_info, extract_word_counts

def clean_text(df, col_name, stop_words):

    clean_col_name = "clean_%s"%col_name

    # Make All Text Lower Case
    df['clean_name'] = df[col_name].apply(lambda x: x.lower())

    # Removing Punctuations
    df[clean_col_name] = df[clean_col_name].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Removing Stopwords
    df[clean_col_name] = df[clean_col_name].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

    # Correct Spelling
    df[clean_col_name] = df[clean_col_name].apply(lambda x: str(TextBlob(x).correct()))

    # Lemmatization
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    df[clean_col_name] = df[clean_col_name].apply(lambda x: lemmatizer.lemmatize(x))

    # If fail to download, run below two line before download
    # import ssl
    # ssl._create_default_https_context = _create_unverified_https_context

    return df, clean_col_name

def clean_text_with_extract_info(df, col_name, save_file_name):
    stop_words = stopwords.words('english')

    # extract text info
    df = extract_word_counts(df, col_name, is_before = True)
    df = extract_general_text_info(df, col_name, stop_words=stop_words)

    # clean text
    df, clean_col_name = clean_text(df, col_name, stop_words=stop_words)

    # extract text info after clean text
    df = extract_word_counts(df, clean_col_name, is_before = False)

    # save the result
    df.to_csv("../../data/%s"%save_file_name)

    return df
