import numpy as np
from scipy.sparse import  hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def encode_categories(df, columns = [], min_df = 10, print_progress=False):
    vectorizer = CountVectorizer(min_df=min_df,
                                 ngram_range=(1, 2))

    if len(columns) == 0:
        return None
    features_name = []
    features = vectorizer.fit_transform(df[columns[0]].values)
    features_name.append(vectorizer.get_feature_names())
    for column in columns[1:]:
        features= hstack((features, vectorizer.fit_transform(df[column].values)))
        features_name.append(vectorizer.get_feature_names())
    if print_progress:
        for index, column in enumerate(columns):
            print("Size of vectorization features of", column, "is", len(features_name[index]))
        print("Shape of total vectorization features of",columns, "is", features.shape)
    return features, features_name


def encode_string_column(df, columns=[], min_df=10, max_features=15000, print_progress=False):
    vectorizer = TfidfVectorizer(min_df=min_df,
                                 max_features=max_features,
                                 ngram_range=(1, 3))

    if len(columns) == 0:
        return None
    features_name = []
    features = vectorizer.fit_transform(df[columns[0]].values)
    features_name.append(vectorizer.get_feature_names())
    for column in columns[1:]:
        features= hstack((features, vectorizer.fit_transform(df[column].values)))
        features_name.append(vectorizer.get_feature_names())

    if print_progress:
        for index, column in enumerate(columns):
            print("Size of vectorization features of", column, "is", len(features_name[index]))
        print("Shape of vectorization features of",columns, "is", features.shape)
    return features, features_name
