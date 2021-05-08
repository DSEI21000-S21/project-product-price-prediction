import numpy as np
from scipy.sparse import  hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def encode_categories(train_df,test_df, columns = [], min_df = 10, print_progress=False):
    if len(columns) == 0:
        return None

    vectorizer = CountVectorizer(min_df=min_df,
                                 ngram_range=(1, 1))
    features_name = []
    train_features = vectorizer.fit_transform(train_df[columns[0]].values)
    test_features = vectorizer.transform(test_df[columns[0]].values)
    features_name.append(vectorizer.get_feature_names())

    for column in columns[1:]:
        train_features= hstack((train_features, vectorizer.fit_transform(train_df[column].values)))
        test_features= hstack((test_features, vectorizer.transform(test_df[column].values)))
        features_name.append(vectorizer.get_feature_names())

    if print_progress:
        for index, column in enumerate(columns):
            print("Size of vectorization features of", column, "is", len(features_name[index]))
        print("Shape of train vectorization features of",columns, "is", train_features.shape)
        print("Shape of test vectorization features of",columns, "is", test_features.shape)
    return train_features.toarray(),test_features.toarray(), features_name


def encode_string_column(train_df,test_df, columns=[], min_df=10, max_features=15000, print_progress=False):
    if len(columns) == 0:
        return None

    vectorizer = TfidfVectorizer(min_df=min_df,
                                 max_features=max_features,
                                 ngram_range=(1, 3))

    features_name = []
    train_features = vectorizer.fit_transform(train_df[columns[0]].values)
    test_features = vectorizer.transform(test_df[columns[0]].values)
    features_name.append(vectorizer.get_feature_names())

    for column in columns[1:]:
        train_features= hstack((train_features, vectorizer.fit_transform(train_df[column].values)))
        test_features= hstack((test_features, vectorizer.transform(test_df[column].values)))
        features_name.append(vectorizer.get_feature_names())

    if print_progress:
        for index, column in enumerate(columns):
            print("Size of vectorization features of", column, "is", len(features_name[index]))
        print("Shape of train vectorization features of",columns, "is", train_features.shape)
        print("Shape of test vectorization features of",columns, "is", test_features.shape)
    return train_features.toarray(),test_features.toarray(), features_name
