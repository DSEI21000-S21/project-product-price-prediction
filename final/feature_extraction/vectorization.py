from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def text_vectorizaion(df, text_col = "clean_item_description", tfidf = True, min_df=10, max_features=100000):
    if tfidf:
        vectorizer = TfidfVectorizer(min_df=min_df,
                                     max_features=max_features,
                                     ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(min_df=min_df,
                                     max_features=max_features,
                                     ngram_range=(1, 2))
    vz = vectorizer.fit_transform(df[text_col])

    return vz, vectorizer.get_feature_names()

