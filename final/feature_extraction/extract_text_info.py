import string

def extract_general_text_info(df, col_name, stop_words):
    # extract_text_count
    df['upper_word_count'] = df[col_name].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    df['upper_char_count'] = df[col_name].apply(lambda x: len([x for x in x if x.isupper()]))
    df['stopword_count'] = df[col_name].apply(lambda x: len([x for x in x.split() if x.lower() in stop_words]))
    df['punctuation_count'] = df[col_name].apply(lambda x: len([x for x in x if x in string.punctuation]))
    df['number_count'] = df[col_name].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    return df


def extract_word_counts(df, col_name, is_before = False):
    prefix = 'bef' if is_before else 'aft'
    df['%s_word_count' % prefix] = df[col_name].apply(lambda x: len(x.split()))
    df['%s_char_count' % prefix] = df[col_name].apply(lambda x: len(x))
    df['%s_avg_word_len' % prefix] = df['%s_char_count' % prefix] / df['%s_word_count' % prefix]
    return df
