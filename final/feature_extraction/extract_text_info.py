import string

def extract_general_row_info(x, col_name, stop_words):
    # extract_text_count
    x['upper_word_count'] = len([x for x in x[col_name].split() if x.isupper()])
    x['upper_char_count'] = len([x for x in x[col_name] if x.isupper()])
    x['stopword_count'] = len([x for x in x[col_name].split() if x.lower() in stop_words])
    x['punctuation_count'] = len([x for x in x[col_name] if x in string.punctuation])
    x['number_count'] = len([x for x in x[col_name].split() if x.isdigit()])
    return x

def extract_row_word_counts(x, col_name, prefix):
    x['%s_word_count' % prefix] = len(x[col_name].split())
    x['%s_char_count' % prefix] = len(x[col_name])
    x['%s_avg_word_len' % prefix] = x['%s_char_count' % prefix] / x['%s_word_count' % prefix]
    return x

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
