import string
import re
from final.data_process.text_cleaning import stop_words

def get_word_count(df, columns, replace_names = {}, print_col=False):
    for col in columns:
        if col in replace_names:
            replace_name = replace_names[col]
        else:
            replace_name = col
        if print_col:
            print("processing column -- %s" % replace_name)
        df["%sWordCount"%replace_name] = df[col].apply(lambda x: len(x.split()))
        df["%sLength" % replace_name] = df[col].apply(lambda x: len(x))
        df["%sUpperCount" % replace_name] = df[col].str.count('[A-Z]')
        df["%sLowerCount" % replace_name] = df[col].str.count('[a-z]')


        df["%sLowerRatio" % replace_name] = df["%sLowerCount" % replace_name] / df["%sLength" % replace_name]
        df["%sUpperRatio" % replace_name] = df["%sUpperCount" % replace_name] / df["%sLength" % replace_name]
        df["%sAvgWordLen" % replace_name] = df["%sLength" % replace_name] / df["%sWordCount" % replace_name]

    return df


def get_special_char_count(df, columns, replace_names = {}, print_col=False):
    for col in columns:
        if col in replace_names:
            replace_name = replace_names[col]
        else:
            replace_name = col
        if print_col:
            print("processing column -- %s" % replace_name)
        df["%sStopWordCount"%replace_name] = df[col].apply(lambda x: len([x for x in x.split() if x.lower() in stop_words]))
        df["%sPunctuationCount"%replace_name] = df[col].apply(lambda x: len([x for x in x if x in string.punctuation]))
        df["%sSpecialCount" % replace_name] = df[col].apply(lambda x: len(x) - len(re.sub('[^A-Za-z0-9]+', ' ', x)))

    return df
