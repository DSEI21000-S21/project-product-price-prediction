import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


punctuation_remover = str.maketrans(string.punctuation, ' '*len(string.punctuation))
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def cleaning_text(words):
    clean_text = []
    for  x in words:
        # lower case the x
        x = x.lower()

        # get only character and number
        x = re.sub('[^A-Za-z0-9]+', ' ', x)

        # remove stop word and not alphabetic word
        x = " ".join(x for x in x.split() if x not in stop_words or x.isalpha())

        # reduces the word-forms to linguistically valid lemmas
        x = lemmatizer.lemmatize(x)

        # replace punctuation by white space
        x = x.translate(punctuation_remover)

        clean_text.append(x)
    return clean_text



