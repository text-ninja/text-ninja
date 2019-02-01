# /nlp/preprocess.py

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np
import sys

INVALID_STRING = 'Error while pre-processing the document. Invalid input to the function remove_special_chars.'
INVALID_LOW_FREQ = 'Error while pre-processing the document. Passed document does not contain any text.'
INVALID_PREPROCESS = 'Error while pre-processing the document. Error in the function rm_stopwords_stem_lowfreq'

"""
    remove_special_chars - to remove special characters from a string.    
"""
def remove_special_chars(paragraphs):

    if len(paragraphs) == 0 or paragraphs is None:
        raise ValueError(INVALID_STRING)    
    try:
        clean_string = re.sub(r"[^a-zA-Z0-9]+", ' ', paragraphs)
        clean_string = re.sub(r'\s+', ' ', clean_string)
        return clean_string.strip()
    except Exception as e:
        print('Error: ', e)
        print('Error while removing special characters.')
        sys.exit(1)


"""
    get_paragraphs - this function will split the data into sentences
"""
def fetch_low_freq_words(paragraphs):

    if paragraphs.size == 0:
        raise ValueError(INVALID_LOW_FREQ)    

    try:
        paragraphs = np.hstack(np.char.split(paragraphs))
        unique, count = np.unique(paragraphs, return_counts=True)
        word_count = np.asarray((unique, count)).T
        low_freq_words = word_count[np.where(word_count[:,1].astype(int) < 2),0]
        return low_freq_words
    except Exception as e:
        print('Error: ', e)
        print('Error while fetching low frequency words from the document.')
        sys.exit(1)


"""
    rm_stopwords_stem_lowfreq - to remove the stop words, low frequency words and perform stemming.
"""
def rm_stopwords_stem_lowfreq(paragraphs, low_freq_words):

    if paragraphs.size == 0:
        raise ValueError(INVALID_PREPROCESS)

    try:
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        tokenized_paragraphs = np.char.split(paragraphs)

        for p in range(tokenized_paragraphs.shape[0]):
            paragraphs[p] = ' '.join([ps.stem(word) for word in tokenized_paragraphs[p] if
                            word not in stop_words and len(word) > 2 and word not in low_freq_words])
        return paragraphs
    except Exception as e:
        print('Error: ', e)
        print('Error while removing preprocessing the data.')
        sys.exit(1)
