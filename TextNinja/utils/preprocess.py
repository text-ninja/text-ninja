# libraries needed to pre-process the data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import word_tokenize


class PreprocessData:

    """
        Class to pre-process the data.

        Functions:

        remove_special_chars - to remove special characters from a string.

        get_paragraphs - this function will split the data into sentences

        rm_stopwords_stem_lowfreq - to remove the stop words, low frequency words and perform stemming.
    """

    def remove_special_chars(self, input_string):
        clean_string = re.sub(r"[^a-zA-Z0-9]+", ' ', input_string)
        clean_string = re.sub('\s+', ' ', clean_string)
        return clean_string.strip()

    def find_low_freq_words(self, input_doc):
        words = input_doc.split(' ')
        low_freq_words = [word for word in words if input_doc.count(word) < 2]
        return low_freq_words

    def rm_stopwords_stem_lowfreq(self, input_string, low_freq_words):
        filtered_text = []
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(input_string)
        word_tokens = [word.lower() for word in word_tokens if
                       word not in stop_words and len(word) > 2 and word not in low_freq_words]
        sentence = ' '.join(word_tokens)
        filtered_text = ps.stem(sentence)
        return filtered_text