import sys
import os
import unittest
import numpy as np

# Sets the execution path
sys.path.insert(0, os.path.realpath('./'))

# internal modulesunit
from nlp.preprocess import remove_special_chars, fetch_low_freq_words, rm_stopwords_stem_lowfreq
from nlp.tfidf_feature_extraction import calc_freq_distr, calc_tf_idf
from nlp.utils import preprocess_data, compute_similarity
# from nlp.utils


class TestUtils(unittest.TestCase):
    """
    Class containing all the unit tests for the utility functions.
    """    
    def test_remove_special_chars(self):
        """
        Tests if the remove_special_chars function works as expected.
        """
        document1 = """Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query."""
        cleaned_text = """Tf idf stands for term frequency inverse document frequency and the tf idf weight is a weight often used in information retrieval and text mining This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus Variations of the tf idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document s relevance given a user query"""
        clean_text = remove_special_chars(document1)
        self.assertTrue(len(clean_text) > 0)
        self.assertEquals(clean_text, cleaned_text)
 

    def test_fetch_low_freq_words(self):
        """
        Tests if the fetch_low_freq_words function works as expected.
        """
        para1 = """Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query."""
        paragraph_list = [para1]
        paragraphs = np.array(paragraph_list)
        clean_text = fetch_low_freq_words(paragraphs)
        self.assertTrue(len(clean_text) > 0)
        self.assertTrue(clean_text.__contains__("scoring"))


    def test_rm_stopwords_stem_lowfreq(self):
        """
        Tests if the rm_stopwords_stem_lowfreq function works as expected.
        """
        para1 = np.array(["""Tf idf stands for term frequency inverse document frequency and the tf idf weight is a weight often used in information retrieval and text mining This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus Variations of the tf idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document s relevance given a user query"""])
        compare_para = ['idf term frequenc invers document frequenc idf weight weight use inform retriev text mine thi weight statist measur use evalu import word document collect corpu the import increas proport number time word appear document offset frequenc word corpu variat idf weight scheme use search engin central tool score rank document relev given user queri']
        low_freq_words = ['stands', 'often']
        clean_text = rm_stopwords_stem_lowfreq(para1, low_freq_words)
        self.assertTrue(len(clean_text) > 0)
        self.assertEquals(clean_text, compare_para)
    

    def test_calc_freq_distr(self):
        """
        Tests if the calc_freq_distr function works as expected.
        """
        input_paragraphs = np.array(["""Tf idf stands for term frequency inverse document frequency and the tf idf weight is a weight often used in information retrieval and text mining This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus Variations of the tf idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document s relevance given a user query"""])
        word_corpus = np.array(['Tf-idf','The','they','This','Variations','appears','are','as','but','central']).transpose()
        freq_distribution = calc_freq_distr(input_paragraphs, word_corpus)
        self.assertTrue(len(freq_distribution) > 0)
        self.assertTrue(freq_distribution.__contains__(0))


    def test_calc_tf_idf(self):
        """
        Tests if the function calc_tf_idf works as expected.
        """
        tf = np.array([[0,1,2],[2,1,0]])
        idf = np.array([[1,2,1]])
        #tfidf = np.array([[0, 0.70710678, 0.70710678], [0.70710678, 0.70710678, 0]])
        result = calc_tf_idf(tf, idf)
        self.assertTrue(result.__contains__(0))


    def test_preprocess_data(self):
        """
        Tests if the preprocess_data function works as expected.
        """
        paragraphs = np.array(["""Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query."""])
        compare_text = np.array(['idf invers document idf weight weight often use mine weight use word document corpu word document word corpu idf often use document queri'], dtype='|S577')
        result = preprocess_data(paragraphs)
        self.assertEquals(result, compare_text)


    def test_compute_similarity(self):
        """
        Tests if the compute_similarity function works as expected.
        """
        paragraphs = np.array(['idf term invers document idf weight weight often use mine weight use import word document number time word appear document word idf often use document queri',
            'typic idf weight term frequenc aka number time word appear document divid number document term invers document frequenc idf number document divid number document term appear',
            'idf invers document frequenc import term comput term import term time term one comput follow']).transpose()
        similar_text_idx = [0]
        simiar_para_idx = compute_similarity(paragraphs, similar_text_idx)
        self.assertEqual(simiar_para_idx, [1])


if __name__ == '__main__':
    """
    Runs all the unit tests defined above.
    """
    unittest.main()
