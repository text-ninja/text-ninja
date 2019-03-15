# /nlp/tfidf_feature_extraction.py

import numpy as np

INVALID_INP_TFIDF = 'Error in TF-IDF Calculation. Invalid Input arrays.'
INVALID_INP_SIMILARITY = 'Error while computing compute paragraph similarity. Invalid inputs.'
EMPTY_INP_FREQ = 'Error while calculating frequency distribution of words. Invalid inputs.'
INVALID_INP_IDF = 'Error while calculating IDF. Invalid inputs.'
INVALID_INP_TF = 'Error while calculating the Term Frequency. Invalid inputs.'
INVALID_INP_CENTRE = 'Error while computing centroid. Invalid inputs.'

"""
Calculate the frequency distribution of each word in a paragraph.
Outputs an array of the frequency distribution for each paragraph.
"""
def calc_freq_distr(input_paragraphs, word_corpus):
    
    if input_paragraphs.size == 0 or word_corpus.size == 0:
        raise ValueError(EMPTY_INP_FREQ)

    # try:
    if True:
        freq_distribution = np.zeros((len(input_paragraphs), len(word_corpus)))
        
        for p in range(len(input_paragraphs)):
            paragraph = input_paragraphs[p].split()
            for w in range(len(word_corpus)):
                freq_distribution[p,w] = paragraph.count(word_corpus[w])
        return freq_distribution

    # except Exception as e:
        # print('Error: ', e)
        # print('Error while calculating frequency distribution of words.')


"""
calc_idf - Calculate and returns the Inverse Document Frequency score. This function takes frequency_distribution dictionary as input.

IDF definition:
    IDF(t) = log(N+1/df(t)+1) + 1
    where t is each word in the word corpus (feature)
    N is the number of paragraphs in the document
    df(t) is the count of documents in which the word appears. 
    
    An extra term 1 has been added to numerator and denominator to avoid divide by zero error. 
    It is equivalent to adding an extra paragraph which contains every word exactly once.
"""
def calc_idf(freq_distribution):
    
    if freq_distribution.size == 0:
        raise ValueError(INVALID_INP_IDF)

    # try:
    if True:
        n_paragraphs, n_words = freq_distribution.shape
        word_distribution = np.array([np.count_nonzero(freq_distribution[:,w]) for w in range(n_words)]).reshape(n_words, 1)
        doc_count = np.zeros((word_distribution.shape))  + float(n_paragraphs)
        idf = np.log(np.divide(1+doc_count,1+word_distribution)).transpose() + 1
        return idf

    # except Exception as e:
        # print('Error: ', e)
        # print('Error while computing IDF.')


"""
calc_tf - Calculate and returns the Term Frequency score. This function takes frequency_distribution array as input.
TF formula:
    TF(t) = Count of each word in the paragraph / Total number of words in the paragraph
"""
def calc_tf(freq_distribution):
    if freq_distribution.size == 0:
        raise ValueError(INVALID_INP_TF)
    # try:    
    if True:
        word_count = np.repeat(np.sum(freq_distribution, axis = 1).reshape(freq_distribution.shape[0], 1), repeats = freq_distribution.shape[1], axis = 1)
        tf = np.divide(freq_distribution, word_count)
        return tf

    # except Exception as e:
        # print('Error: ', e)
        # print('Error while calculating Term Frequency.')


"""
calc_tf_idf - Calculate TF-TDF of each word in the document.
Inputs:
1. Array of Term frequency (TF) of each paragraph in the document
2. IDF array of document.
Output: 
Returns a TF-IDF array.
"""
def calc_tf_idf(tf, idf):
    
    if tf.size == 0 or idf.size == 0:
        raise ValueError(INVALID_INP_TFIDF)

    # try:
    if True:    
        tf_idf = np.multiply(tf, idf)
        norm = np.linalg.norm(tf_idf, axis = 1).reshape(tf.shape[0], 1)
        return tf_idf/norm

    # except Exception as e:
        # print('Error: ', e)
        # print('Error while calculating TF-IDF.')


"""
compute_para_similarity - Calculates the similarity of each paragraph.
Multiply the TF-IDF vectors of each paragraph with the centroid of the TF-IDF 
vectors of the paragraph with which you wish to compute the similarity.
Input:  array of TF-IDF values of paragraph. 
Output: Similar paragraphs
"""
def compute_para_similarity(tf_idf_array, similar_text, similarity_centroid):
    
    if tf_idf_array.size == 0 or similarity_centroid.size == 0 or len(similar_text) == 0:
        raise ValueError(INVALID_INP_SIMILARITY)
    # try:
    if True:
        similar_para_idx = []
        for i in range(1, tf_idf_array.shape[0]):
            if i not in similar_text and np.matmul(similarity_centroid, tf_idf_array[i, :]) > 0.35:  
                similar_para_idx.append(i)
        return similar_para_idx

    # except Exception as e:
        # print('Error: ', e)
        # print('Error while computing paragraph similarity.')


def compute_similarity_centroid(tf_idf_array, similar_text):
    if tf_idf_array.size == 0 or len(similar_text) == 0:
        raise ValueError(INVALID_INP_CENTRE)
    # try:
    if True:
        similarity_centroid = np.sum(tf_idf_array[similar_text], axis = 0)/len(similar_text)
        return similarity_centroid
    # except Exception as e:
        # print('Error: ', e)
        # print('Error while computing centroid.')
