# /nlp/utils.py

import time
from nlp.preprocess import remove_special_chars, rm_stopwords_stem_lowfreq, fetch_low_freq_words
from nlp.tfidf_feature_extraction import calc_freq_distr, calc_idf, calc_tf, calc_tf_idf, compute_para_similarity, compute_similarity_centroid
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

EMPTY_INP_SIMILARITY = 'Invalid inputs to the function compute_similarity.'


"""
    This function is a wrapper which calls the function to pre-process the data.
"""
def preprocess_data(paragraph_list):
    
    # try:
    if True:
        time_in = time.time()
        paragraphs = np.array(paragraph_list)

        low_freq_words = fetch_low_freq_words(paragraphs)

        for p in range(paragraphs.shape[0]):
            if len(paragraphs[p]) == 0 or paragraphs[p] is None:
                continue;
            paragraphs[p] = remove_special_chars(paragraphs[p])

        clean_text = rm_stopwords_stem_lowfreq(paragraphs, low_freq_words)
        
        print('Time taken to pre-process the data: ', time.time() - time_in)
        return clean_text
    
    # except:
        # print('Error while pre-processing the data')
        # sys.exit("ohh no!! errors!!")


"""
This function returns the index of the similar paragraphs.
"""
def compute_similarity(paragraph_list, similar_text_idx):
    if len(paragraph_list) == 0 or len(similar_text_idx) == 0:
        raise ValueError(EMPTY_INP_SIMILARITY)
    
    # try:
    if True:
        clean_text = preprocess_data(paragraph_list)

        word_corpus = np.hstack(np.char.split(clean_text))
    
        # Calculate Word Frequency in the document
        freq_distribution = calc_freq_distr(clean_text, word_corpus)

        # Calculate Inverse Document Frequency
        word_idf = calc_idf(freq_distribution)
        
        # Calculate TF term
        word_tf = calc_tf(freq_distribution)

        # Calculate TF-IDF 
        word_tf_idf = calc_tf_idf(word_tf, word_idf)

        if len(similar_text_idx) > 1:
            similarity_centroid = compute_similarity_centroid(word_tf_idf, similar_text_idx)
        else:
            similarity_centroid = word_tf_idf[similar_text_idx[0]] 

        # Check Similarity 
        similar_para_idx = compute_para_similarity(word_tf_idf, similar_text_idx, similarity_centroid)
        
        return similar_para_idx
    # except:
        # print('Error while finding similar paragraphs.')
        # sys.exit("ohh no!! errors!!")
