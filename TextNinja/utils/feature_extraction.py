import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import math

class featureExtraction:

    """
        Class to perform feature extraction.

        Functions:

        TFIDFVector - Calculate TF-IDF vectors for the input paragraphs. (using sklearn)

        calc_freq_distr - Calculate the frequency distribution of each word in a paragraph.
                          Outputs a list of dictionaries for each paragraph.
                          Dictionary Keys: Paragraph_id, Frequency_distribution (Dictionary)
                          Frequency dictionary-Key: word, Value: word count in the paragraph

        calc_tf - Calculate and returns the TF score. This function takes frequency_distribution dictionary as input.

        calc_idf - Calculate and returns the IDf score. This function takes frequency_distribution dictionary as input.

        calc_tf_idf - Calculate and returns the TF-IDF score.

        para_similarity - Calculates the similarity of each paragraph.
                        Multiply the TF-IDF vectors of each paragraph with the TF-IDF vector of the paragraph with the centroid (placed at 0th position).
                        Input passed is a numpy array. (convert tf_idf_vectors into numpy array)

        compute_average - Calculate the new TF-IDF centroid.
    """

    def TFIDFVector(self, input_paragraph):
        vectorizer = TfidfVectorizer(max_df=0.90, min_df=0.10)  # ignores frequency lower than a threshold
        tfidf = vectorizer.fit_transform(input_paragraph)
        print(vectorizer.get_feature_names())
        return tfidf.todense()

    def calc_freq_distr(self, input_paragraphs):
        para_count = 0
        para_list = []
        for paragraph in input_paragraphs:
            para_count += 1
            word_count_dict = {}
            for word in paragraph.split(' '):
                if word in word_count_dict:
                    word_count_dict[word] += 1
                else:
                    word_count_dict[word] = 1
            para_dict = {'Paragraph_id': para_count, 'Freq_Dict': word_count_dict}
            para_list.append(para_dict)
        return para_list

    def calc_tf(self, doc_freq_dist):
        TF_scores = []
        for para in doc_freq_dist:
            para_id = para['Paragraph_id']
            paragraph = para['Freq_Dict']
            total_words_p = sum(paragraph.values())
            for word, count in paragraph.items():
                word_dict = {'Paragraph_id': para_id, 'word': word, 'TF': count / total_words_p}
                TF_scores.append(word_dict)
        return TF_scores

    def calc_idf(self, doc_freq_dist):
        IDF_scores = {}
        total_docs = len(doc_freq_dist)
        for para in doc_freq_dist:
            for word, count in para['Freq_Dict'].items():
                if word not in IDF_scores:
                    count = sum([word in paragraph['Freq_Dict'] for paragraph in doc_freq_dist])
                    IDF_scores[word] = math.log(total_docs / count)
        return IDF_scores

    def calc_tf_idf(self, TF_scores, IDF_scores):
        TFIDF = []
        for dict in TF_scores:
            TFIDF_scores = {
                'Paragraph_id': dict['Paragraph_id'],
                'word': dict['word'],
                'TFIDF': dict['TF'] * IDF_scores[dict['word']]
            }
            TFIDF.append(TFIDF_scores)
        print(TFIDF)

    def para_similarity(self, tf_idf_array):
        similar_text = []
        for i in range(1, tf_idf_array.shape[1]):
            if np.matmul(tf_idf_array[:, 0], tf_idf_array[:, i]) > 0.4:
                similar_text.append(i)
        return similar_text

    def compute_average(self, average_tfidf, avg_para_count, tfidf_vector, para_count):
        temp_count = avg_para_count + para_count
        average_tfidf = (average_tfidf * avg_para_count + tfidf_vector * para_count) / temp_count
        avg_para_count = temp_count
        return average_tfidf, avg_para_count
