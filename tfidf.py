from sklearn.feature_extraction.text import TfidfVectorizer
class Tfidf():
    def __init__(self,concerns):
        # self.uniq = uniq
        self.concerns = concerns

    def tfidf(self):
        Tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        # print(tfidf.vocabulary_)
        # tfidf.fit(data)
        # a = vectorizer.transform(data).toarray()
        # print(a)
        # tokenize and build vocab
        tf_idf_mat = Tfidf.fit_transform(self.concerns)
        print("shape of tfidf matrix.....")
        print(tf_idf_mat)
        return tf_idf_mat

# class Cosine(object):
#     def __init__(self, op1):
#         self.op1=op1
        
    
#     def cosinesimilarty(self):
#         coasine_mat = cosine_similarity(self.op1)
#         print("cosin_matrix is ")
#         print(coasine_mat)
#         print(coasine_mat.shape)

#         return coasine_mat