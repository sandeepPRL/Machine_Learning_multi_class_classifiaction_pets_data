from sklearn.metrics.pairwise import cosine_similarity
class Cosine(object):
	def __init__(self, op1):
		self.op1=op1
		
	
	def cosinesimilarty(self):
	    cosine_mat = cosine_similarity(self.op1)
	    print('cosine sililarity is')
	    print(cosine_mat)
	    
	    # print(coasine_mat.shape)

	    return cosine_mat