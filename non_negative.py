from sklearn.decomposition import NMF
class NMF():
	def __init__(self,op1):
		self.op1 = op1
	def non_negative_matrices(self):
	    NMF_model = NMF(n_components=None, init='random', random_state=0)
	    nmf_mat = NMF_model.fit_transform(self.op1)
	    # print("nmf matrix")
	    # print(nmf_mat)
	    
	    return nmf_mat
