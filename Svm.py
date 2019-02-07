from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

class SVM():
	def __init__(self,X_train, y_train,X_test,y_test):
	    self.X_train = X_train
	    self.y_train = y_train
	    self.X_test = X_test
	    self.y_test= y_test
        
	def fit(self):
		clf = OneVsRestClassifier(svm.SVC(C=1.3,kernel='linear'))
		self.FIT=clf.fit(self.X_train, self.y_train)
        
	def prediction(self):
	    self.y_pred = self.FIT.predict(self.X_test)
        
	def get_accuracy(self):
	    svm_accuracy=metrics.accuracy_score(self.y_test,self.y_pred)

	    print("svm classifier score is .............")
	    print(svm_accuracy)
	    return svm_accuracy

	def result(self):
		self.fit()
		self.prediction()
		return self.get_accuracy()
