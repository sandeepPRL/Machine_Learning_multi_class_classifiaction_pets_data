import os
from newprog import *
from Svm import SVM

'''

from logistic_regression import *
lr=LR(X_train, y_train,X_test,y_test)
print('logistic_regression',lr)
from naive_bayes import *
NB = Naive_bayes(X_train, y_train,X_test,y_test)
print('naive_bayes',NB)
from Decison_tree import *
DT = D_Tree(X_train, y_train,X_test,y_test)
print('Decison_tree',DT)
'''

print "test"
playground = ModelPlayground()
load=playground.preprocessing(is_tfidf=True, is_cosinesimilarty=True, is_Nmf=False)
print load
loadset = playground.get_labels()
z=load
# z= z.reshape(z.shape[1:])
y=loadset

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(z, y, test_size = .3)

S_vm=SVM(X_train, y_train,X_test,y_test)
Svm_score = S_vm.result()
print('SVM',Svm_score)

from logistic_regression import *
lr=LR(X_train, y_train,X_test,y_test)
print('logistic regression',lr)

from naive_bayes import *
NB = Naive_bayes(X_train, y_train,X_test,y_test)
print('naive_bayes',NB)

from Decison_tree import *
DT = D_Tree(X_train, y_train,X_test,y_test)
print('Decison_tree',DT)


#playground.Svm(X_train, y_train,X_test, y_test)
#X_train = np.array(X_train)

# print(X_train.shape)
# # X_test = np.array(X_test)

# # print(X_test.shape)
# # Y = np.array(y_train)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# ranForestTrain = RandomForestClassifier().fit(X_train, y_train).predict(X_train)
# ranForestText = RandomForestClassifier().fit(X_train, y_train).predict(X_test)

# YTest = np.array(y_test)
# from sklearn.metrics import f1_score
# score_test = f1_score(YTest, ranForestText, average='micro')
# score_train = f1_score(y_train, ranForestTrain, average='micro')
# #
# #
# print('print score..train....')
# print(score_train)

# print('print score..test....')
# print(score_test)
