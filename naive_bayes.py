from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class Naive_bayes:
    def __init__(self,X_train, y_train,X_test,y_test):

        N_classifier = OneVsRestClassifier(GaussianNB())

        N_classifier.fit(X_train, y_train)
        naive_pred = N_classifier.predict(X_test)
        NB_accuracy=metrics.accuracy_score(y_test,naive_pred)

        print("Naive bayes accuracy is..........")
        print(NB_accuracy)

        return
