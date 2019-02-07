from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
class LR:
    def __init__(self,X_train, y_train,X_test,y_test):
        log_clf = OneVsRestClassifier(LogisticRegression())
        log_clf.fit(X_train, y_train)
        y_pred_class = log_clf.predict(X_test)

        logic_regression = metrics.accuracy_score(y_test,y_pred_class)
        print("logistic regression classifier score is .............")
        print(logic_regression)
        return