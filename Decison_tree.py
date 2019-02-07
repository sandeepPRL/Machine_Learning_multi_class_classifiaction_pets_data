
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

class D_Tree:
    def __init__(self,X_train, y_train,X_test,y_test):

            D_tree = DecisionTreeClassifier(criterion='gini', random_state=0, splitter='best')
            D_tree.fit(X_train, y_train)
            decison_tree_pred = D_tree.predict(X_test)

            D_score=mean_squared_error(y_test,decison_tree_pred)
            print("decison tree score is")
            print(D_score)
            return
