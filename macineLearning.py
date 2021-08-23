import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def quistion_4_desicionTree():
    global df
    df = df.loc[(df["class"] == 12) | (df["class"] == 15), :]
    df = df.replace(np.nan, 0)
    X = df[df.columns[:-2]]
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_test)
    class_test = pd.DataFrame(y_test, columns=['class']).loc[:, "class"]
    dtc = DecisionTreeClassifier(max_depth=2)
    dtc.fit(X_train, y_train)
    prediction = dtc.predict(X_test)

    # regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    # regressor.fit(X_train, y_train)
    # prediction = regressor.predict(X_test)

    print(confusion_matrix(prediction, class_test))
    score = accuracy_score(y_test, prediction)
    print(score)

# 3 and 9 got 83% a random forest
def quistion_4_RandomForest():
    global df
    df = df.loc[(df["class"] == 3) | (df["class"] == 9), :]
    df = df.replace(np.nan, 0)
    X = df[df.columns[:-2]]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class_test = pd.DataFrame(y_test, columns=['class']).loc[:, "class"]
    regressor = RandomForestRegressor()
    # param_grid = {
    #     'n_estimators': [200, 500],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth': [4, 5, 6, 7, 8]
    # }
    # CV_rfc = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)
    regressor.fit(X_train, y_train)
    prediction = regressor.predict(X_test)

    print(accuracy_score(class_test, np.round(abs(prediction))))
    print(confusion_matrix(class_test, np.round(abs(prediction))))
    # print(classification_report(class_test, np.round(abs(prediction))))