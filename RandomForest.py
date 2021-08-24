import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split


def quistion_4_RandomForest(df, class1, class2, class3=None, class4=None):
    if class3 is not None and class4 is not None:
        df = df.loc[
             (df["class"] == class1) | (df["class"] == class2) | (df["class"] == class3) | (df["class"] == class4), :]
    else:
        df = df.loc[(df["class"] == class1) | (df["class"] == class2), :]
    df = df.replace(np.nan, 0)
    X = df[df.columns[:-2]]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_test = pd.DataFrame(y_test, columns=['class']).loc[:, "class"]
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    prediction = regressor.predict(X_test)

    print(accuracy_score(class_test, np.round(abs(prediction))))
    if class3 is not None and class4 is not None:
        y_unique = y_test.unique()
        mcm = multilabel_confusion_matrix(y_test, np.round(abs(prediction)), labels=y_unique)
        print(mcm)
    else:
        print(confusion_matrix(class_test, np.round(abs(prediction))))
    print(classification_report(class_test, np.round(abs(prediction))))
