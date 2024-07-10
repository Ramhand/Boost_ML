import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('./data/raw/wbm.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
    data = pd.read_csv(data)
    data.drop_duplicates(inplace=True)
    with open('./data/raw/wbm.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    print(data.head(10))

x = data.drop(columns='Outcome')
y = data['Outcome']

corr = data.corr()
refined_x = corr.loc[corr['Outcome'] > .2]
refined_x = data[refined_x.index.to_list()]
refined_x.drop(columns='Outcome', inplace=True)

xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.2, random_state=42)
rxtr, rxte, rytr, ryte = train_test_split(refined_x, y, train_size=0.2, random_state=42)

dic = {XGBClassifier: [[xtr, xte], [rxtr, rxte]], GradientBoostingClassifier: [[xtr, xte], [rxtr, rxte]]}
results = {}
maxim = 0

for i in range(len(dic.keys())):
    model = list(dic.keys())[i]()
    for j in dic[list(dic.keys())[i]]:
        if len(j[0].columns.to_list()) == 8:
            name = 'Regular Data'
        else:
            name = 'Refined Data'
        model.fit(j[0], ytr)
        pred = model.predict(j[1])
        acc = accuracy_score(yte, pred)
        results[model] = acc
        if acc > maxim:
            maxim = acc
        print(f'Model {i + 1} Base Accuracy ({name}):\t{acc}')

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 25],
            'min_samples_leaf': [1, 2, 4, 8],
        }
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid.fit(j[0], ytr)
        print(
            f'Grid Search Model {i + 1} ({name}):\n\tModel {i + 1} Best Parameters: {grid.best_params_}\n\tModel {i + 1} Best Accuracy: {grid.best_score_}')
        mod = grid.best_estimator_
        pred = mod.predict(j[1])
        acc_check = accuracy_score(pred, yte)
        if acc_check > maxim:
            maxim = acc_check
        results[mod] = acc_check
        cm = confusion_matrix(pred, yte)
        print(f'Model {i + 1} Best Estimator Prediction Accuracy: {acc_check}')
        sns.heatmap(cm, annot=True, cbar=True, fmt='.2f')
        plt.show()

for k, v in results.items():
    if v == maxim:
        best = k

with open('./models/best_model.dat', 'wb') as file:
    pickle.dump(best, file)

