from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import itertools

data1_1 = pd.read_csv('./data/all compounds.CSV',header=0)
data1 = np.array(data1_1)
x = data1[:,1:]
y = data1[:,0]
m1 = []
for i in range(1000):
    print(i)
    np.random.seed(i)
    col_rand_array = np.arange(x.shape[1])
    np.random.shuffle(col_rand_array)
    x1 = x[:,col_rand_array[0:15]]
    m = []
    for r in range(20):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=r).split(x1, y)
        RF = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=11)
        scores = cross_val_score(RF, x1, y, cv=kfold, scoring='accuracy', n_jobs=-1)
        m.append(scores.mean())
    m1.append(np.mean(m))
print(m1)
m1 = pd.DataFrame(m1)
m1.to_csv('./data/random selected 15 feature acc.CSV')
