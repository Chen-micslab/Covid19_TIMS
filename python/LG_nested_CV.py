import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

data1_1 = pd.read_csv('./data/15 Lipid.CSV', header=0)
data1 = np.array(data1_1)
np.random.seed(98)
permutation = np.random.permutation(data1[:,0].shape[0])
data1 = data1[permutation]
x = data1[:,1:]
y = data1[:,0]
from sklearn.preprocessing import StandardScaler
st = StandardScaler().fit(x)
x = st.transform(x)
c_np = np.zeros((20,5))
g_np = np.zeros((20,5))
acc_m = []
for i in range(20):
    print(i)
    skf=StratifiedKFold(n_splits=5,random_state=i,shuffle=True).split(x,y)
    c_m = []
    g_m = []
    for k,(train, test) in enumerate(skf):
        g = 0
        h = 0
        for j in range(1,50):
                b = []
                for state in range(10):
                    acd = LogisticRegression(C=j,max_iter=10000)
                    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train], y[train])
                    scores = cross_val_score(acd, x[train], y[train], cv=kfold, scoring='accuracy', n_jobs=10)
                    b.append(scores.mean())
                c = np.mean(b)
                if c > g:
                    g = c
                    h = j
        c_m.append(h)
    c_np[i,] = c_m
    skf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True).split(x, y)
    for k,(train, test) in enumerate(skf):
        acd = LogisticRegression(C=c_m[k],max_iter=10000).fit(x[train], y[train])
        acc = acd.score(x[test],y[test])
        acc_m.append(acc)
print(acc_m)
print(np.mean(acc_m))
print(c_np)
print(g_np)
acc_m = pd.DataFrame(acc_m)
c_np = pd.DataFrame(c_np)
c_np.to_csv('./data/LG C.CSV')
