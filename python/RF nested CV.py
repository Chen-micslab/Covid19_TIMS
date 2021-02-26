from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

data1_1 = pd.read_csv('./data/15 Lipid.CSV',header=0)
data1 = np.array(data1_1)

np.random.seed(98)
permutation = np.random.permutation(data1[:,0].shape[0])
data1 = data1[permutation]
x = data1[:,1:]
y = data1[:,0]

n = np.zeros((20,5))
m = np.zeros((20,5))
acc_m = []
for i in range(20):
    print(i)
    skf=StratifiedKFold(n_splits=5,random_state=i,shuffle=True).split(x,y)
    n1 = []
    m1 = []
    for k,(train, test) in enumerate(skf):
        g = 0
        h = [0,0]
        for n_tree in [400,600,800,1000]:
            for m_dep in [2,4,6,8,10,13,15,18,20,30]:
                b = []
                for state in range(10):
                    RF = RandomForestClassifier(n_estimators=n_tree,max_depth=m_dep,random_state=11).fit(x[train],y[train])
                    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train], y[train])
                    scores = cross_val_score(RF, x[train], y[train], cv=kfold, scoring='accuracy', n_jobs=-1)
                    b.append(scores.mean())
                c = np.mean(b)
                if c > g:
                    g = c
                    h[0] = n_tree
                    h[1] = m_dep
        n1.append(h[0])
        m1.append(h[1])
    n[i,] = n1
    m[i,] = m1
    skf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True).split(x, y)
    for k, (train, test) in enumerate(skf):
        RF = RandomForestClassifier(n_estimators=n1[k], max_depth=m1[k], random_state=11).fit(x[train], y[train])
        acc = RF.score(x[test], y[test])
        acc_m.append(acc)
n = pd.DataFrame(n)
n.to_csv('./data/RF n trees.CSV')
m = pd.DataFrame(m)
m.to_csv('./data/RF max_depth.CSV')