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


data1_1 = pd.read_csv('./data/15 Lipid.CSV',header=0)
data1 = np.array(data1_1)
np.random.seed(98)
permutation = np.random.permutation(data1[:,0].shape[0])
data1 = data1[permutation]
x = data1[:,1:]
y = data1[:,0]
from sklearn.preprocessing import StandardScaler
st = StandardScaler().fit(x)
x = st.transform(x)
pca = PCA(n_components=0.99)
pca.fit(x)
x = pca.transform(x)
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
        h = [0, 0]
        for j in [0.7,7,70,700,7000]:
            for l in [0.5,5,50,500,5000,50000,500000]:
                b = []
                for state in range(10):
                    acd = SVC(C=j, gamma=l / 10000).fit(x[train], y[train])
                    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train], y[train])
                    scores = cross_val_score(acd, x[train], y[train], cv=kfold, scoring='accuracy', n_jobs=10)
                    b.append(scores.mean())
                c = np.mean(b)
                if c > g:
                    g = c
                    h[0] = j
                    h[1] = l
        c_m.append(h[0])
        g_m.append(h[1])
    c_np[i,] = c_m
    g_np[i,] = g_m
    skf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True).split(x, y)
    for k,(train, test) in enumerate(skf):
        acd = SVC(C=c_m[k], gamma=g_m[k] / 10000).fit(x[train],y[train])
        acc = acd.score(x[test],y[test])
        acc_m.append(acc)
c_np = pd.DataFrame(c_np)
g_np = pd.DataFrame(g_np)
c_np.to_csv('./data/SVM C.CSV')
g_np.to_csv('./data/SVM gamma.CSV')
