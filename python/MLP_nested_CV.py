import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc


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
        for hidden1 in range(10,300,10):
            for hidden2 in range(10,300,10):
                b = []
                for state in range(10):
                    mlp = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2), solver='adam', random_state=1, max_iter=100000).fit(x[train],y[train])
                    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train], y[train])
                    scores = cross_val_score(mlp, x[train], y[train], cv=kfold, scoring='accuracy', n_jobs=-1)
                    b.append(scores.mean())
                c = np.mean(b)
                if c > g:
                    g = c
                    h[0] = hidden1
                    h[1] = hidden2
        n1.append(h[0])
        m1.append(h[1])
    n[i,] = n1
    m[i,] = m1

n = pd.DataFrame(n)
n.to_csv('./data/mlp hidden1.CSV')
m = pd.DataFrame(m)
m.to_csv('./data/mlp hidden2.CSV')
