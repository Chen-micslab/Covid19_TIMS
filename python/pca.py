import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd

from matplotlib.pyplot import MultipleLocator

data = pd.read_csv('./data/pca information.CSV',header=0)
X = data.iloc[:, 1:]
y = data.iloc[:,0]
print(X.shape)
color = []
for i in range(len(data)):
        color.append(data.iloc[i,0])
from sklearn.preprocessing import StandardScaler
st = StandardScaler().fit(X)
X = st.transform(X)
pca = PCA(n_components=0.99)
X_q = pca.fit(X).transform(X)
ratio = pca.explained_variance_ratio_
print(ratio[1])
print(X_q.shape)
fig1, ax1 = plt.subplots()
classes = ['Asymptomatic','QC','Healthy']
font1 =  {'family' : 'Arial',
'weight' : 'normal',
'size' : 22,
}
font2 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 18,}
a = plt.scatter(X_q[:, 0], X_q[:, 1], c=color,cmap='Spectral', s=40)
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.yticks(fontproperties = 'Arial', size = 20)
plt.xticks(fontproperties = 'Arial', size = 20)
x_major_locator=MultipleLocator(20)
y_major_locator=MultipleLocator(15)
ax1.xaxis.set_major_locator(x_major_locator)
ax1.yaxis.set_major_locator(y_major_locator)

ax1.set(ylim=[-30, 30])
ax1.set(xlim=[-30, 70])
print(a.legend_elements()[0])
plt.legend(handles = a.legend_elements()[0], labels=classes,prop = font2,bbox_to_anchor=(0.7, 0.8), loc=3, borderaxespad=0,handletextpad=0.05)
plt.gca().set_aspect('equal', 'datalim')
plt.rcParams['figure.figsize'] = (8,8)
plt.xlabel('Dim1({0:0.1f}%)'.format(100*ratio[0]),font1)
plt.ylabel('Dim2({0:0.1f}%)'.format(100*ratio[1]),font1)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 300
plt.show()