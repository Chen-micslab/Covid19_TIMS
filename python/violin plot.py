import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./data/Lipid CV.CSV',header=0)
fig1, ax1 = plt.subplots()
font1 =  {'family' : 'Arial',
'weight' : 'normal',
'size' : 22,
}
font2 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 18,}
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
sns.set(style='whitegrid', color_codes=True)
sns.violinplot( y='cv', data=data,linewidth=2,color="mediumturquoise")
plt.xlabel('lipidomic',font1)
plt.ylabel('Relative Standard Deviation',font1)
plt.yticks(fontproperties = 'Arial', size = 18)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 200
plt.show()