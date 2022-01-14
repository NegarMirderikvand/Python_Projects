import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
df=pd.read_csv('/LAX-air-pollution.csv')
X=df
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
W=p.components_.T
y=p.fit_transform(scaledX)
yhat=X.dot(W)
plt.figure(1)
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="blue",marker='o',alpha=0.5)
plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

names=df.index
#Annotate the day number on PC scores data points
for i, txt in enumerate(names):
    plt.annotate('day{}'.format(txt), (yhat.iloc[i,0]*1.02, yhat.iloc[i,1]*1.02))

pd.DataFrame(W[:,:3],index=df.columns,columns=['PC1','PC2','PC3'])
pd.DataFrame(p.explained_variance_ratio_,index=np.arange(7)+1,columns=['Explained Variability'])

#scree plot
plt.figure(2)
plt.bar(np.arange(1,8),p.explained_variance_,color="blue",edgecolor="red")
plt.xlabel('Number of Components')
plt.ylabel('Explained Variability')

#This gives you the biplot for the first two PC scores
xs=yhat.iloc[:,0]#xs represents PC score 1
ys=yhat.iloc[:,1]#ys represents PC score 2
plt.figure(3)
#plot the arrows associated with variables
for i in range(len(W[:,0])):
# arrows project features (ie columns from csv) as vectors onto PC axes
#here we multiply W by abs(max(xs)) and abs(max(ys)) to scale the biplots
    plt.arrow(np.mean(xs), np.mean(ys), W[i,0]*abs(max(xs)), W[i,1]*abs(max(ys)),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(W[i,0]*abs(max(xs))+np.mean(xs), +np.mean(ys)+W[i,1]*abs(max(ys)),
             list(df.columns.values)[i], color='r')
#'''
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="blue",marker='o',alpha=0.5)
#plot the points and place the texts (only day numbers) on the points. The number 1.025 
#means that locate day numbers about 2.5% far from the points
for i in range(len(xs)):
    plt.plot(xs[i], ys[i], 'bo')
    plt.text(xs[i]*1.1, ys[i]*1.1, list(df.index)[i], color='b')
#increase the limit to make more space for biplots (arrows)
plt.xlim(min(xs)-5,max(xs)+20)

plt.xlabel('PC Scores1')
plt.ylabel('PC Scores2')

plt.show()

