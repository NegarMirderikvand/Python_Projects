import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#df=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/LAX-air-pollution.csv')
df=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/men-track-records.csv')
X=df.iloc[:,:-1]
#X=df
#X=scale(X)
p=PCA()
p.fit(X)
W=p.components_.T
#Get the PC scores based on the centered X 
y=p.fit_transform(X)

#Compute the PC scores based on the original values of X (just for easier interpretation)
yhat=X.dot(W)
plt.figure(1)
#Get the scatter plot of the first two PC scores
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="red",marker='o',alpha=0.5)


plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

#Put the name of the contries on the plotted datapoints (this is called annotation)
names=df.Country
names=names.agg(lambda x: x[:5])
for i, txt in enumerate(names):
    plt.annotate(txt, (yhat.iloc[i,0], yhat.iloc[i,1]))

#Get the first three columns of the matrix of loadings 
pd.DataFrame(W[:,:3],index=df.columns[:-1],columns=['PC1','PC2','PC3'])
#Compute the explained variability by the PC scores
pd.DataFrame(p.explained_variance_ratio_,index=np.arange(8)+1,columns=['Explained Variability'])
#Get the scree plot
plt.figure(2)
plt.bar(np.arange(1,9),p.explained_variance_,color="blue",edgecolor="Red")
