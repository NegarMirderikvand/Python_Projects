import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
import datetime
import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/Marketc.csv')
df.columns
np.sum(df.isna())
df.describe()
df.dtypes
cancellations=df['InvoiceNo'].str.contains('C')
#len(re.findall('C',df['InvoiceNo'].iloc[220]))
df=df.loc[cancellations!=True]
df['InvoiceDate']=df['InvoiceDate'].agg(pd.Timestamp)
today=df['InvoiceDate'].iloc[0].now()
df['Days']=df['InvoiceDate'].agg(lambda x:(today-x).days)


Recency=df.groupby('CustomerID').Days.min()


Frequency=df.groupby('CustomerID').count().iloc[:,0]

df['Cost']=df['Quantity']*df['UnitPrice']
Monetary=df.groupby('CustomerID').sum().loc[:,'Cost']

X=pd.concat((pd.DataFrame(Recency),pd.DataFrame(Frequency),pd.DataFrame(Monetary)),axis=1)

#X=scale(X)

standardize=StandardScaler()
standardize.fit(X)
X=standardize.fit_transform(X)

X=pd.DataFrame(X)
X.columns=['R','F','M']
n_clusters=3
kmeans = KMeans(n_clusters=n_clusters).fit(X)

pd.DataFrame(kmeans.cluster_centers_,columns=X.columns)

Xorg=standardize.inverse_transform(X)
Xorg=pd.DataFrame(Xorg)
Xorg.columns=['R','F','M']
Xorg['clusters']=kmeans.labels_
Xorg.groupby('clusters').agg('mean')


fig=plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlim([-.1, 1])
labels=kmeans.labels_
average_silhouette=silhouette_score(X,labels)
sample_silhouette_values=silhouette_samples(X,labels)
y0=10
for i in range(n_clusters):
    silhouette_values=sample_silhouette_values[labels==i]
    silhouette_values.sort()
    y1=y0+silhouette_values.shape[0]
    color = cm.nipy_spectral(float(i) / n_clusters)
    #if i==0:
    ax1.fill_betweenx(np.arange(y0,y1),0, silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
    #else:
    #ax1.fill_betweenx(np.arange(y0,y1),0, silhouette_values,facecolor='red', edgecolor='red', alpha=0.7)   
    ax1.text(-0.05, y0+1*silhouette_values.shape[0],str(i))
    y0 =y1 +10  
    ax1.set_title("Silhouette plot")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Customers")
    ax1.axvline(x=average_silhouette, color="red", linestyle="--")
  