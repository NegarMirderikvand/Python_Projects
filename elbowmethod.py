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


X.columns=['R','F','M']
wss=[]
for n in range(1,11):
 kmeans = KMeans(n_clusters=n).fit(X)
 wss.append(kmeans.inertia_)
plt.plot(range(1,11),wss,c='r',marker='o',alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Within Sum of Squares Distance')
