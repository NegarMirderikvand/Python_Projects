import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
df=pd.read_csv('LAX-air-pollution.csv')
X=df
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
#compute the explained variability by the first two PC scores
orgexplained=p.explained_variance_ratio_.cumsum()[1]


explained=np.zeros(1000)
for i in np.arange(1000):
    #perform random sampling with replacement and generate new matrices
    s=np.random.choice(X.shape[0],X.shape[0],replace=True)
    Xnew=scaledX[s,:]
    #apply PCA on the new matrices and compute the explained variability
    p=PCA()
    p.fit(Xnew)
    explained[i]=p.explained_variance_ratio_.cumsum()[1]

#plot the histogram for the explained variability and draw the 2.5% and 97.5% quantiles
plt.hist(explained,bins=60,color=(1,0.94,0.86),edgecolor=(0.54,0.51,0.47))
plt.axvline(np.quantile(explained,0.975),color=(0.46,0.93,0))
plt.axvline(np.quantile(explained,0.025),color=(0.46,0.93,0))
plt.axvline(orgexplained,color='red')
plt.show()
