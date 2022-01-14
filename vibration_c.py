import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.ar_model import AR, ARResults
from statsmodels.tsa import stattools
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.stats.diagnostic import acorr_ljungbox



#import ruptures as rpt
d=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/vibration1.csv',header=None)

signal=d.iloc[1000:2000,0]
signal.index=np.arange(1000)

plt.figure()
plt.plot(signal[:128])


fitar=AR(signal)
fittedar=fitar.fit(1,trend='c')
acorr_ljungbox(fittedar.resid)
plt.figure()
plot_acf(fittedar.resid,lags=20)
plt.figure()
plt.scatter(signal[1:],fittedar.fittedvalues,marker='o')

'''
plt.plot(signal[11:25])

np.corrcoef(signal[1:],signal[0:-1])
np.corrcoef(signal[5:],signal[0:-5])
np.corrcoef(signal[6:],signal[0:-6])
'''


fitar30=AR(signal)
fitted30=fitar30.fit(30)
fitted30.predict(start=1000,end=1000)#predict 1001
d.iloc[2000]#true value

fitar100=AR(signal)
fitted100=fitar100.fit(100)
acorr_ljungbox(fitted100.resid)
plt.figure()
plot_acf(fitted100.resid)
plt.figure()
plt.scatter(signal[100:],fitted100.fittedvalues,marker='o')



#ARMA Models
fitarma=ARMA(signal,(3,2))
fittedarma=fitarma.fit(solver='newton',disp=False)
#fittedarma=fitarma.fit()
plt.figure()
plot_acf(fittedarma.resid)
plt.figure()
plt.scatter(signal,fittedarma.fittedvalues,marker='o')

#Crossvalidation
i=0
test_size=20
msear=np.empty(test_size)
msep=np.empty(test_size)
msearma=np.empty(test_size)
X= signal.iloc[:720]
predp=np.empty(test_size)
predar=np.empty(test_size)
predarma=np.empty(test_size)
p=10
n=2;m=1
for i in range(test_size):
    print(i) 
    fitar=AR(X[:(700+i)])
    fittedar=fitar.fit(1)
    j=700+i
    v=fittedar.predict(start=j,end=j)
    predar[i]=v
    
    fitarp=AR(X[:(700+i)])
    fittedp=fitarp.fit(p)
    vp=fittedp.predict(start=j,end=j)
    predp[i]=vp
    
    fitarma=ARMA(X[:(700+i)],(n,m))
    fittedarma=fitarma.fit(solver='newton',disp=False)
    varm=fittedarma.predict(start=j,end=j)
    predarma[i]=varm
    
msear=np.mean((X[700:720]-predar)**2)
msep=np.mean((X[700:720]-predp)**2)
msearma=np.mean((X[700:720]-predarma)**2)
        







