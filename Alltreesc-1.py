import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

df = pd.read_csv('')


y=df.iloc[:,6]

enc=LabelEncoder()
enc.fit(y)
y=enc.fit_transform(y)
X=df.iloc[:,:6]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
dectree = tree.DecisionTreeClassifier(max_depth=5)
bag = BaggingClassifier(n_estimators=100,oob_score=True)
rf = RandomForestClassifier(n_estimators=1000,oob_score=True,max_features='auto')
boost = AdaBoostClassifier(n_estimators=1000)

dectree.fit(X_train,y_train)
bag.fit(X_train,y_train)
rf.fit(X_train,y_train)
boost.fit(X_train,y_train)
print('Tree','Bagging','Boosting','Random Forrest\n',np.round_(dectree.score(X_test,y_test),2),np.round_(bag.score(X_test,y_test),2),np.round_(boost.score(X_test,y_test),2),np.round_(rf.score(X_test,y_test),2),'\nTraining error\n',np.round_(dectree.score(X_train,y_train),2),np.round_(bag.score(X_train,y_train),2),np.round_(boost.score(X_train,y_train),2),np.round_(rf.score(X_train,y_train),2))
print('RF cross-val error:\n',1-rf.oob_score_)
print('Bagging cross-val error:\n',1-bag.oob_score_)
print(pd.DataFrame(rf.feature_importances_,index=df.columns[:-1],columns=['Mean Decrease in Gini']))

