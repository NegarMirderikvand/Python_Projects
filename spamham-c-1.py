import numpy as np;
import pandas as pd;
from nltk.corpus import stopwords;
import re
from nltk.stem import SnowballStemmer;
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier;
from sklearn.metrics import confusion_matrix;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
text=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/spam.csv',encoding='latin-1')


#text=pd.read_csv('',encoding='latin-1')
text.columns=['Category','Message']# Put some names on the columns
#Extract some variables (features) that might be useful for prediction
text['word_count'] = text['Message'].agg(lambda x: len(x.split(" ")))#Count the number of words in each message

#Get the number of characters in each message
text['char_count'] = text['Message'].agg(lambda x:len(x))

#Remove the stopwprds
stop = stopwords.words('english')

stop.append('u')
stop.append('ur')
stop.append('2')
stop.append('4')

#Change all the words to lowercase
text['Message']=text['Message'].agg(lambda x:x.lower())

#Number of stopwords used in the message can be an appropriate feature
text['stopwords'] = text['Message'].agg(lambda x: len([w for w in x.split() if w in stop]))

#Compare the features obtained so far for ham and spam messages
df2=text.groupby(text['Category'])
df2.agg('mean')


#Remove all charactes
'''
ex1='Hello. Can I ask you a question?! That will cost me 100$, but it is OK.'
Regular expression is a powerful package to manage strings.
\:means a special sequence. 
\d:retuns a matche where a string contains a digit. 
\s:returns a match where the string contains white space
re.findall('\s',ex1)
\w:returns all word and numeric characters 
re.findall('\w',ex1)
len(re.findall('\w',ex1))
+:returns all characters' occurances
re.findall('l+',ex1)
[hi?]:returns matches to each of the characters inside [] 
re.findall('[hi?]',ex1)
Combine some commands:
re.findall('\w\s',ex1)#Find word characters followed by a white space
re.findall('\d+\$',ex1)
[^hi]:return all characters except those followed by ^
re.findall('[^hi?]',ex1)
re.findall('[^hi?\s]',ex1)
re.findall('[\w\s]',ex1)
re.findall('[^\w\s]',ex1)
re.sub:finds a pattern and substitute it with a replacement
re.sub('will','might',ex1)
re.sub('[^\w\s]',"",ex1)
'''
text['Message'] = text['Message'].agg(lambda x:re.sub('[^\w\s]','',x))

#Stemming all words

stemmer = SnowballStemmer("english")
'''
h='Stemming is the process of reducing a word to its word stem or to the roots of words known as a lemma. Stemming is important in natural language understanding (NLU) and natural language processing (NLP).'

h=re.sub('[^\w\s]',"",h)#remove the characters
v=h.split()
type(v)
len(v)
stemmer.stem(v[0])
h1=(" ").join([stemmer.stem(x) for x in v])
'''
text['Message'] = text['Message'].agg(lambda x:(" ").join([stemmer.stem(w) for w in x.split()]))


spamtext=text[text['Category']=='spam']['Message']
hamtext=text[text['Category']=='ham']['Message']

#The method isalpha() checks whether the string consists of alphabetic characters only.
#Get all words used in spam and ham mesagges (those words which are not stopwords)
spam_no_stop =spamtext.agg(lambda x:' '.join([word for word in x.split() if word not in stop]))
hamw_no_stop =hamtext.agg(lambda x:' '.join([word for word in x.split() if word not in stop]))

#split the words again and count them across all the messages
spam_word_counts=spam_no_stop.str.split(expand=True).stack().value_counts()


ham_word_counts=hamw_no_stop.str.split(expand=True).stack().value_counts()

#Select those words frequently used across all messages
spamwords_usable=spam_word_counts[spam_word_counts>=20]
hamwords_usable=ham_word_counts[ham_word_counts>=20]

#Make some indicator variables to see whether the spam words are used in the messages or not

s1=set(spamwords_usable.index)
Union=s1.union(set(hamwords_usable.index))
Union=pd.Series(list(Union))

allfeatures=np.zeros((text.shape[0],Union.shape[0]))
for i in np.arange(Union.shape[0]):
 allfeatures[:,i]=text['Message'].agg(lambda x:len(re.findall('\s{}\s'.format(Union[i]),x)))


#Complete_data=pd.concat([text,pd.DataFrame(hamfeatures),pd.DataFrame(spamfeatures),pd.DataFrame(commonfeatures)],1)
Complete_data=pd.concat([text,pd.DataFrame(allfeatures)],1)


X=Complete_data.iloc[:,2:]
y=Complete_data['Category']
enc=LabelEncoder()
enc.fit(y)
y = enc.transform(y)
repeat=50
acc_lasso_ham=np.empty(repeat)
acc_lasso_spam=np.empty(repeat)
acc_ridge_ham=np.empty(repeat)
acc_ridge_spam=np.empty(repeat)
acc_elnet_ham=np.empty(repeat)
acc_elnet_spam=np.empty(repeat)

for i in range(repeat):
    print(i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lassologreg = LogisticRegression(C=15,penalty="l1",solver="liblinear")
    ridgelogreg = LogisticRegression(C=15,penalty="l2",solver="liblinear")
    #c same as t in slides
    elaslogreg=SGDClassifier(loss='log',penalty='elasticnet',alpha=0.0001,l1_ratio=1,tol=0.001)
    #l1_ratio equivalent to alpha in slides    
    #alpha same as lambda in slides
    lassologreg.fit(X_train,y_train)
    ridgelogreg.fit(X_train,y_train)
    #logreg.fit(X_train,y_train)
    elaslogreg.fit(X_train,y_train)
    
    lasso=confusion_matrix(y_test,lassologreg.predict(X_test))
    ridge=confusion_matrix(y_test,ridgelogreg.predict(X_test))
    elnet=confusion_matrix(y_test,elaslogreg.predict(X_test))
    
    
    acc_lasso_ham[i]=lasso[0,0]/sum(lasso[0,:])
    acc_lasso_spam[i]=lasso[1,1]/sum(lasso[1,:])
    acc_ridge_ham[i]=ridge[0,0]/sum(ridge[0,:])
    acc_ridge_spam[i]=ridge[1,1]/sum(ridge[1,:])
    acc_elnet_ham[i]=elnet[0,0]/sum(elnet[0,:])
    acc_elnet_spam[i]=elnet[1,1]/sum(elnet[1,:])

print('GLM Lasso Ham','\n',np.mean(acc_lasso_ham))
print('GLM Lasso Spam','\n',np.mean(acc_lasso_spam))
print('GLM Ridge Ham','\n',np.mean(acc_ridge_ham))
print('GLM Ridge Spam','\n',np.mean(acc_ridge_spam))
print('GLM Net Ham','\n',np.mean(acc_elnet_ham))
print('GLM Net Spam','\n',np.mean(acc_elnet_spam))
