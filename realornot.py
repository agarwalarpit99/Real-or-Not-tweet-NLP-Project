import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

lemma=WordNetLemmatizer()
train_data=pd.read_csv('/home/arpit/Downloads/train.csv')
test_data=pd.read_csv('/home/arpit/Downloads/test.csv')
train_data.drop(['keyword','location'],inplace=True,axis=1)
test_data.drop(['keyword','location'],inplace=True,axis=1)

test_data.info()

len(train_data)

train_corpus=[]
test_corpus=[]

for  i in range(len(train_data)):
    review=re.sub('[^a-zA-Z]',' ',train_data['text'][i])
    review=review.lower()
    review=review.split()
    review=[lemma.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    train_corpus.append(review)
    
for  j in range(len(test_data)):
    review1=re.sub('[^a-zA-Z]',' ',test_data['text'][j])
    review1=review1.lower()
    review1=review1.split()
    review1=[lemma.lemmatize(word) for word in review1 if word not in stopwords.words('english')]
    review1=' '.join(review1)
    test_corpus.append(review1)   
    
 from sklearn.feature_extraction.text import TfidfVectorizer
 
 tf=TfidfVectorizer(max_features=4000)
 
 Xtrain=tf.fit_transform(train_corpus).toarray()
 
 Ytrain=train_data['target']
 
 Xtest=tf.fit_transform(test_corpus).toarray()
 
 from sklearn.naive_bayes import MultinomialNB
 
 model=MultinomialNB().fit(Xtrain,Ytrain)
 
 predictions=model.predict(Xtest)
 
 from sklearn.linear_model import LogisticRegression
 
 lr=LogisticRegression().fit(Xtrain,Ytrain)
 
 predictions_logistic=lr.predict(Xtest)
    
    
    
submit=pd.DataFrame()
submit['id']=test_data['id']   
submit['target']=predictions_logistic 

submit.to_csv('realornotsubmission.csv',index=False)

from sklearn.svm import SVC
svm_model=SVC().fit(Xtrain,Ytrain)

svm_predict=svm_model.predict(Xtest)

submit2=pd.DataFrame()
submit2['id']=test_data['id']   
submit2['target']=svm_predict 

submit2.to_csv('realornotsubmission2.csv',index=False)
    
    
    
    
    
    
    
    