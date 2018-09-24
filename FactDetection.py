import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
id=[]
dataset={}
nonfactual=[]
factual=[]
facts=[]
lem=WordNetLemmatizer()
stop=set(stopwords.words('english'))
stop.update(['#','.',':','http',"...",'@','!','&','-',',',';','$','/','\'','|'])
vec = TfidfVectorizer(stop_words=stop)
def preprocess(tweet):
    tweet = tweet.lower()
    tweet = ''.join(re.sub(r"http\S+", "", tweet))
    tweet = ''.join(re.sub(r"https\S+", "", tweet))
    tweet = ''.join(re.sub(r"www.\S+", "", tweet))
    tweet = re.sub('\.\.\.', '', tweet)
    tweet = ''.join(c for c in unicodedata.normalize('NFC', tweet) if c <= '\uFFFF')
    l = [lem.lemmatize(t) for t in nltk.word_tokenize(tweet)]
    l = ' '.join(l)
    return l;
f1=open('Dataset/tweets.jsonl')
f2=open('Dataset/facts.txt',encoding='utf8')
for line in f1.readlines():
    tweet = json.loads(line)
    id.append({"id":str(tweet["id"])})
    dataset[str(tweet["id"])]=tweet["text"]
for line in f2.readlines():
    l=line.split("<||>")
    facts.append({"id":l[0]})
for x in id:
    if x not in facts:
        nonfactual.append(preprocess(dataset[x['id']]))
    else:
        factual.append(preprocess(dataset[x['id']]))
Y=[]
i=0
while i!=len(factual):
    Y.append(1)
    i=i+1
while i!=len(factual+nonfactual):
    Y.append(0)
    i=i+1
vec=TfidfVectorizer(stop_words=stop)
X=vec.fit_transform(factual+nonfactual)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
clf = RandomForestClassifier(max_depth=2, random_state=0,class_weight="balanced")
clf.fit(X_train,y_train)
print(clf.predict_proba(X_test))
print(confusion_matrix(y_test,clf.predict(X_test)))
fpr,tpr,thresholds=metrics.roc_curve(y_test,clf.predict(X_test),pos_label=1)
auc=metrics.auc(fpr,tpr)
print("Accuracy : ",metrics.accuracy_score(y_test,clf.predict(X_test)))
print("AUC : ",auc)
scores=cross_val_predict(clf,X,Y,cv=5,method='predict_proba')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))