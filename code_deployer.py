#!/usr/bin/env python
# coding: utf-8

# In[2]:
#Importation librarie
import numpy as np
import scipy as sp
import sklearn
import pandas as pd
import mglearn

import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import re # méthode regex librairie
from nltk.corpus import stopwords
from nltk import word_tokenize


from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer, f1_score
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import mlflow.sklearn

# In[ ]:
data= pd.read_csv("C:/Users/JbPer/Openclassroom/dataPost.csv")



# In[13]:


tags = [t[1:len(t) - 1].split('><') for t in data['Tags']]


# In[14]:


data['tags'] = [t[1:len(t) - 1].split('><') for t in data['Tags']]


# In[16]:


def remove_punct(_str):
    """
    Enleve les ponctuations du text.

    Arguments:
        text(String): Row text with html
    Returns:
        cleaned String
    """
    _str = re.sub('[^\w]'," ", _str).lower()
    _str = re.sub('['+string.punctuation+']', ' ', _str)
    _str = re.sub('[\r]', ' ', _str)
    _str = re.sub('[\n]', ' ', _str)
    _str = re.sub('[«»…"]', ' ', _str)
    _str = re.sub('[0-9]', ' ', _str)
    return _str


# In[17]:


def remove_stop_words_en(_str):
    english_stopwords = set(stopwords.words('english'))
    filtre_stopen =  lambda text: [token for token in text if token.lower()
                                   not in english_stopwords]
    return [ txt for txt in filtre_stopen(word_tokenize(_str)) if len(txt)>2]


# In[18]:


def stemm(_str):
    snow_stemmer = SnowballStemmer(language='english')
    stemmi = lambda x: [snow_stemmer.stem(y) for y in x]
    return stemmi(_str)




# In[20]:


def Preprocessing(_str):
    str_remove_punc= remove_punct(_str)
    str_remove_stop = remove_stop_words_en(str_remove_punc)
    str_stem = stemm(str_remove_stop)
    return str_stem


# In[21]:


class predictors(TransformerMixin):

        def transform(self, X, **transform_params):
            return [Preprocessing(_str) for _str in X]

        def fit(self, X, y=None, **fit_params):
                return self

        def get_params(self, deep=True):
            return {}



# In[50]:
mlb = MultiLabelBinarizer()
tags_mlb = mlb.fit_transform(tags)


# In[51]:


data_out = pd.DataFrame(data=tags_mlb, columns=mlb.classes_)


# In[53]:


# Caractéristique et label

X= data['Post']
ylabels = data_out


# In[55]:
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=.25, random_state=42)


# In[57]:


from sklearn.pipeline import Pipeline


# In[58]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# In[59]:


def document(_str):
    return _str


# In[60]:


pipe = Pipeline([("clearner", predictors()),
                 ('vectorizer', TfidfVectorizer(max_features=500, tokenizer=document, lowercase=False)),
                 ('model', OneVsRestClassifier(SVC(random_state=42)))])


# In[61]:


pipe.fit(X_train, y_train)


# In[62]:


y_pred = pipe.predict(X_test)


# In[69]:


from sklearn.metrics import make_scorer, f1_score


# In[63]:


y_pred = pd.DataFrame(data=y_pred, columns=mlb.classes_)



# In[66]:


print(accuracy_score(y_test, y_pred))


# In[73]:


print("recall score micro : ",recall_score(y_test, y_pred, average='micro'))
print("recall score macro : ",recall_score(y_test, y_pred, average='macro'))


# In[74]:


print("Precision Score micro : ",precision_score(y_test, y_pred,average='micro'))
print("Precision Score macro : ",precision_score(y_test, y_pred,average='macro'))


# In[75]:


print("F1_Score : ",f1_score(y_test, y_pred, average='macro'))
print("F1_Score : ",f1_score(y_test, y_pred, average='micro'))
print("F1_Score : ",f1_score(y_test, y_pred, average='weighted'))


# In[155]:


Text_norm = ["java est  mon sauveur,gloire aux machine learning"]


# In[156]:
my_prediction = pipe.predict(Text_norm)


# In[157]:
my_prediction = pd.DataFrame(data=my_prediction, columns=mlb.classes_)


# In[158]:
for (sample, pred) in zip(Text_norm,my_prediction):
    print(sample,"Prediction=>", pred)