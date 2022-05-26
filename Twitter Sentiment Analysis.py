#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries 

# In[1]:


import pandas as pd
import numpy as np


# ## Reading the Dataset 

# In[2]:


df = pd.read_csv(r'C:\Users\User\Documents\Projects\NLP Projects\Twitter Tweets\Twitter Tweets Dataset.csv')
df


# ## Treating Null Values 

# ### Checking for the Null Values

# In[3]:


df[df['selected_text'].isnull()==True]


# ### Here we are dropping the null value rows because the number of null value rows are very less compared to the entire dataset

# In[4]:


df.drop(314,inplace=True)


# In[5]:


sum(df['selected_text'].isnull())


# ## Data Preprocessing

# ### Train Test Split

# In[6]:


X=df['selected_text']
y=df['sentiment']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
train_sentences,test_sentences,train_labels,test_labels=train_test_split(X,y,test_size=0.2,random_state=42)


# In[7]:


print(f'Total training samples :{train_sentences.shape}')
print("\n")
print(f'Total training labels {train_labels.shape}')
print("\n")
print(f'Total test samples:{test_sentences.shape}')
print("\n")
print(f'Total test labels {test_labels.shape}')


# ## Training the Model Using Naive Bayes Classifier 

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
model =Pipeline([('tfidf_vectorizer',TfidfVectorizer(lowercase = True,
                                                     stop_words = 'english',
                                                    analyzer = 'word')),
                 
                 ('naive_bayes', MultinomialNB())])


model.fit(train_sentences, train_labels)


# ## Model Accuracy Score 

# In[9]:


accuracy=model.score(test_sentences,test_labels)
print(f'model_accuracy:{accuracy}')


# In[10]:


model_preds=model.predict(test_sentences)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
    return model_results
calculate_results(y_true=test_labels,y_pred=model_preds)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




