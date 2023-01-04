#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


data=pd.read_csv(r"C:\Users\Admin\Desktop\Course DS\Data\brain_stroke.csv")
data.head()


# In[6]:


#selecting the variables for the model 
data1=data.iloc[:,[1,2,3,7,8,9,10]]
data1.head()


# In[7]:


#For converting categorical variable into binary ,label encoder technique
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["smoking_status"]=le.fit_transform(data1["smoking_status"])
data1


# In[8]:


#For check data balancing or not
data1["stroke"].value_counts()


# In[9]:


##split data for dependent an independent variale 
x=data1.iloc[:,:-1]
y=data1.iloc[:,-1]


# In[10]:


#Imbalancing techiniqye to convert into balance data ## Over samling Technique
from imblearn.over_sampling import RandomOverSampler
r=RandomOverSampler()
x_data,y_data=r.fit_resample(x,y)


# In[11]:


#check again data is balanced or not
from collections import Counter 
print(Counter(y_data))


# In[12]:


#Standardization for better accuracy 
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
scalerx=ss.fit_transform(x_data)
print(scalerx)


# In[13]:


#Split data into train and test data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.25,random_state=100)


# In[14]:


#import classification model :: Logistic Regression
from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
l1


# In[15]:


#Fitting the model 
model=l1.fit(x_train,y_train)
model


# In[16]:


#Predict the values using the x_test data
y_pred=model.predict(x_test)
y_pred


# In[17]:


##Accuracy Score of the model


# In[18]:


model.score(x_test,y_test)


# In[19]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred,  normalize=True, sample_weight=None)


# In[20]:


# Confusion Matrix for checking the accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred)
print("accuracy score : ",ac)
print("confusion matrix :")
print(cm)
print("classification report :",cr)


# In[ ]:





# In[24]:


# for i in data1.columns:
#     plt.figure(figsize = (15,6))
#     dfta1[i].value_counts().plot(kind = 'pie', autopct = '%1.1f%%')
#     plt.xticks(rotation = 90)
#     plt.show()


# In[23]:


#Data Visualization
import seaborn as sns
for i in data1.columns:
    plt.figure(figsize = (10,6))
    sns.countplot(data1[i], data = data1, palette = 'hls')
    plt.xticks(rotation = 90)
    plt.show()

