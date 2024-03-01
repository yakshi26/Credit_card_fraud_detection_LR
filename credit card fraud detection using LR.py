#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("creditcard.csv")
data.head()


# In[2]:


data.tail()


# In[3]:


fraud=data.loc[data['Class']==1]
normal=data.loc[data['Class']==0]


# In[4]:


fraud.sum()


# In[5]:


len(fraud)


# In[6]:


len(normal)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x=data["Amount"],y=data["Class"])


# In[11]:


sns.relplot(x=data["Amount"],y=data["Time"],hue="Class",data=data)


# In[13]:


sns.barplot(x=data["Class"],y=data["Amount"])


# In[15]:


X=data.drop(["Class"],axis=1)
y=data["Class"]


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[20]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)


# In[21]:


pred=clf.predict(X_test)


# In[22]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[23]:


print(confusion_matrix(y_test,pred))


# In[24]:


print(accuracy_score(y_test,pred))


# In[25]:


print(classification_report(y_test,pred))


# In[32]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

a = [0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]

a_scaled = scaler.transform([a])

print(a_scaled)
clf.predict(a_scaled)


# In[ ]:




