#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy import stats
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[416]:


get_ipython().system('pip install lmfit')


# In[417]:


get_ipython().system('pip install seaborn')
#pip install missingno


# In[374]:


get_ipython().system('pip install missingno')


# In[2]:


df = pd.read_csv("C:/Users/jay patel/Desktop/Covid_prediction_model/data/Cleaned-Data.csv")


# In[3]:


df.head()


# In[4]:



pd.pandas.set_option('display.max_columns',None)


# In[5]:


display("Shape of dataset")
print("Rows:",df.shape[0],"\nColumns:",df.shape[1])


# In[6]:


df.info()


# In[7]:


for i in df.columns:
    print("\nColumn Name:",i,"-->",df[i].unique(),"-->Unique Count",len(df[i].unique()))


# In[8]:


severity_columns = df.filter(like='Severity_None').columns


# In[9]:


df['Severity_None'].replace({1:'None',0:'No'},inplace =True)


# In[10]:


df['Condition']=df[severity_columns].values.tolist()


# In[11]:


def removing(list1):
    list1 = set(list1) 
    list1.discard("No")
    a = ''.join(list1)
    return a


# In[12]:


df['Condition'] = df['Condition'].apply(removing)


# In[13]:


age_columns = df.filter(like='Age_').columns
gender_columns = df.filter(like='Gender_').columns
contact_columns = df.filter(like='Contact_').columns


# In[14]:


No_risk_age = df.groupby(['Severity_None'])[age_columns].sum()
No_risk_gender = df.groupby(['Severity_None'])[gender_columns].sum()
No_risk_contact = df.groupby(['Severity_None'])[contact_columns].sum()


# In[15]:


Low_risk_age = df.groupby(['Severity_Mild'])[age_columns].sum()
Low_risk_gender = df.groupby(['Severity_Mild'])[gender_columns].sum()
Low_risk_contact = df.groupby(['Severity_Mild'])[contact_columns].sum()


# In[16]:


Moderate_risk_age = df.groupby(['Severity_Moderate'])[age_columns].sum()
Moderate_risk_gender = df.groupby(['Severity_Moderate'])[gender_columns].sum()
Moderate_risk_contact = df.groupby(['Severity_Moderate'])[contact_columns].sum()


# In[17]:


Severe_risk_age = df.groupby(['Severity_Severe'])[age_columns].sum()
Severe_risk_gender = df.groupby(['Severity_Severe'])[gender_columns].sum()
Severe_risk_contact = df.groupby(['Severity_Severe'])[contact_columns].sum()


# In[18]:


sns.countplot(df['Condition'])


# In[19]:


df['Symptoms_Score'] = df.iloc[:,:5].sum(axis=1) + df.iloc[:,6:10].sum(axis=1)


# In[20]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Condition'] = le.fit_transform(df['Condition'])


# In[21]:


from pylab import rcParams
rcParams['figure.figsize'] = 13, 18
corrmat = df.corr()
k = 22
cols = corrmat.nlargest(k, 'Condition')['Condition'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[22]:


df.head()


# In[48]:


X= df.drop(['Condition','None_Sympton','Nasal-Congestion','Runny-Nose','None_Experiencing','Age_0-9','Age_10-19','Age_20-24','Age_25-59','Gender_Female','Gender_Male','Gender_Transgender','Contact_Dont-Know','Contact_No','Sore-Throat','Severity_Mild','Severity_Moderate','Severity_None','Severity_Severe','Country','Contact_Yes','Symptoms_Score'],axis=1)
y= df['Condition']


# In[47]:


X=np.array(X).reshape((-1,1))


# In[45]:


y=np.array(y).reshape((-1,1))


# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


import pandas as pd
train=pd.concat([X_train, y_train])
train.to_csv("C:/Users/jay patel/Desktop/Covid_prediction_model/data/train.csv",index=False)


# In[28]:


train


# In[32]:


dev


# In[31]:


dev=pd.concat([X_test, y_test])
dev.to_csv("C:/Users/jay patel/Desktop/Covid_prediction_model/data/test.csv",index=False)


# In[33]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'lbfgs')
model.fit(X_train, y_train)


# In[41]:


X_train.info()


# In[34]:


# use the model to make predictions with the test data
y_pred = model.predict(X_test)


# In[35]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[36]:


import pickle


# In[37]:


pickle.dump(model, open('model.pkl','wb'))


# In[61]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 1, 1, 0, 0, 0, 0]]))


# In[ ]:




