#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessory libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading data set to the python environment
data=pd.read_csv(r'E:\blessy\ict\PROJECT\train.csv')


# In[3]:


#displaying the first 5 rows of the dataset
data.head()


# In[4]:


data.shape #determine the number of rows and columns present in the dataset


# In[5]:


data.describe() #to get the basic information


# In[6]:


data.dtypes #finding the data types of each 


# In[7]:


data['selection'].value_counts() #finding value counts of target column


# In[8]:


#changing the data types of height and weight
data["height"] = data["height"].apply(lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1]))
data["weight"] = data["weight"].str.extract("(\d+)").astype(float)


# In[9]:


data.dtypes


# In[10]:


data.isna().sum()


# In[11]:


data['selection'].value_counts().plot.bar()


# In[12]:


features_with_na = [features for features in data.columns if data[features].isnull().sum()>=1]


# In[13]:


for feature in features_with_na:
    print(feature,np.round(data[feature].isnull().mean(),3), ' % missing values') # Round to 3 Desimal points


# there are missing values in gender,weight,ball controlling skills,jump skills,penlties conversion,mental strength,shot accuracy,strong foot,behaviour rating,matches played,fitness rating,coaching,and year of experience

# In[14]:


#finding the distribution
freqgraph=data.select_dtypes(include=['float'])
freqgraph.hist(figsize=(20,15))
plt.show()


# In[15]:


plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(),cmap='coolwarm',fmt='.2f',annot =True)


# In[16]:


plt.figure(figsize=(15, 5))
sns.barplot(x=data['selection'], y=data['age'])
plt.xticks(rotation='vertical')


# In[17]:


plt.bar(data['selection'],data['years_of_experience'])
plt.xticks(rotation=90)
plt.show()


# In[18]:


data.columns


# In[19]:


features_with_na


# In[20]:


for col in['weight', 'ball_controlling_skills', 'jumping_skills', 'penalties_conversion_rate', 'mental_strength', 'shot_accuracy',
  'behaviour_rating', 'matches_played', 'fitness_rating', 'years_of_experience']:
    data[col]=data[col].fillna(data[col].median())


# In[21]:


categorical_columns = ['strong_foot', 'gender', 'coaching' ]
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)


# In[22]:


data.isna().sum()


# In[23]:


data.columns


# In[24]:


num_cols=['age', 'height', 'weight',
       'ball_controlling_skills', 'body_reflexes', 'body_balance',
       'jumping_skills', 'penalties_conversion_rate', 'mental_strength',
       'goalkeeping_skills', 'defending_skills', 'passing_skills',
       'dribbling_skills', 'shot_accuracy', 'body_strength_stamina',
       'max_running_speed', 'behaviour_rating',
       'matches_played', 'fitness_rating', 'trophies_won', 'years_of_experience', 'no_of_disqualifications',
       'selection']


# In[25]:


for i in num_cols:
    plt.figure()
    plt.boxplot(data[i])
    plt.title(i)


# In[26]:


outliers=['age', 'height', 'weight',
       'ball_controlling_skills', 'body_reflexes', 'body_balance',
       'jumping_skills', 'mental_strength',
       'goalkeeping_skills', 'passing_skills',
       'dribbling_skills', 'shot_accuracy', 'body_strength_stamina',
       'max_running_speed', 'behaviour_rating',
       'matches_played', 'fitness_rating', 'no_of_disqualifications',
       'selection']


# In[27]:


#outliers are detected
for feature in outliers:
  Q1 = data[feature].quantile(0.25)
  Q2 = data[feature].quantile(0.50)
  Q3 = data[feature].quantile(0.75)
  print('\n',feature,'\n')
  print('Q1  25 % value =',Q1)
  print('Q2  50 % value =',Q2)
  print('Q3  75 % value =',Q3)
  IQR=Q3-Q1
  print('IQR=',IQR)
  up_lim=Q3+1.5*IQR
  low_lim=Q1-1.5*IQR
  print('\nUpper limit=',up_lim)
  print('Lower limit=',low_lim)


# In[28]:


#one hot encoding
data1=pd.get_dummies(data)


# In[29]:


data1.head()


# In[30]:


#seperating feature and target variable
X=data1.drop('selection',axis=1)
y=data1['selection']


# In[31]:


#split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state = 42)


# In[32]:


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logit_model= logr.fit(X_train,y_train)
y_pred_logr = logit_model.predict(X_test)


# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score


# In[34]:


print('Accuracy_Score =',accuracy_score(y_test,y_pred_logr))
print('Precision_Score =',precision_score(y_test,y_pred_logr))

print('Recall_Score = ',recall_score(y_test,y_pred_logr))
print('f1_Score = ',f1_score(y_test,y_pred_logr))


# In[35]:


confusion_matrix(y_test,y_pred_logr)


# In[36]:


#from sklearn.svm import SVC
#svmclf = SVC(kernel = 'linear')
#svmclf = svmclf.fit(X_train,y_train)
#y_pred_svm = svmclf.predict(X_test)


# In[37]:


#print('Accuracy is :',accuracy_score(y_test,y_pred_svm))
#print(confusion_matrix(y_test,y_pred_svm))


# In[38]:


# Decision Tree


# In[39]:


#from sklearn.tree import DecisionTreeClassifier
#dt_clf = DecisionTreeClassifier()
#dt_clf = dt_clf.fit(X_train,y_train)
#y_pred_dt = dt_clf.predict(X_test)


# In[40]:


#print('Accuracy is:',accuracy_score(y_test,y_pred_dt))


# In[41]:


#Random Forest


# In[42]:


#from sklearn.ensemble import RandomForestClassifier
#rf_clf = RandomForestClassifier()
#rf_clf_model =rf_clf.fit(X_train,y_train)
#y_pred_rf_clf=rf_clf_model.predict(X_test)


# In[ ]:




