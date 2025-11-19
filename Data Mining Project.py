#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")


# In[26]:


df = pd.read_csv('D:\\data mining project\\globalterrorismdb_0718dist.csv', encoding='latin1')
df


# In[27]:


df.rename(columns={'iyear':'Year','imonth':'Month','city':'City','iday':'Day','country_txt':'Country','region_txt':'Region',
                   'attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded',
                   'summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type',
                   'motive':'Motive'},inplace=True)
df['Casualities'] = df.Killed + df.Wounded
df=df[['Year','Month','Day','Country','Region','City','latitude','longitude','AttackType','Killed','Wounded','Casualities',
       'Target','Group','Target_type','Weapon_type']]
df


# In[28]:


# Remove all duplicate rows 
df = df.drop_duplicates(keep=False)
len(df)


# In[29]:


# Drop rows with missing latitude or longitude
df = df.dropna(subset=['longitude', 'latitude'])


# In[30]:


# df.fillna(0,inplace=True)
# len(df)


# In[31]:


# Check for missing values
print(df.isnull().sum())


# In[32]:


# Replace NaN values with 0
df.fillna(0, inplace=True)
df


# In[33]:


# Check for missing values
print(df.isnull().sum())


# In[34]:


year_on_x_axis = df['Year'].unique()
no_of_attacks_y_axis = df['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = year_on_x_axis,
           y = no_of_attacks_y_axis,
           palette = 'viridis')
plt.xticks(rotation = 45)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks each year')
plt.title('Attacks of each year')
plt.show()


# In[35]:


plt.subplots(figsize=(15,8))
sns.barplot(df['Country'].value_counts()[:15].index,df['Country'].value_counts()[:15].values,palette='Reds_d')
plt.title('Countries With The Most Attacks')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 45)
plt.show()


# In[36]:


pd.crosstab(df.Year, df.Region).plot(kind='area',figsize=(15,8))
plt.title('Terrorist Attacks by Region each Year')
plt.ylabel('Number of Attacks')
plt.show()


# In[37]:


pd.crosstab(df['AttackType'], df['Weapon_type']).plot(kind='area',figsize=(18,10))
plt.title('Relation between Attack Type and Weapon Used')
plt.ylabel('Rate of Usage')
plt.xticks(rotation= 45)
plt.show()


# In[38]:


x = df['Weapon_type']
y = df['Target_type']
plt.figure(figsize = (18,10))
plt.scatter(x, y)
plt.xticks(rotation = 45)
plt.show()


# In[39]:


Target_Type_on_x_axis = df['Target_type'].unique()
no_of_attacks_y_axis = df['Target_type'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = Target_Type_on_x_axis,
           y = no_of_attacks_y_axis,
           palette = 'Blues_d')
plt.xticks(rotation = 50)
plt.xlabel('Target Type')
plt.ylabel('Number of Attacks last 50 years')
plt.title('Terrorism Targets last 50 years')
plt.show()


# In[40]:


from sklearn.model_selection import train_test_split

X = df.drop("Casualities", axis=1)
y = df["Casualities"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# In[42]:


cat_cols = ["Country", "Region", "City", "Target_type", "Weapon_type"]
num_cols = ["Year", "Month", "Day", "latitude", "longitude", "Killed", "Wounded"]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("scale", StandardScaler(), num_cols)
])

# Cast categorical variables to strings before preprocessing
X_train[cat_cols] = X_train[cat_cols].astype(str)
X_test[cat_cols] = X_test[cat_cols].astype(str)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
df


# In[43]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_lr=lr.score(X_test, y_test)
print("Linear Regression Accuracy:",accuracy_lr )  


# In[44]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_dt=accuracy_score(y_test,y_pred)
print('Decision Tree Accuracy : ',accuracy_dt)


# In[45]:


if accuracy_lr > accuracy_dt:
    print("The Linear Regression algorithm is better than Decision Tree algorithm ")
else:
    print("The Decision Tree algorithm is better than Linear Regression algorithm")


# In[ ]:




