#!/usr/bin/env python
# coding: utf-8

# # Loading the datasets and other libraries

# In[1]:


# loading the packages and importing the train and test data sets----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as mt

train_data = pd.read_csv("C:\\Users\\vivek\\Desktop\\bigdata\\train.csv")


test_data = pd.read_csv("C:\\Users\\vivek\\Desktop\\bigdata\\test.csv")


# In[2]:


# checking the shape of test data set
test_data.shape


# In[3]:


# importing the sample submission file
submission = pd.read_csv("C:\\Users\\vivek\\Desktop\\bigdata\\sample_submission.csv")


# # Data Exploration and Data Anlysis----------------------------------------------------------------------------------------------------------

# In[4]:


# before proceeding ahead , I would like to check how the saleprice is distributed and henece plotting the target variable

import seaborn as sns

target_variable = train_data.SalePrice
mt.figure()
sns.distplot(target_variable)
mt.title('SalePrice- Distrubution')
mt.show()


# In[5]:


# log transforming the target variable and plotting it again to check if the skew is reduced or not

sns.distplot(np.log(target_variable))
mt.title('Distribution of Log-transformed SalePrice')
mt.xlabel('log(SalePrice)')
mt.show()


# In[6]:


# checking for the null values in both train and test datasets

train_data.isna().sum().sort_values(ascending=False).head(50)


# In[7]:


test_data.isna().sum().sort_values(ascending=False).head(50)


# In[8]:


# from the above step, it is observed that the Alley', 'FireplaceQu','PoolQC', 'Fence', 'MiscFeature' have more than 70% of the data missing
# henec droping these attributes in both the train and test datasets

train_data = train_data.drop(['Alley', 'FireplaceQu','PoolQC', 'Fence', 'MiscFeature'], axis =1)
train_data.head()


# In[9]:


test_data = test_data.drop(['Alley', 'FireplaceQu','PoolQC', 'Fence', 'MiscFeature'], axis =1)


# In[10]:


train_data.info()


# In[11]:


# from the above step in the train datset we have NAs in the following numeric attributes
# after looking at the dataset, I have decided to fill these NA's with 0--but, in the next steps i might require to transform the
#attributes into log values and log(0) is infinity, henec filling with 1 instead of zero

train_data['LotFrontage'] = train_data['LotFrontage'].fillna(1)
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(1)
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(1)


# In[12]:


test_data.isna().sum().sort_values(ascending=False).head(50)


# In[13]:


# from the above step in the test datset we have NAs in the following numeric attributes
# after looking at the dataset, I have decided to fill these NA's with 0--but, in the next steps i might require to transform the
#attributes into log values and log(0) is infinity, henec filling with 1 instead of zero


test_data['LotFrontage'] = test_data['LotFrontage'].fillna(1)
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(1)
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(1)
test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(1)
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(1)
test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(1)
test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].fillna(1)
test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(1)
test_data['GarageCars'] = test_data['GarageCars'].fillna(1)
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(1)


# In[14]:


# checking for the null values in the categorical data of the train dataset
catvariables_columns = train_data.select_dtypes(include='object').columns
train_data[catvariables_columns].isna().sum().sort_values(ascending=False).head(20)


# In[15]:


# creating a copy of the train dataset and filling the NA's in the copy with 'None'

copy_train_data = train_data.copy()
traindata_fill_NA = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                     'MasVnrType']
for cat in traindata_fill_NA:
    copy_train_data[cat] = copy_train_data[cat].fillna("None")


# In[16]:


# checking for the null values in the categorical data of the test dataset

test_catvariables_columns = test_data.select_dtypes(include='object').columns

test_data[test_catvariables_columns].isna().sum().sort_values(ascending=False).head(20)


# In[17]:


# creating a copy of the test dataset and filling the NA's in the copy with 'None'

copy_test_data = test_data.copy()
testdata_fill_NA = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                     'MasVnrType', 'MSZoning', 'Utilities','Functional','Electrical','KitchenQual','SaleType','Exterior2nd','Exterior1st']
for cat in testdata_fill_NA:
                      copy_test_data[cat] = copy_test_data[cat].fillna("None")


# In[18]:


copy_test_data.shape


# In[19]:


copy_train_data.isna().sum().sort_values(ascending=False).head()


# In[21]:

# Sactter plots for identifying outliers ---------------------
import matplotlib.pyplot as plt
correlation = copy_train_data.corr()
numeric_columns = copy_train_data.select_dtypes(exclude='object').columns
corr_to_saleprice = correlation['SalePrice']
x_cols = 5
x_rows = 8
fig, ax_arr = plt.subplots(x_rows, x_cols, figsize=(16,20), sharey=True)
plt.subplots_adjust(bottom=-0.8)
for j in range(x_rows):
    for i in range(x_cols):
        plt.sca(ax_arr[j, i])
        index = i + j*x_cols
        if index < len(numeric_columns):
            plt.scatter(copy_train_data[numeric_columns[index]], copy_train_data.SalePrice)
            plt.xlabel(numeric_columns[index])
            plt.title('Corr to SalePrice = '+ str(np.around(corr_to_saleprice[index], decimals=3)))
plt.show()


# In[22]:


correlation['SalePrice'].sort_values(ascending=False)


# In[23]:


f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=12)
sns.heatmap(correlation)
plt.show()


# In[24]:


# identifying the outliers from the above scattered plots and dropping them from the train dataset

copy_train_data = copy_train_data.drop(copy_train_data['LotFrontage']
                                     [copy_train_data['LotFrontage']>200].index)
copy_train_data = copy_train_data.drop(copy_train_data['LotArea']
                                     [copy_train_data['LotArea']>100000].index)
copy_train_data = copy_train_data.drop(copy_train_data['MasVnrArea']
                                     [copy_train_data['MasVnrArea']>1500].index)
copy_train_data = copy_train_data.drop(copy_train_data['BsmtFinSF1']
                                     [copy_train_data['BsmtFinSF1']>4000].index)
copy_train_data = copy_train_data.drop(copy_train_data['BsmtFinSF2']
                                     [copy_train_data['BsmtFinSF2']>1200].index)

copy_train_data = copy_train_data.drop(copy_train_data['TotalBsmtSF']
                                     [copy_train_data['TotalBsmtSF']>4000].index)

copy_train_data = copy_train_data.drop(copy_train_data['1stFlrSF']
                                     [copy_train_data['1stFlrSF']>4000].index)

copy_train_data = copy_train_data.drop(copy_train_data['GrLivArea']
                                     [copy_train_data['GrLivArea']>4000].index)


copy_train_data = copy_train_data.drop(copy_train_data['OpenPorchSF']
                                     [copy_train_data['OpenPorchSF']>450].index)
copy_train_data = copy_train_data.drop(copy_train_data['EnclosedPorch']
                                     [copy_train_data['EnclosedPorch']>400].index)
copy_train_data = copy_train_data.drop(copy_train_data['EnclosedPorch']
                                     [copy_train_data['EnclosedPorch']>400].index)





# In[25]:


copy_train_data['Electrical'] = copy_train_data['Electrical'].fillna(copy_train_data['Electrical']).mode(0)


# In[26]:


copy_train_data.info()


# In[27]:


Copy_train_data = copy_train_data.drop(['Electrical'], axis =1)


# In[28]:


Copy_test_data = copy_test_data.drop(['Electrical'], axis =1)


# In[29]:


# as we checked the skew at the begining, to reduce the skew, transforming the target variable to log and replacing the name 

copy_train_data['SalePrice'] = np.log(copy_train_data['SalePrice'])
copy_train_data = copy_train_data.rename(columns={'SalePrice': 'SalePrice_log'})


# In[30]:


# checking the attributes skew with new target variable

correlation = copy_train_data.corr()
correlation['SalePrice_log'].sort_values(ascending=False)


# In[31]:


copy_train_data.shape


# In[32]:


# combining the train and test datasets to encode the categorical variables

full_data = pd.concat([copy_train_data,copy_test_data], axis=0,sort=False)


# In[33]:


#on-hot encoding of the full dataset

full_data = pd.get_dummies(full_data)


# In[34]:


full_data.shape


# In[35]:


# spliting the full dataset into train and test datasets based on their ID's

copy_train_datam = full_data[full_data['Id']<1461]
copy_test_datam = full_data[full_data['Id']>1460]


# In[36]:


# from the above correlations and from the observations in scatter plots, we can drop the follwing attributes
attributes_drop = ['SalePrice_log', 'MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 
                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd']

X = copy_train_datam.drop(attributes_drop, axis=1)

# segregate the target variable 
y = copy_train_datam.SalePrice_log

# One-hot-encoding to transform all categorical data
X = pd.get_dummies(X)


# In[37]:


X.shape


# In[38]:


attributes_drop = ['SalePrice_log','MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 
                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd']

X_t = copy_test_datam.drop(attributes_drop, axis=1)

# Create target object and call it y
#y = copy_train_data.SalePrice_log

# One-hot-encoding to transform all categorical data
X_t = pd.get_dummies(X_t)


# In[39]:


X_t.shape


# # Machine Learning Models

# In[40]:


# Split into validation and training data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.7, test_size=0.3, random_state=1)


# In[41]:



# Linear regression implementation and calculation of Accuracy and other metrics

from sklearn.linear_model import LinearRegression
from sklearn import metrics
linear_model = LinearRegression()
linear_model.fit(train_X, train_y)
linear_model_predictions = linear_model.predict(val_X)
print("Accuracy: ", linear_model.score(val_X, val_y)*100)
print('Mean Absolute Error:', metrics.mean_absolute_error((val_y), (linear_model_predictions)))  
print('Mean Squared Error:', metrics.mean_squared_error((val_y), (linear_model_predictions)))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error((val_y), (linear_model_predictions))))


# In[42]:


# Random Forest implementation and calculation of Accuracy and other metrics

from sklearn.ensemble import RandomForestRegressor
random_model = RandomForestRegressor(random_state=5)
random_model.fit(train_X, train_y)
random_model_predictions = random_model.predict(val_X)


print("Accuracy: ", random_model.score(val_X, val_y)*100)
print('Mean Absolute Error:', metrics.mean_absolute_error((val_y), (random_model_predictions))) 
print('Mean Squared Error:', metrics.mean_squared_error((val_y), (random_model_predictions)))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error((val_y), (random_model_predictions))))


# In[43]:


X_t.shape


# In[44]:


# from the above results after implementing random forest and linear regression
# linea regression has better accuracy and low error henec linear regression is selected as the final model

y_pred = linear_model.predict(X_t)
submission['SalePrice'] = np.expm1(y_pred)


# In[45]:



submission.to_csv('sample_submission.csv',index=False)


# In[46]:


len(y_pred)


# In[47]:


sns.distplot(y_pred)
mt.title('Distribution y_pred (predicted SalePrice)')
mt.xlabel('log(SalePrice)')
mt.show()


# In[ ]:




