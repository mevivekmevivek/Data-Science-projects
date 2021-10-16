####################################################################################################
#Sai Vivek Kammari
#a180777
#Sem II 2020
#Masters in Data Science


####################################################################################################

# # loading all the required libraries and Data files
#     

# In[1]:


# loading all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import date
from bayes_opt import BayesianOptimization
from sklearn import neighbors
from sklearn import ensemble
import xgboost as xgb


# In[2]:


# loading all the data files into pandas data frames

airstores = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//air_store_info.csv//air_store_info.csv")
hpgstores = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//hpg_store_info.csv//hpg_store_info.csv")
airreserves = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//air_reserve.csv//air_reserve.csv")
airvisits = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//air_visit_data.csv//air_visit_data.csv")
hpgreserves = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//hpg_reserve.csv//hpg_reserve.csv")
dateinformation = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//date_info.csv//date_info.csv")
storeidlookup = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//store_id_relation.csv//store_id_relation.csv")
predictions = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//sample_submission.csv//sample_submission.csv")


# In[3]:


# checking the shapes of all the datasets
print("airstores.shape: ", airstores.shape)
print("hpgstores.shape: ", hpgstores.shape)
print("airreserves.shape: ",airreserves.shape)
print("airvisits.shape: ", airvisits.shape)
print("hpgreserves.shape: ", hpgreserves.shape)
print("dateinformation.shape: ", dateinformation.shape)
print("storeidlookup.shape: ", storeidlookup.shape)
print("predictions.shape: ",predictions.shape)


# In[4]:


print(airstores.head())
print( hpgstores.head())
print(airreserves.head())
print( airvisits.head())
print( hpgreserves.head())
print( dateinformation.head())
print( storeidlookup.head())
print(predictions.head())


# In[5]:


# checking for number of unique stores, the visit date is given for

x= airvisits.air_store_id.nunique()
y = airstores.air_store_id.nunique()

p = airvisits.air_store_id.isin(airstores.air_store_id).all()

if x == y :
    print(" The number of stores in air visits dataset is equal to number of stores in airstores dataset ")
if p == True :
    print()
    print(" All the stores in air visits are present in airstores dataset ")

    


# In[6]:


predictions.head()


# In[7]:


predictions[["source",'storeid','visit_date']] = predictions.id.str.split("_",expand=True,)
predictions


# In[8]:


predictions['air_store_id']=predictions['source'].str.cat(predictions['storeid'], sep='_')
predictions


# In[145]:



predictions.air_store_id.nunique()


# In[9]:


predictions = predictions[['id', 'air_store_id', 'visit_date', 'visitors']]
predictions['visit_date'] = pd.to_datetime(predictions['visit_date'])

print(predictions.shape)
predictions.head()


# In[10]:


airvisits['id']=airvisits['air_store_id'].str.cat(airvisits['visit_date'], sep='_')

airvisits = airvisits[['id', 'air_store_id', 'visit_date', 'visitors']]


airvisits.head()


# In[11]:




airvisits.visit_date = pd.to_datetime(airvisits.visit_date)
series8 = airvisits.groupby('visit_date').sum()
series8.hist()
plt.show()

series8.plot()
plt.show()




ax = series8.plot(figsize=[20,10])
ax.set(xlabel="Visit_Date",
       ylabel="'Daily Visitors'",
       title='Historic',
       xlim=["2016-01-13", "2017-05-31"])

date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.show()


# In[12]:


airvisits['day_of_week'] = airvisits.visit_date.dt.weekday
airvisits.head(6)


# In[13]:



series1 = airvisits.groupby('day_of_week').sum()
ax = series1.plot.bar()
ax.set_xlabel('Weekday (Monday=0, Sunday=6)')



# In[14]:


completestoresdata = pd.merge(airstores, storeidlookup, how='left', on='air_store_id')
completestoresdata.shape


# In[15]:


completestoresdata


# In[16]:


completestoresdata = pd.merge(completestoresdata, hpgstores, how='left', on='hpg_store_id', suffixes=['_air', '_hpg'])


# In[17]:


completestoresdata


# # reservation data

# In[18]:


airreserves.head()


# In[19]:


print(hpgreserves.shape)
hpgreserves.head()


# In[20]:


hpgreserves = pd.merge(hpgreserves, storeidlookup, on='hpg_store_id')[airreserves.columns]


# In[21]:


hpgreserves.shape


# In[22]:


reserves = pd.concat([airreserves, hpgreserves], axis=0)
reserves.shape


# In[23]:


reserves.head()


# In[24]:


reserves.visit_datetime = pd.to_datetime(reserves.visit_datetime)
reserves.reserve_datetime = pd.to_datetime(reserves.reserve_datetime)
reserves['reserve_ahead'] = reserves.visit_datetime - reserves.reserve_datetime
reserves['hours_ahead'] = reserves.reserve_ahead / pd.Timedelta('1 hour')


# In[25]:


reserves


# In[26]:




airvisits['visit_date'] = pd.to_datetime(airvisits['visit_date'])
predictions['visit_date']= pd.to_datetime(predictions['visit_date'])


# In[27]:


airvisits['day_of_week'] = airvisits['visit_date'].dt.dayofweek
airvisits.head()


# In[28]:


predictions['day_of_week'] = predictions['visit_date'].dt.dayofweek
predictions.head()


# In[29]:


airvisits['month'] = airvisits['visit_date'].dt.month
airvisits.head()


# In[30]:


predictions['month'] = predictions['visit_date'].dt.month
predictions.head()


# In[31]:


airvisits['day_of_year'] = airvisits['visit_date'].dt.dayofyear
airvisits.head()


# In[32]:


predictions['day_of_year'] = predictions['visit_date'].dt.dayofyear
predictions.head()


# In[33]:


airvisits['days_in_month'] = airvisits['visit_date'].dt.days_in_month
airvisits.head()


# In[34]:


predictions['days_in_month'] = predictions['visit_date'].dt.days_in_month
predictions.head()


# In[35]:


airvisits['week_of_year'] = airvisits['visit_date'].dt.weekofyear
airvisits.head()


# In[36]:


predictions['week_of_year'] = predictions['visit_date'].dt.weekofyear
predictions.head()


# In[37]:


airvisits['month_end'] = airvisits['visit_date'].dt.is_month_end
airvisits.head()


# In[38]:


predictions['month_end'] = predictions['visit_date'].dt.is_month_end
predictions.head()


# In[39]:


start_day = pd.to_datetime('2016-01-01')


# In[40]:


x = (airvisits.visit_date - start_day)


# In[41]:


airvisits['days_from_startday'] = x.apply(lambda dt: dt.days)


# In[42]:


predictions['days_from_startday'] = x.apply(lambda dt: dt.days)


# In[43]:



holidayinfo = dateinformation
holidayinfo.head()


# In[44]:


holidayinfo['succeded by holiday'] = holidayinfo.holiday_flg.shift(-1).fillna(0).astype(int)
holidayinfo['preceeded by holiday'] = holidayinfo.holiday_flg.shift(1).fillna(0).astype(int)


# In[45]:


holidayinfo.head()


# In[46]:


holidayinfo=holidayinfo.drop(columns = 'day_of_week', axis=1)


# In[47]:


holidayinfo.head()


# In[48]:


#merging the full dataset and holiday information

holidayinfo["calendar_date"]=holidayinfo["calendar_date"].astype('datetime64[D]')

airvisits = pd.merge(left=airvisits, right=holidayinfo, how='left',left_on='visit_date', right_on='calendar_date')

predictions = pd.merge(left=predictions, right=holidayinfo, how='left',left_on='visit_date', right_on='calendar_date')


# In[142]:


predictions.to_csv('C:\\Users\\vivek\\Desktop\\bigdata\\bigdata project 2\\recruit-restaurant-visitor-forecasting\\predictions.csv', index = False)


# In[49]:



airvisits.info()


# In[50]:


predictions.info()


# In[51]:


airvisits['year'] = airvisits['visit_date'].dt.year
airvisits = airvisits.drop(columns=['calendar_date'])


# In[52]:


predictions['year'] = predictions['visit_date'].dt.year
predictions = predictions.drop(columns=['calendar_date'])


# In[53]:


airvisits.info()


# In[ ]:





# # Getting the location of the stores

# In[54]:


completestoresdata.head()


# In[55]:


area_split = completestoresdata.air_area_name.str.split(' ', expand=True)
completestoresdata['Area'] = area_split[0]
completestoresdata['city'] = area_split[1]


# In[56]:


p = area_split.iloc[:, 2:]


# In[57]:


completestoresdata['street'] = p.apply(lambda row: ' '.join(row.dropna()), axis=1)


# In[58]:


completestoresdata.head()


# In[59]:


storesbystreet = completestoresdata.groupby(['air_area_name']).size().to_frame(name='stores in same street').reset_index()
storesbycity = completestoresdata.groupby(['Area', 'city']).size().to_frame(name='stores in same city').reset_index()
storesbyArea = completestoresdata.groupby('Area').size().to_frame(name='stores in same Area').reset_index()


# In[60]:


print(storesbystreet.head(2))
print(storesbycity.head(2))
print(storesbyArea.head(2))


# In[61]:


completestoresdata = pd.merge(left=completestoresdata, right=storesbystreet, how='left', on='air_area_name')
completestoresdata = pd.merge(left=completestoresdata, right=storesbycity, how='left', on=['Area', 'city'])
completestoresdata = pd.merge(left=completestoresdata, right=storesbyArea, how='left', on='Area')


# In[62]:


completestoresdata.head()


# In[63]:


airvisits = pd.merge(left=airvisits, right=completestoresdata, how='left', on='air_store_id')


# In[64]:


predictions = pd.merge(left=predictions, right=completestoresdata, how='left', on='air_store_id')


# In[65]:


airvisits.info()


# In[66]:


predictions.info()


# In[67]:


#display(reserves.head())
#display(reserves.info())


# In[68]:


reserves['plannedvisit'] = reserves.visit_datetime.dt.date

reserves["plannedvisit"]=reserves["plannedvisit"].astype('datetime64[D]')

reserves


# In[69]:


reserves['reservedate'] = reserves.reserve_datetime.dt.date
reserves


# In[70]:



reserves['bufferhours'] = reserves.reserve_ahead / pd.Timedelta('1 hour')
reserves['bufferdays'] = reserves.reserve_ahead.apply(lambda T: T.days)


# In[71]:


reserves['reservebuffer'] = reserves.visit_datetime - reserves.reserve_datetime.astype('datetime64[ns]')
reserves


# In[72]:


reserves['bufferhours'] = reserves.reserve_ahead / pd.Timedelta('1 hour')
reserves


# In[73]:


reserves['bufferdays'] = reserves.reserve_ahead.apply(lambda delta_t: delta_t.days)
reserves


# In[74]:


reservationsummary = reserves.groupby(['air_store_id', 'plannedvisit'])['reserve_visitors', 'hours_ahead'].agg({'reserve_visitors': ['count','sum'], 'hours_ahead': 'mean'}).reset_index()


# In[148]:


reservationsummary.columns=['air_store_id', 'plannedvisit', 'numofreserves', 'numofreservevisitors', 'bufferhoursavg']


# In[149]:


reservationsummary


# # merging the reservation data with the Visits data
# 

# In[77]:


airvisits = pd.merge(left=airvisits, right=reservationsummary, how='left',left_on=['air_store_id', 'visit_date'], right_on=['air_store_id', 'plannedvisit'])


# In[78]:


airvisits.info()


# In[79]:


predictions = pd.merge(left=predictions, right=reservationsummary, how='left',left_on=['air_store_id', 'visit_date'], right_on=['air_store_id', 'plannedvisit'])


# In[80]:


predictions.info()


# In[81]:


airvisits['genre-area'] = airvisits['air_area_name'].astype(str)+'_'+airvisits['air_genre_name'].astype(str)
airvisits.head()


# In[82]:


predictions['genre-area'] = predictions['air_genre_name'].astype(str)+'_'+predictions['air_area_name'].astype(str)
predictions.head()


# In[83]:


airvisits['store-weekday'] = airvisits['day_of_week'].astype(str)+'_'+airvisits['air_area_name'].astype(str) 
airvisits.head()


# In[84]:


predictions['store-weekday'] = predictions['day_of_week'].astype(str)+'_'+predictions['air_area_name'].astype(str) 
predictions.head()


# In[85]:


airvisits['store-weekday-holiday'] = airvisits['store-weekday'].astype(str)+'_'+airvisits['holiday_flg'].astype(str) 
airvisits.head()


# In[86]:


predictions['store-weekday-holiday'] = predictions['store-weekday'].astype(str)+'_'+airvisits['holiday_flg'].astype(str) 
predictions.head()


# In[87]:


airvisits['dataset'] = 'past'
predictions['dataset'] = 'future'


# # Merging the airvisist and Predictions data before encoding categorical data
# 

# In[88]:


full_data = pd.concat([airvisits,predictions], axis=0,sort=False)


# In[89]:


full_data


# In[90]:


groupbycols = ['air_store_id', 'day_of_week', 'holiday_flg']

visitorstats = full_data                .query('dataset == "past"')                .groupby(groupbycols)                ['visitors']                .agg(['mean', 'median', 'min', 'max'])                .rename(columns=lambda colname: str(colname)+'_visitors')                .reset_index()


# In[91]:


visitorstats


# In[92]:


full_data = full_data.merge(visitorstats, how='left', on=groupbycols)


# In[93]:


full_data.info()


# # Grouping the attributes based on datatype

# In[94]:


# grouping the varibales based on their datatypes

# categorical columns
catagoricalcolumns = ['air_store_id', 'air_genre_name', 'air_area_name', 'hpg_genre_name', 'hpg_area_name','Area', 'city', 'street', 
            'genre-area', 'store-weekday', 'store-weekday-holiday'
           ]


# In[95]:



idcol = 'id'

# target variable
targetvariable = 'visitors'


# In[96]:


# binary columns 
binary_columns = ['month_end', 'holiday_flg', 'succeded by holiday', 'preceeded by holiday']


# In[97]:


# numeric columns
numericcolumns = ['day_of_week', 'year', 'month', 'day_of_year', 'days_in_month', 'week_of_year', 'days_from_startday', 
            'latitude_air', 'longitude_air','stores in same street', 'stores in same city', 'stores in same Area','numofreserves', 'numofreservevisitors', 'bufferhoursavg', 
            'mean_visitors', 'median_visitors', 'min_visitors', 'max_visitors' ]


# # Label Encoding

# In[98]:


cat_to_int_cols = ['air_store_id_int',
 'air_genre_name_int',
 'air_area_name_int',
 'hpg_genre_name_int',
 'hpg_area_name_int',
 'Area_int',
 'city_int',
 'street_int',
 'genre-area_int',
 'store-weekday_int',
 'store-weekday-holiday_int']

#cat_to_int_encoders = {}


# In[99]:


full_data.info()


# In[100]:


encoder1 = LabelEncoder()


# In[101]:


full_data['air_store_id_int'] = encoder1.fit_transform(full_data['air_store_id'].astype(str))
full_data['air_genre_name_int'] = encoder1.fit_transform(full_data['air_genre_name'].astype(str))

full_data['hpg_area_name_int'] = encoder1.fit_transform(full_data['hpg_area_name'].astype(str))

full_data['air_area_name_int'] = encoder1.fit_transform(full_data['air_area_name'].astype(str))

full_data['hpg_genre_name_int'] = encoder1.fit_transform(full_data['hpg_genre_name'].astype(str))

full_data['Area_int'] = encoder1.fit_transform(full_data['Area'].astype(str))

full_data['city_int'] = encoder1.fit_transform(full_data['city'].astype(str))

full_data['street_int'] = encoder1.fit_transform(full_data['street'].astype(str))

full_data['genre-area_int'] = encoder1.fit_transform(full_data['genre-area'].astype(str))

full_data['store-weekday_int'] = encoder1.fit_transform(full_data['store-weekday'].astype(str))

full_data['store-weekday-holiday_int'] = encoder1.fit_transform(full_data['store-weekday-holiday'].astype(str))


# In[102]:


full_data


# In[103]:


full_data[cat_to_int_cols]


# In[104]:


full_data.shape


# # Splitting the dataset after encoding
# 

# In[105]:


historic = full_data.query('dataset=="past"')
future  = full_data.query('dataset=="future"')

tsel = historic.visit_date < '2017-02-01'
X_train = historic[tsel]
y_train = historic[tsel][targetvariable].apply(np.log1p)

X_train = X_train[numericcolumns + binary_columns + cat_to_int_cols]


vsel = historic.visit_date >= '2017-02-01'
X_val = historic[vsel][numericcolumns + binary_columns + cat_to_int_cols]
y_val = historic[vsel][targetvariable].apply(np.log1p)


# In[106]:


print(X_val.shape)
print(y_val.shape)


# In[107]:


print(X_train.shape)
print(y_train.shape)


# # Trying out Xgboost , RandomForest and KNN Regression models
# 
# 

# # KNeighborsRegressor

# In[108]:


X_train.isnull().any()
X_train['numofreserves'] = X_train['numofreserves'].fillna(0)
X_train['numofreservevisitors'] = X_train['numofreservevisitors'].fillna(0)
X_train['bufferhoursavg'] = X_train['bufferhoursavg'].fillna(0)


# In[109]:


X_val['numofreserves'] = X_val['numofreserves'].fillna(0)
X_val['numofreservevisitors'] = X_val['numofreservevisitors'].fillna(0)
X_val['bufferhoursavg'] = X_val['bufferhoursavg'].fillna(0)


# In[110]:



Knn = neighbors.KNeighborsRegressor()
Knn.fit(X_train, y_train)  
pred3 = Knn.predict(X_val)

RMSLE3 = metrics.mean_squared_error(y_val, pred3)**0.5
print('RMLSE value of the KNN Model is:', RMSLE3)


# # XGB 

# In[111]:



from sklearn.metrics import mean_absolute_error
dtrainmatrix = xgb.DMatrix(X_train, label=y_train)

dvalmatrix = xgb.DMatrix(X_val, label=y_val)

evalset = [ (dtrainmatrix, 'train'), (dvalmatrix, 'eval') ]


# In[112]:



mean_train = np.mean(y_train)
baseline_predictions = np.ones(y_val.shape) * mean_train
mae_baseline = mean_absolute_error(y_val, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))


# # Trying out XGB with some defalut parametric values
# 

# In[113]:


random_parameters = {'colsample_bytree': 0.5,
                  'eta': 0.11,
                  'gamma': 8,
                  'max_depth': 3,
                  'min_child_weight': 80,
                  'objective': 'reg:linear',
                  'seed': 2018,
                  'subsample': 1}


# In[114]:


trial_model = xgb.train(params=random_parameters ,
                  dtrain=dtrainmatrix, 
                  num_boost_round=1000, 
                  evals=evalset,
                  early_stopping_rounds=60,
                  verbose_eval=100
                 )
bestit = trial_model.best_iteration
bestsc = trial_model.best_score


# # Checking for the best parameters using BayesianOptimization

# In[ ]:





# In[116]:


def xgboptimimum(min_child_weight, colsample_bytree, max_depth, subsample, gamma):
    params = {
        'objective': 'reg:linear',
        'eta': 0.1,
        'seed': 2018,
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'colsample_bytree': colsample_bytree,
        'subsample': subsample,
        'gamma': gamma
    }
    
    m1 = xgb.train(params=params, 
                  dtrain=dtrainmatrix, 
                  num_boost_round=1000, 
                  evals=evalset,
                  early_stopping_rounds=60,
                  verbose_eval=False
                 )
    best_iter = m1.best_iteration
    best_score= m1.best_score
    return -best_score


# In[117]:


ByesOptXgb = BayesianOptimization(f=xgboptimimum, 
                                    pbounds={
                                        'max_depth': (2,13),
                                        'min_child_weight': (70, 130),
                                        'colsample_bytree': (0.2, 0.9),
                                        'subsample': (0, 1),
                                        'gamma': (0, 7)
                                    })


# In[118]:


ByesOptXgb .maximize(init_points=7, n_iter=30)


# In[119]:


ByesOptXgb.res


# In[120]:


bestparams = ByesOptXgb.max['params']


# In[121]:


print(bestparams)


# In[122]:



bestparams['max_depth']= int(bestparams['max_depth'])
bestparams['min_child_weight']= int(bestparams['min_child_weight'])


# In[123]:


print(bestparams)


# In[124]:


modifiedmodel = xgb.train(params=bestparams ,
                  dtrain=dtrainmatrix, 
                  num_boost_round=1000, 
                  evals=evalset,
                  early_stopping_rounds=60,
                  verbose_eval=100
                 )


# In[125]:


X_val[targetvariable] = modifiedmodel.predict(xgb.DMatrix(X_val[numericcolumns + binary_columns + cat_to_int_cols]))


# In[126]:



RMSLE1 = metrics.mean_squared_error(y_val, X_val[targetvariable])**0.5


# In[127]:


print('RMSLE value of the Xgboost Model is:', RMSLE1)


# # RandomForestRegressor

# In[128]:


X_train = historic[tsel]
y_train = historic[tsel][targetvariable].apply(np.log1p)

X_train = X_train[numericcolumns + binary_columns + cat_to_int_cols]
vsel = historic.visit_date >= '2017-02-01'
X_val = historic[vsel][numericcolumns + binary_columns + cat_to_int_cols]
y_val = historic[vsel][targetvariable].apply(np.log1p)


# #For random forest the hyperparmeters tuning was done using Grid search

# #It took very long time, henec using the selected parameters directly

# In[129]:


#from sklearn.model_selection import GridSearchCV

# parameters for GridSearchCV
#randomforest_param = {"n_estimators": [15, 25, 30],
#              "max_depth": [3, 5],
#              "min_samples_split": [15, 20],
#              "min_samples_leaf": [5, 10, 20],
#              "max_leaf_nodes": [20, 40],
#              "min_weight_fraction_leaf": [0.1]}
#rfmodel_gridsearch = GridSearchCV(model2, param_grid=param_grid2)
#rfmodel_gridsearch.fit(X_train, y_train )


# In[130]:



model2 = ensemble.RandomForestRegressor(n_estimators=25, random_state=3, max_depth=20, min_weight_fraction_leaf=0.0002)
X_train.isnull().any()
X_train['numofreserves'] = X_train['numofreserves'].fillna(0)
X_train['numofreservevisitors'] = X_train['numofreservevisitors'].fillna(0)
X_train['bufferhoursavg'] = X_train['bufferhoursavg'].fillna(0)
model2.fit(X_train, y_train)

X_val['numofreserves'] = X_val['numofreserves'].fillna(0)
X_val['numofreservevisitors'] = X_val['numofreservevisitors'].fillna(0)
X_val['bufferhoursavg'] = X_val['bufferhoursavg'].fillna(0)
preds2 = model2.predict(X_val)
RMSLE2 = metrics.mean_squared_error(y_val, preds2)**0.5
print('RMSLE value of the RANDOMFOREST Model is:', RMSLE2)


# In[131]:


RMSLE_Values = {'MODEL': ['XGBOOST','RANDOMFOREST','KNN'],
        'RMSLE': [RMSLE1,RMSLE2,RMSLE3]}

df_RMSLE_Values = pd.DataFrame(RMSLE_Values , columns = ['MODEL', 'RMSLE'])

print(df_RMSLE_Values)


# 
# # Xgboost performs better- XGB is the final model
# 

# In[132]:


# training the model on the entire historic data

X_train = historic[numericcolumns + binary_columns + cat_to_int_cols]
y_train = historic[targetvariable].apply(np.log1p)    # apply np.log1p() (log(1+x)) to visitors count, to correct for high skewness

print('Training set dimensions...')
print('- X_train:', X_train.shape)
print('- y_train:', y_train.shape)

final_model = xgb.train(params=bestparams, 
                  dtrain=xgb.DMatrix(X_train, label=y_train), 
                  num_boost_round=bestit*2
                 )


# In[133]:


future[targetvariable] = final_model.predict(xgb.DMatrix(future[numericcolumns + binary_columns + cat_to_int_cols]))
future[targetvariable] = future[targetvariable].apply(np.expm1).clip(lower=0.)


# In[134]:


future.head()


# In[135]:


sub = future[[idcol, targetvariable]].copy()


# In[136]:


sub


# In[137]:


sub.to_csv('C:\\Users\\vivek\\Desktop\\bigdata\\bigdata project 2\\recruit-restaurant-visitor-forecasting\\sample submission.csv', index = False)


# In[138]:


completedata = pd.concat([historic,future], axis=0,sort=False)


# In[139]:


completedata.columns


# In[140]:



xx = completedata[['visit_date', 'visitors','dataset']]
xx.head()


# # Plotting the complete data (Historic+Future)
# 

# In[141]:



series1 = xx.groupby('visit_date').sum()
ax = series1.plot(figsize=[20,10])
ax.set(xlabel="Visit_Date",
       ylabel="'Daily Visitors'",
       title='Historic + Forecast',
       xlim=["2016-01-13", "2017-05-31"])

date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




