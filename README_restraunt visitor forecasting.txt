 Notes for  Recruit restraunt forecasting
1) I have build the entire code on Ananconda-Jupyter notebook- 
2) Please ensure that the following libraries and classes are imported

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
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

example
pip install matplotlib

3) Please import the data files by changing the path in the first section (# loading all the data files into pandas data frames) of the code

example
airstores = pd.read_csv("C://Users//vivek//Desktop//bigdata//bigdata project 2//recruit-restaurant-visitor-forecasting//air_store_info.csv//air_store_info.csv")

Please update the path in the brackets

4) the code in the .py file is in the sequential order and can be run in the same sequence

5) the code takes around 30 minutes for running
6) For getting the 'submission file' please update the location

sub.to_csv('C:\\Users\\vivek\\Desktop\\bigdata\\bigdata project 2\\recruit-restaurant-visitor-forecasting\\sample submission.csv', index = False)




