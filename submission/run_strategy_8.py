

import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing.data import QuantileTransformer
from boruta import boruta_py
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import copy
from time import time

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from caimcaim import CAIMD






# XGBoost params:
def get_params():
    #
    params = {}
    params["objective"] = "binary:logistic"
    params["booster"] = "gbtree"
    params["eval_metric"] = "auc"
    params["eta"] = 0.01  #
    #params["min_child_weight"] = 50
    params["subsample"] = 0.50
    params["colsample_bytree"] = 0.8
    params["max_depth"] = 40
    params["nthread"] = 8
    params["seed"] = 8000
    plst = list(params.items())
    #
    return plst

# load data
data_train = pd.read_csv('data/train2.csv',index_col=0)

# copy where we do all the processing
data_train_copy = data_train.copy()

# remove constant attribute

data_train_unique_values = data_train.apply(pd.Series.nunique)

series_col_unique  = data_train_unique_values.where(data_train_unique_values==1,other=0)
list_attributes_constant = series_col_unique[series_col_unique==1].index.tolist()

data_train_copy = data_train_copy.drop(list_attributes_constant,axis=1)

# remove dubplicates
data_train_copy.drop_duplicates(inplace=True)

# removing binary feature

filtering_binary =data_train.apply(pd.Series.nunique) ==2
filtering_binary['TARGET']=False
col=data_train.loc[:,filtering_binary].columns.tolist()
data_train_copy = data_train_copy.drop(col,axis=1)



# fix numerical attributes for var3


median_valuemedian = data_train_copy['var3'].median()
data_train_copy['var3'].replace(to_replace=-999999,value=median_valuemedian,inplace=True)

# fix the 9999999999 values

l_999=['delta_imp_aport_var13_1y3', 'delta_imp_aport_var17_1y3', 'delta_imp_aport_var33_1y3',
   'delta_imp_compra_var44_1y3', 'delta_imp_reemb_var17_1y3', 'delta_imp_trasp_var17_in_1y3',
   'delta_imp_trasp_var33_in_1y3', 'delta_imp_venta_var44_1y3', 'delta_num_aport_var13_1y3',
   'delta_num_aport_var17_1y3', 'delta_num_aport_var33_1y3', 'delta_num_compra_var44_1y3', 'delta_num_reemb_var17_1y3',
   'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_venta_var44_1y3']


for colum_to_replace in l_999:
     data_train_copy[colum_to_replace].replace(to_replace=9999999999,value=0,inplace=True)



#remove row with all zero and dublicated
data_train_copy.loc[(data_train_copy!=0).any(axis=1)]
data_train_copy.drop_duplicates(inplace=True)



# scale vaalues with quantile

matrix_data = data_train_copy.as_matrix()[:,:-1]

transofer=QuantileTransformer(output_distribution='normal')
matrix_data=transofer.fit_transform(matrix_data)


# reduce with pca

pca = PCA(n_components=50)

matrix_data = pca.fit_transform(matrix_data)

# oversampling

ada =  SMOTEENN()

label = data_train_copy['TARGET'].as_matrix()

matrix_data, label = ada.fit_sample(matrix_data, label)


X_train, X_val, y_train, y_val = train_test_split(matrix_data, label, test_size=0.2)


# the parameters have been downloaded from here https://www.kaggle.com/c/santander-customer-satisfaction/discussion/19330
# fit model

xgtrain = xgb.DMatrix(X_train, label=y_train.tolist())
xgval = xgb.DMatrix(X_val, label=y_val.tolist())


# Number of boosting iterations.
num_round = 10000

evallist = [(xgtrain, 'train'), (xgval, 'val')]

model = xgb.train(dtrain=xgtrain, evals=evallist, params=get_params(), num_boost_round=num_round, early_stopping_rounds=45000)



#load test data
data_test = pd.read_csv('data/test2.csv',index_col=0)

data_test = data_test.drop(list_attributes_constant,axis=1)
data_test = data_test.drop(col,axis=1)

data_test['var3'].replace(to_replace=-999999,value=median_valuemedian,inplace=True)

for colum_to_replace in l_999:
    data_test[colum_to_replace].replace(to_replace=9999999999,value=0,inplace=True)



matrix_test = data_test.as_matrix()

matrix_test = transofer.transform(matrix_test)

matrix_test = pca.transform(matrix_test)

xgtest = xgb.DMatrix(matrix_test)


classes_predicted = model.predict(xgtest, ntree_limit=model.best_iteration)


# Make Submission
test_aux = pd.read_csv('data/test2.csv')
result = pd.DataFrame({"Id": test_aux["ID"], 'TARGET': classes_predicted})

result.to_csv("sub_strategy_8.csv", index=False)


