


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV

from boruta import boruta_py

import pandas as pd
import matplotlib.pyplot as plt
import copy
from time import time

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from caimcaim import CAIMD


# load data
data_train = pd.read_csv('data/train2.csv',index_col=0)

# copy where we do all the processing
data_train_copy = data_train.copy()

# remove constant attribute

data_train_unique_values = data_train.apply(pd.Series.nunique)

series_col_unique  = data_train_unique_values.where(data_train_unique_values==1,other=0)
list_attributes_constant = series_col_unique[series_col_unique==1].index.tolist()

data_train_copy = data_train_copy.drop(list_attributes_constant,axis=1)



# removing binary feature

filtering_binary =data_train.apply(pd.Series.nunique) ==2
filtering_binary['TARGET']=False
col=data_train.loc[:,filtering_binary].columns.tolist()
data_train_copy = data_train_copy.drop(col,axis=1)


# oversampling

ada =  SMOTEENN()

matrix_data = data_train_copy.as_matrix()[:,:-1]

label = data_train_copy['TARGET'].as_matrix()

matrix_data, label = ada.fit_sample(matrix_data, label)



classifier = RandomForestClassifier(max_features=11,
                                        n_estimators=114,
                                        criterion= 'gini',
                                        bootstrap=False,
                                        min_samples_split= 3,
                                        min_samples_leaf=4,
                                        class_weight='balanced',
                                        max_depth=None)

classifier.fit(matrix_data, label)



#load test data
data_test = pd.read_csv('data/test2.csv',index_col=0)

data_test = data_test.drop(list_attributes_constant,axis=1)
data_test = data_test.drop(col,axis=1)

matrix_test = data_test.as_matrix()

classes_predicted = classifier.predict(matrix_test)

index = data_test.index.values

result = np.hstack([index.reshape(-1,1),classes_predicted.reshape(-1,1)])


data_frame_result = pd.DataFrame(result,columns=['ID','TARGET'])

data_frame_result.to_csv('sub_strategy_3.csv',index=None)



















