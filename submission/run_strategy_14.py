

import xgboost as xgb
import numpy as np
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.combine import SMOTEENN
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

from imblearn.under_sampling import TomekLinks

def fit_cluster(df,ks=[2, 20, 50, 100, 200, 300], s_perc = 0.5):

    df_h = sample_data_frame(df,s_perc = s_perc)

    matrix = df_h.as_matrix()[:, :-1]

    transofer = QuantileTransformer(output_distribution='normal')
    matrix_data = transofer.fit_transform(matrix)

    dict_kmeans_model = {}

    for k in ks:
        clusterer = KMeans(n_clusters=k, precompute_distances=True)
        clusterer.fit(matrix_data)
        dict_kmeans_model[k] = clusterer

    return dict_kmeans_model

def sample_data_frame(df,s_perc = 0.5):
    df_happy = df.loc[df['TARGET'] == 0]
    df_unhappy = df.loc[df['TARGET'] == 1]

    df_unhappy_s = df_unhappy.sample(int(s_perc * df_unhappy.shape[0]),
                                                               random_state=21)
    df_happy_s =  df_happy.sample(df_unhappy_s.shape[0], random_state=32)

    return pd.concat([df_unhappy_s, df_happy_s])




def get_extreme_value(v,th_extreme = 3):

    unique_value = np.unique(v)
    max_value = np.argmax(unique_value)
    unique_value = np.delete(unique_value, max_value)
    IQR = np.percentile(unique_value, 0.75) - np.percentile(unique_value, 0.25)
    new_value = np.percentile(unique_value, 0.75) + (IQR * th_extreme)
    return new_value



# XGBoost params:
def get_params():
    #
    params = {}
    params["objective"] = "binary:logistic"
    params["booster"] = "gbtree"
    params["eval_metric"] = "auc"
    params["eta"] = 0.1  #
    #params["min_child_weight"] = 50
    params["subsample"] = 0.8
    params["colsample_bytree"] = 1
    params["max_depth"] = 10
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



# fix numerical attributes for var3 with extreme

new_value_var3 = get_extreme_value(data_train_copy['var3'].as_matrix())
data_train_copy['var3'].replace(to_replace=-999999,value=new_value_var3,inplace=True)

# fix the 9999999999 values with extreme

l_999=['delta_imp_aport_var13_1y3', 'delta_imp_aport_var17_1y3', 'delta_imp_aport_var33_1y3',
   'delta_imp_compra_var44_1y3', 'delta_imp_reemb_var17_1y3', 'delta_imp_trasp_var17_in_1y3',
   'delta_imp_trasp_var33_in_1y3', 'delta_imp_venta_var44_1y3', 'delta_num_aport_var13_1y3',
   'delta_num_aport_var17_1y3', 'delta_num_aport_var33_1y3', 'delta_num_compra_var44_1y3', 'delta_num_reemb_var17_1y3',
   'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_venta_var44_1y3']

dic_new_values = {}

for colum_to_replace in l_999:
    new_value = get_extreme_value(data_train_copy[colum_to_replace].as_matrix())
    dic_new_values[colum_to_replace] = new_value
    data_train_copy[colum_to_replace].replace(to_replace=9999999999,value=new_value,inplace=True)



#remove row with all zero and dublicated
data_train_copy.loc[(data_train_copy!=0).any(axis=1)]
data_train_copy.drop_duplicates(inplace=True)


# scale vaalues with quantile

matrix_data = data_train_copy.as_matrix()[:,:-1]

transofer = QuantileTransformer(output_distribution='normal')
matrix_data = transofer.fit_transform(matrix_data)


# create cluster models on a homogenous samples

dict_kmeans_model = fit_cluster(data_train_copy,ks=[2, 50, 100, 200, 300],s_perc = 1)


# recover the cluster predictions from the model

l_clusters_kmeans_numeric = []

for k in [2, 50, 100, 200, 300]:
    cluster_labels_kmeans=dict_kmeans_model[k].predict(matrix_data)
    l_clusters_kmeans_numeric.append(cluster_labels_kmeans.reshape(-1,1))

matrix_cluster_kmeans = np.hstack(l_clusters_kmeans_numeric)

matrix_data = np.hstack([matrix_data,matrix_cluster_kmeans])



# undersampling

ada =  TomekLinks()

label = data_train_copy['TARGET'].as_matrix()

matrix_data, label = ada.fit_sample(matrix_data, label)


# reduce with pca

pca = PCA(n_components=100)

matrix_data = pca.fit_transform(matrix_data)




X_train, X_val, y_train, y_val = train_test_split(matrix_data, label, test_size=0.4)


# the parameters have been downloaded from here https://www.kaggle.com/c/santander-customer-satisfaction/discussion/19330
# fit model

xgtrain = xgb.DMatrix(X_train, label=y_train.tolist())
xgval = xgb.DMatrix(X_val, label=y_val.tolist())


# Number of boosting iterations.
num_round = 400

evallist = [(xgtrain, 'train'), (xgval, 'val')]

model = xgb.train(dtrain=xgtrain, evals=evallist, params=get_params(), num_boost_round=num_round, early_stopping_rounds=45000)



#load test data
data_test = pd.read_csv('data/test2.csv',index_col=0)

data_test = data_test.drop(list_attributes_constant,axis=1)
data_test = data_test.drop(col,axis=1)

data_test['var3'].replace(to_replace=-999999,value=new_value_var3,inplace=True)

for colum_to_replace in l_999:
    data_test[colum_to_replace].replace(to_replace=9999999999,value=dic_new_values[colum_to_replace],inplace=True)



matrix_test = data_test.as_matrix()

matrix_test = transofer.transform(matrix_test)

# recover the cluster predictions from the model

l_clusters_kmeans_numeric = []

for k in [2, 50, 100, 200, 300]:
    cluster_labels_kmeans=dict_kmeans_model[k].predict(matrix_test)
    l_clusters_kmeans_numeric.append(cluster_labels_kmeans.reshape(-1,1))

matrix_cluster_kmeans = np.hstack(l_clusters_kmeans_numeric)

matrix_test = np.hstack([matrix_test,matrix_cluster_kmeans])


matrix_test = pca.transform(matrix_test)

xgtest = xgb.DMatrix(matrix_test)


classes_predicted = model.predict(xgtest, ntree_limit=model.best_iteration)


# Make Submission
test_aux = pd.read_csv('data/test2.csv')
result = pd.DataFrame({"Id": test_aux["ID"], 'TARGET': classes_predicted})

result.to_csv("sub_strategy_14.csv", index=False)


