from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
    

def random_forest_classifer_params(matrix_feature,labels,number_rounds = 3,test_size_value = 0.5, n_estimators= 114,
                    min_samples_split= 3,min_samples_leaf= 4,max_features= 11,
                    max_depth=None, criterion= 'gini',bootstrap=False):


    classifier = RandomForestClassifier(max_features=max_features,
                                        n_estimators=n_estimators,
                                        criterion= criterion,
                                        bootstrap=bootstrap,
                                        min_samples_split= min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        class_weight='balanced',
                                        max_depth=max_depth)

    f1_score_list = []
    cm_list = []
    cm_max_list = []

    rounds = StratifiedShuffleSplit(n_splits=number_rounds, 
                                    test_size=test_size_value,
                                    random_state=0)

    for train_index, test_index in rounds.split(matrix_feature,labels):
            matrix_train = matrix_feature[train_index]
            classes_train = labels[train_index]
            matrix_test = matrix_feature[test_index]
            classes_test = labels[test_index]
            classifier.fit(matrix_train,classes_train)
            classes_predicted = classifier.predict(matrix_test)
            cm = confusion_matrix(classes_test, classes_predicted,labels=[0,1])
            # normalize the confusion matrix
            cm= cm / cm.astype(np.float).sum(axis=1).reshape(-1,1)
            #this threshold is used for plotting 
            thresh = cm.max() / 2.
            cm_list.append(cm)
            cm_max_list.append(thresh)
            #I compute the score taking into account the weighted schema, 
            #which means weighted by the support (the number of true instances for each label)
            precision, recall, fscore, support = score(classes_test, 
                                                       classes_predicted,
                                                       average='macro')
            f1_score_list.append(fscore)

    return f1_score_list,cm_list





# I am using a standard setting, I can use the second method when the parameters are tunned.
   
def random_forest_classifer(matrix_feature,labels,number_rounds = 3,test_size_value = 0.5,number_trees = 100):

    classifier = RandomForestClassifier(n_estimators=number_trees, 
                                        max_features="sqrt",class_weight='balanced')

    f1_score_list = []
    cm_list = []
    cm_max_list = []

    rounds = StratifiedShuffleSplit(n_splits=number_rounds, 
                                    test_size=test_size_value,
                                    random_state=0)

    for train_index, test_index in rounds.split(matrix_feature,labels):
            matrix_train = matrix_feature[train_index]
            classes_train = labels[train_index]
            matrix_test = matrix_feature[test_index]
            classes_test = labels[test_index]
            classifier.fit(matrix_train,classes_train)
            classes_predicted = classifier.predict(matrix_test)
            cm = confusion_matrix(classes_test, classes_predicted,labels=[0,1])
            # normalize the confusion matrix
            cm= cm / cm.astype(np.float).sum(axis=1).reshape(-1,1)
            #this threshold is used for plotting 
            thresh = cm.max() / 2.
            cm_list.append(cm)
            cm_max_list.append(thresh)
            #I compute the score taking into account the weighted schema, 
            #which means weighted by the support (the number of true instances for each label)
            precision, recall, fscore, support = score(classes_test, 
                                                       classes_predicted,
                                                       average='macro')
            f1_score_list.append(fscore)

    return f1_score_list,cm_list

