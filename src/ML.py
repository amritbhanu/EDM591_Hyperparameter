from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import f1_score, mean_squared_error,roc_auc_score,precision_score,recall_score,accuracy_score,confusion_matrix

# max_features, min_samples_split, max_depth, min_samples_leaf, min_impurity_split is threshold
# bounds (0.01,1), (2,20), (1,50), (1,20), (0,1)
# "continuous", "continuous", "integer", "continuous","continuous"
def run_dectreeclas(k,train_data,train_labels,test_data,test_labels, metric):
    model = DecisionTreeClassifier(**k)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return evaluation(metric, prediction, test_labels), [model.feature_importances_]


def run_dectreereg(k,train_data,train_labels,test_data,test_labels):
    model = DecisionTreeRegressor(**k)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return mean_squared_error(test_labels, prediction), [model.feature_importances_]



# max_features, min_samples_split, max_leaf_nodes, min_samples_leaf, min_impurity_split is threshold, n_estimators
# bounds (0.01,1), (2,20), (1,50), (1,20), (0,1), (50,100)
# "continuous", "continuous", "integer", "continuous","continuous", "integer"
def run_rfclas(k,train_data,train_labels,test_data,test_labels, metric):
    model = RandomForestClassifier(**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return evaluation(metric, prediction, test_labels), [model.feature_importances_]

def run_rfreg(k,train_data,train_labels,test_data,test_labels):
    model = RandomForestRegressor(**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return mean_squared_error(test_labels, prediction), [model.feature_importances_]


# C, kernel, degree
# bounds (0.1,100), ["linear","poly","rbf","sigmoid"], (1,20)
# "continuous", "categorical", "integer"
def run_svmclas(k,train_data,train_labels,test_data,test_labels, metric):
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    train_data = scaling.transform(train_data)
    test_data = scaling.transform(test_data)
    model = SVC(cache_size=20000,**k)
    model.fit(train_data, train_labels)
    #print(model.coef_)
    prediction = model.predict(test_data)
    return evaluation(metric,prediction,test_labels),[]

def run_svmreg(k,train_data,train_labels,test_data,test_labels):
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    train_data = scaling.transform(train_data)
    test_data = scaling.transform(test_data)
    model = SVR(cache_size=20000,**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return mean_squared_error(test_labels, prediction),[]

def evaluation(measure, prediction, test_labels):
    if measure == "accuracy":
        return accuracy_score(test_labels, prediction)
    if measure == "recall":
        return recall_score(test_labels, prediction, average='micro')
    if measure == "precision":
        return precision_score(test_labels, prediction, average='micro')
    if measure == "false_alarm":
        tn, fp, fn, tp = confusion_matrix(test_labels, prediction).ravel()
        return float(fp)/(fp+tn)
    if measure == "f1":
        return f1_score(test_labels, prediction, average='micro')
    if measure == "auc":
        return roc_auc_score(test_labels, prediction, average='micro')
