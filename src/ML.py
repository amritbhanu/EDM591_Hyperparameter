from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import f1_score, mean_squared_error

# max_features, min_samples_split, max_depth, min_samples_leaf, min_impurity_split is threshold
# bounds (0.01,1), (2,20), (1,50), (1,20), (0,1)
# "continuous", "continuous", "integer", "continuous","continuous"
def run_dectreeclas(k,train_data,train_labels,test_data,test_labels):

    model = DecisionTreeClassifier(**k).fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return f1_score(test_labels, prediction, average='micro')


def run_dectreereg(k,train_data,train_labels,test_data,test_labels):
    model = DecisionTreeRegressor(**k).fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return mean_squared_error(test_labels, prediction)



# max_features, min_samples_split, max_leaf_nodes, min_samples_leaf, min_impurity_split is threshold, n_estimators
# bounds (0.01,1), (2,20), (1,50), (1,20), (0,1), (50,100)
# "continuous", "continuous", "integer", "continuous","continuous", "integer"
def run_rfclas(k,train_data,train_labels,test_data,test_labels):
    model = RandomForestClassifier(**k).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction, average='micro')

def run_rfreg(k,train_data,train_labels,test_data,test_labels):
    model = RandomForestRegressor(**k).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return mean_squared_error(test_labels, prediction)


# C, kernel, degree
# bounds (0.1,100), ["linear","poly","rbf","sigmoid"], (1,20)
# "continuous", "categorical", "integer"
def run_svmclas(k,train_data,train_labels,test_data,test_labels):
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    train_data = scaling.transform(train_data)
    test_data = scaling.transform(test_data)
    model = SVC(cache_size=20000,**k).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction, average='micro')

def run_svmreg(k,train_data,train_labels,test_data,test_labels):
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    train_data = scaling.transform(train_data)
    test_data = scaling.transform(test_data)
    model = SVR(cache_size=20000,**k).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return mean_squared_error(test_labels, prediction)