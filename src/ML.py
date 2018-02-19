from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import f1_score, mean_squared_error,roc_curve,auc,accuracy_score,confusion_matrix
import numpy as np
from sklearn.preprocessing import label_binarize

metrics=['accuracy','recall','precision','false_alarm']

# max_features, min_samples_split, max_depth, min_samples_leaf, min_impurity_split is threshold
# bounds (0.01,1), (2,20), (1,50), (1,20), (0,1)
# "continuous", "continuous", "integer", "continuous","continuous"
def run_dectreeclas(k,train_data,train_labels,test_data,test_labels, metric):

    model = DecisionTreeClassifier(**k)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    dic = {}

    for i in metrics:
        dic[i] = evaluation(i, prediction, test_labels)

    return dic[metric], [dic, model.feature_importances_]


def run_dectreereg(k,train_data,train_labels,test_data,test_labels):
    model = DecisionTreeRegressor(**k)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return mean_squared_error(test_labels, prediction), [{}, model.feature_importances_]



# max_features, min_samples_split, max_leaf_nodes, min_samples_leaf, min_impurity_split is threshold, n_estimators
# bounds (0.01,1), (2,20), (1,50), (1,20), (0,1), (50,100)
# "continuous", "continuous", "integer", "continuous","continuous", "integer"
def run_rfclas(k,train_data,train_labels,test_data,test_labels, metric):
    model = RandomForestClassifier(**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    dic = {}
    for i in metrics:
        dic[i] = evaluation(i, prediction, test_labels)
    return dic[metric], [dic, model.feature_importances_]

def run_rfreg(k,train_data,train_labels,test_data,test_labels):
    model = RandomForestRegressor(**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return mean_squared_error(test_labels, prediction), [{},model.feature_importances_]


# C, kernel, degree
# bounds (0.1,100), ["linear","poly","rbf","sigmoid"], (1,20)
# "continuous", "categorical", "integer"
def run_svmclas(k,train_data,train_labels,test_data,test_labels, metric):
    # from sklearn.preprocessing import MinMaxScaler
    # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    # train_data = scaling.transform(train_data)
    # test_data = scaling.transform(test_data)
    model = SVC(cache_size=20000,**k)
    model.fit(train_data, train_labels)
    #print(model.coef_)
    prediction = model.predict(test_data)
    dic = {}
    for i in metrics:
        dic[i] = evaluation(i, prediction, test_labels)
    return dic[metric], [dic, []]

def run_svmreg(k,train_data,train_labels,test_data,test_labels):
    # from sklearn.preprocessing import MinMaxScaler
    # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    # train_data = scaling.transform(train_data)
    # test_data = scaling.transform(test_data)
    model = SVR(cache_size=20000,**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return mean_squared_error(test_labels, prediction),[{},[]]

def evaluation(measure, prediction, test_labels, class_target=-1):
    confu = confusion_matrix(test_labels, prediction)
    fp = confu.sum(axis=0) - np.diag(confu)
    fn = confu.sum(axis=1) - np.diag(confu)
    tp = np.diag(confu)
    tn = confu.sum() - (fp + fn + tp)
    if measure == "accuracy":
        return accuracy_score(test_labels, prediction)
    if measure == "recall":
        recall = 0
        if class_target == -1:
            for m in range(len(tp)):
                if tp[m] != 0 and (tp[m] + fn[m]) != 0:
                    recall += float(tp[m]) / (tp[m] + fn[m])
            return recall / len(tp)
        else:
            if tp[class_target] != 0 and (tp[class_target] + fn[class_target]) != 0:
                float(tp[class_target]) / (tp[class_target] + fn[class_target])
            else:
                return 0

    if measure == "precision":
        precision = 0
        if class_target == -1:
            for m in range(len(tp)):
                if tp[m] != 0 and (tp[m] + fp[m]) != 0:
                    precision += float(tp[m]) / (tp[m] + fp[m])
            return precision / len(tp)
        else:
            if tp[class_target] != 0 and (tp[class_target] + fp[class_target]) != 0:
                float(tp[class_target]) / (tp[class_target] + fp[class_target])
            else:
                return 0
    if measure == "false_alarm":
        fals=0
        if class_target==-1:
            for m in range(len(fp)):
                if fp[m] != 0 and (fp[m] + tn[m]) !=0 :
                    fals+=float(fp[m])/(fp[m]+tn[m])
            return fals/len(fp)
        else:
            if fp[class_target] != 0 and (fp[class_target] + tn[class_target]) != 0:
                float(fp[class_target]) / (fp[class_target] + tn[class_target])
            else:
                return 0

    if measure == "f1":
        if class_target==-1:
            return f1_score(test_labels, prediction, average='macro')
        else:
            return f1_score(test_labels, prediction, pos_label=class_target, average='binary')
    # if measure == "auc":
    #     ## Label binarizer
    #     test_labels=np.array(test_labels)
    #     prediction=np.array(prediction)
    #     y = label_binarize(test_labels, classes=range(len(tp)))
    #
    #     n_classes = y.shape[1]
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     print(test_labels)
    #     for i in range(n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], prediction[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #     print(fpr)
    #     # Compute micro-average ROC curve and ROC area
    #     # fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), prediction.ravel())
    #     # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
