from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd
from random import seed, shuffle
from DE import DE
from ML import *
import time
import os
from demos import cmd
from sklearn.model_selection import StratifiedKFold
import pickle
from collections import OrderedDict

# dataset1_math=regression
# dataset1_portuguese=regression
# dataset2=classification
# dataset3=classification

learners_class=[run_dectreeclas,run_rfclas,run_svmclas]
learners_reg=[run_dectreereg, run_rfreg,run_svmreg]
learners_para_dic=[OrderedDict([("min_samples_split",2),("min_impurity_decrease",0.0), ("max_depth",None),
                               ("min_samples_leaf", 1)]), OrderedDict([("min_samples_split",2),
                                ("max_leaf_nodes",None), ("min_samples_leaf",1), ("min_impurity_decrease",0.0),("n_estimators",10)]),
                   OrderedDict([("C", 1.0), ("kernel", 'rbf'),
                                ("degree", 3)])]
learners_para_bounds=[[(2,20), (0,1), (1,20), (1,50)],[ (2,20), (2,50), (1,20), (0,1), (10,100)],
                      [(0.1,100), ("linear","poly","rbf","sigmoid"), (1,20)]]
learners_para_categories=[[ "integer", "continuous", "integer", "integer"],["integer", "integer", "integer","continuous", "integer"],
                          ["continuous", "categorical", "integer"]]

metrics=['accuracy','recall','precision','false_alarm','times','features']
ROOT=os.getcwd()

def late(corpus,labels,ranges,class_flag,res,metric,columns):
    temp = {}
    if class_flag:
        for num, i in enumerate(learners_class):
            temp[i.__name__] = {}
            temp['cols']=columns
            for m in metrics:
                temp[i.__name__][m] = []
            for _ in range(5):
                shuffle(ranges)
                corpus,labels=corpus[ranges],labels[ranges]
                skf = StratifiedKFold(n_splits=5)
                for train_index, test_index in skf.split(corpus, labels):
                    train_data, train_labels = corpus[train_index], labels[train_index]
                    test_data, test_labels = corpus[test_index], labels[test_index]

                    skf1 = StratifiedKFold(n_splits=5)
                    train_, vali_ = list(skf1.split(train_data, train_labels))[0]
                    training_data, training_labels = train_data[train_], train_labels[train_]
                    vali_data, vali_labels = train_data[vali_], train_labels[vali_]

                    start_time = time.time()
                    if metric=='false_alarm':
                        de = DE(Goal="Min",termination="Late", NP=30)
                    else:
                        de = DE(termination="Late",NP=30)
                    v, _ = de.solve(learners_class[num], OrderedDict(learners_para_dic[num]),
                                         learners_para_bounds[num], learners_para_categories[num], training_data
                                         , training_labels, vali_data, vali_labels, metric)
                    temp[i.__name__]['times'].append(time.time() - start_time)

                    _,val = learners_class[num](v.ind, train_data, train_labels, test_data, test_labels, metric)

                    for x in val[0]:
                        temp[i.__name__][x].append(val[0][x])
                    temp[i.__name__]['features'].append(val[1])
    else:
        for num, i in enumerate(learners_reg):
            temp[i.__name__] = {}
            temp['cols'] = columns
            for m in metrics[-2:]+['MSE']:
                temp[i.__name__][m] = []
            for _ in range(5):
                for k in range(5):
                    shuffle(ranges)
                    train_data, train_labels, test_data, test_labels = corpus[ranges[:int(0.8 * len(ranges))]] \
                        , labels[ranges[:int(0.8 * len(ranges))]] \
                        , corpus[ranges[int(0.8 * len(ranges)):]], labels[ranges[int(0.8 * len(ranges)):]]

                    ran = range(0, len(train_labels))
                    training_data, training_labels, vali_data, vali_labels = train_data[ran[:int(0.8 * len(ran))]] \
                        , train_labels[ran[:int(0.8 * len(ran))]] \
                        , train_data[ran[int(0.8 * len(ran)):]], train_labels[ran[int(0.8 * len(ran)):]]

                    start_time = time.time()
                    de = DE(Goal="Min", termination="Late",NP=30)
                    v, _ = de.solve(learners_reg[num], OrderedDict(learners_para_dic[num]),
                                         learners_para_bounds[num], learners_para_categories[num], training_data
                                         , training_labels, vali_data, vali_labels)
                    temp[i.__name__]['times'].append(time.time() - start_time)
                    t, val = learners_reg[num](v.ind, train_data, train_labels, test_data, test_labels)

                    temp[i.__name__]['MSE'].append(t)
                    temp[i.__name__]['features'].append(val[1])
    with open('../dump/'+metric+res+ '_late.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

def early(corpus,labels,ranges,class_flag,res,metric,columns):
    temp = {}
    if class_flag:
        for num, i in enumerate(learners_class):
            temp[i.__name__] = {}
            temp['cols']=columns
            for m in metrics:
                temp[i.__name__][m] = []
            for _ in range(5):
                shuffle(ranges)
                corpus,labels=corpus[ranges],labels[ranges]
                skf = StratifiedKFold(n_splits=5)
                for train_index, test_index in skf.split(corpus, labels):
                    train_data, train_labels = corpus[train_index], labels[train_index]
                    test_data, test_labels = corpus[test_index], labels[test_index]

                    skf1 = StratifiedKFold(n_splits=5)
                    train_, vali_= list(skf1.split(train_data, train_labels))[0]
                    training_data, training_labels = train_data[train_], train_labels[train_]
                    vali_data, vali_labels = train_data[vali_], train_labels[vali_]

                    start_time = time.time()
                    if metric=='false_alarm':
                        de = DE(Goal="Min",GEN=5,NP=30)
                    else:
                        de = DE(GEN=5,NP=30)
                    v, _ = de.solve(learners_class[num], OrderedDict(learners_para_dic[num]),
                                         learners_para_bounds[num], learners_para_categories[num], training_data
                                         , training_labels, vali_data, vali_labels,metric)
                    temp[i.__name__]['times'].append(time.time() - start_time)

                    _, val = learners_class[num](v.ind, train_data, train_labels, test_data, test_labels, metric)

                    for x in val[0]:
                        temp[i.__name__][x].append(val[0][x])
                    temp[i.__name__]['features'].append(val[1])
    else:
        for num, i in enumerate(learners_reg):
            temp[i.__name__] = {}
            temp['cols'] = columns
            for m in metrics[-2:] + ['MSE']:
                temp[i.__name__][m] = []
            for _ in range(5):
                for k in range(5):
                    shuffle(ranges)
                    train_data, train_labels, test_data, test_labels = corpus[ranges[:int(0.8 * len(ranges))]] \
                        , labels[ranges[:int(0.8 * len(ranges))]] \
                        , corpus[ranges[int(0.8 * len(ranges)):]], labels[ranges[int(0.8 * len(ranges)):]]

                    ran=range(0, len(train_labels))
                    training_data, training_labels, vali_data, vali_labels = train_data[ran[:int(0.8 * len(ran))]] \
                        , train_labels[ran[:int(0.8 * len(ran))]] \
                        , train_data[ran[int(0.8 * len(ran)):]], train_labels[ran[int(0.8 * len(ran)):]]

                    start_time = time.time()
                    de = DE(Goal="Min", GEN=5,NP=30)
                    v, _ = de.solve(learners_reg[num], OrderedDict(learners_para_dic[num]),
                                         learners_para_bounds[num], learners_para_categories[num], training_data
                                         , training_labels, vali_data, vali_labels)
                    temp[i.__name__]['times'].append(time.time() - start_time)
                    t, val = learners_reg[num](v.ind, train_data, train_labels, test_data, test_labels)

                    temp[i.__name__]['MSE'].append(t)
                    temp[i.__name__]['features'].append(val[1])

    with open('../dump/' +metric+res+ '_early.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

def _test(res=''):

    seed(1)
    np.random.seed(1)
    file_names=[]
    for _, _, files in os.walk(ROOT + "/../data/preprocessed_data/"):
        for file in files:
            if file.startswith(res):
                file_names.append(file.split(".csv")[0])
    if res=='dataset1_math' or res=='dataset1_portuguese':
        class_flag=False
        for f in file_names:
            df = pd.read_csv("../data/preprocessed_data/" + f + ".csv")
            corpus, labels = np.array(df[df.columns[:-1]].values.tolist()), np.array(df[df.columns[-1]].values.tolist())
            ranges = range(0, len(labels))
            early(corpus, labels, ranges, class_flag, f, '', df.columns[:-1].values.tolist())
            late(corpus, labels, ranges, class_flag, f, '', df.columns[:-1].values.tolist())
    else:
        class_flag=True
        for metric in metrics:
            for f in file_names:
                df = pd.read_csv("../data/preprocessed_data/" + f + ".csv")
                corpus, labels = np.array(df[df.columns[:-1]].values.tolist()), np.array(df[df.columns[-1]].values.tolist())
                ranges = range(0, len(labels))
                early(corpus, labels, ranges, class_flag, f, metric, df.columns[:-1].values.tolist())
                late(corpus, labels, ranges, class_flag, f, metric, df.columns[:-1].values.tolist())



if __name__ == '__main__':
    eval(cmd ())