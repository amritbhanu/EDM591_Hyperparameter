from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd
from random import seed, shuffle
import numpy as np
from DE import DE
from ML import *
import time
from demos import cmd
from sklearn.model_selection import StratifiedKFold
import pickle
from collections import OrderedDict

# dataset1=regression
# dataset2=classification
# dataset3=classification

learners_class=[run_dectreeclas,run_rfclas,run_svmclas]
learners_reg=[run_dectreereg, run_rfreg,run_svmreg]
learners_para_dic=[OrderedDict([("max_features",1), ("min_samples_split",1),("min_impurity_split",1), ("max_depth",1),
                               ("min_samples_leaf", 1)]), OrderedDict([("max_features",1), ("min_samples_split",1),
                                ("max_leaf_nodes",1), ("min_samples_leaf",1), ("min_impurity_split",1),("n_estimators",1)]),
                   OrderedDict([("C", 1), ("kernel", 'linear'),
                                ("degree", 1)])]
learners_para_bounds=[[(0.01,1), (2,20), (0,1), (1,20), (1,50)],[(0.01,1), (2,20), (2,50), (1,20), (0,1), (50,100)],
                      [(0.1,100), ("linear","poly","rbf","sigmoid"), (1,20)]]
learners_para_categories=[["continuous", "integer", "continuous", "integer", "integer"],["continuous", "integer", "integer", "integer","continuous", "integer"],
                          ["continuous", "categorical", "integer"]]


def _test(res=''):
    seed(1)
    np.random.seed(1)
    df=pd.read_csv("../data/"+res+".csv")
    corpus,labels=np.array(df[df.columns[:-1]].values.tolist()),np.array(df[df.columns[-1]].values.tolist())
    ranges = range(0, len(labels))
    if res=='dataset1':
        class_flag=False
    else:
        class_flag=True
    temp={}

    for num,i in enumerate(learners_class):
        start_time = time.time()
        if class_flag:
            skf = StratifiedKFold(n_splits=10)
            l=[]
            paras=[]
            for train_index, test_index in skf.split(corpus, labels):
                train_data, train_labels = corpus[train_index], labels[train_index]
                test_data, test_labels = corpus[test_index], labels[test_index]
                de = DE(GEN=5, termination="Late")
                v,pareto=de.solve(learners_class[num],OrderedDict(learners_para_dic[num]),learners_para_bounds[num],learners_para_categories[num],train_data
                         , train_labels,test_data,test_labels)
                l.append(v.fit)
                paras.append(v.ind)
            temp[learners_class[num].__name__] = [l,paras, time.time() - start_time]
            print(l,paras, time.time() - start_time)
        else:
            l=[]
            paras = []
            for k in range(10):
                shuffle(ranges)
                train_data, train_labels, test_data , test_labels = corpus[ranges[:int(0.8*len(ranges))]]\
                    , labels[ranges[:int(0.8 * len(ranges))]]\
                    ,corpus[ranges[int(0.8*len(ranges)):]],labels[ranges[int(0.8*len(ranges)):]]
                de = DE(Goal="Min", GEN=5)

                v,pareto=de.solve(learners_reg[num], OrderedDict(learners_para_dic[num]), learners_para_bounds[num], learners_para_categories[num],
                         train_data
                         , train_labels, test_data, test_labels)
                l.append(v.fit)
                paras.append(v.ind)
            temp[learners_class[num].__name__] = [l, paras, time.time() - start_time]
            print(l, paras, time.time() - start_time)
    with open('../dump/'+res+'_late.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd ())