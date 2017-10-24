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

learners_class=[run_dectreeclas,run_rfclas]
learners_reg=[run_dectreereg,run_rfreg]
learners_para_dic=[OrderedDict([("max_features",None), ("min_samples_split",2),("min_impurity_split",0.0), ("max_depth",None),
                               ("min_samples_leaf", 1)]), OrderedDict([("max_features","auto"), ("min_samples_split",2),
                                ("max_leaf_nodes",None), ("min_samples_leaf",1), ("min_impurity_split",None),("n_estimators",10)])]
learners_para_bounds=[[(0.01,1), (2,20), (0,1), (1,20), (1,50)],[(0.01,1), (2,20), (2,50), (1,20), (0,1), (50,100)]]
learners_para_categories=[["continuous", "integer", "continuous", "integer", "integer"],["continuous", "integer", "integer", "integer","continuous", "integer"]]


def _test(res=''):
    seed(1)
    np.random.seed(1)
    df=pd.read_csv("../data/"+res+".csv")
    corpus,labels=np.array(df[df.columns[:-1]].values.tolist()),np.array(df[df.columns[-1]].values.tolist())
    ranges = range(0, len(labels))
    class_flag=False
    temp={}

    for num,i in enumerate(learners_class):
        start_time = time.time()
        if class_flag:
            skf = StratifiedKFold(n_splits=10)
            l=[]
            for train_index, test_index in skf.split(corpus, labels):
                train_data, train_labels = corpus[train_index], labels[train_index]
                test_data, test_labels = corpus[test_index], labels[test_index]
                de = DE()
                v,pareto=de.solve(learners_class[num],OrderedDict(learners_para_dic[num]),learners_para_bounds[num],learners_para_categories[num],train_data
                         , train_labels,test_data,test_labels)
                l.append(v.fit)
            temp[learners_class[num].__name__] = [l, time.time() - start_time]
            print(l, time.time() - start_time)
        else:
            l=[]
            for k in range(10):
                shuffle(ranges)
                train_data, train_labels, test_data , test_labels = corpus[ranges[:int(0.8*len(ranges))]]\
                    , labels[ranges[:int(0.8 * len(ranges))]]\
                    ,corpus[ranges[int(0.8*len(ranges)):]],labels[ranges[int(0.8*len(ranges)):]]
                de = DE(Goal="Min")

                v,pareto=de.solve(learners_reg[num], OrderedDict(learners_para_dic[num]), learners_para_bounds[num], learners_para_categories[num],
                         train_data
                         , train_labels, test_data, test_labels)
                l.append(v.fit)
            temp[learners_reg[num].__name__]=[l,time.time()-start_time]
            print(l,time.time()-start_time)
    with open('../dump/'+res+'_untuned.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd())