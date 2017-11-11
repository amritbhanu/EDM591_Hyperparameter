from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd
from random import seed, shuffle
import numpy as np
from ML import *
import time
from demos import cmd
from sklearn.model_selection import StratifiedKFold
import pickle
from collections import OrderedDict

learners_class=[run_dectreeclas,run_rfclas,run_svmclas]
learners_reg=[run_dectreereg, run_rfreg,run_svmreg]
learners_para_dic=[OrderedDict(), OrderedDict(), OrderedDict()]

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
            for train_index, test_index in skf.split(corpus, labels):
                train_data, train_labels = corpus[train_index], labels[train_index]
                test_data, test_labels = corpus[test_index], labels[test_index]
                v=learners_class[num](OrderedDict(learners_para_dic[num]),train_data
                         , train_labels,test_data,test_labels)
                l.append(v)
            temp[learners_class[num].__name__] = [l, time.time() - start_time]
            print(l, time.time() - start_time)
        else:
            l=[]
            for k in range(10):
                shuffle(ranges)
                train_data, train_labels, test_data , test_labels = corpus[ranges[:int(0.8*len(ranges))]]\
                    , labels[ranges[:int(0.8 * len(ranges))]]\
                    ,corpus[ranges[int(0.8*len(ranges)):]],labels[ranges[int(0.8*len(ranges)):]]
                v = learners_reg[num](OrderedDict(learners_para_dic[num]), train_data
                                        , train_labels, test_data, test_labels)
                l.append(v)
            temp[learners_reg[num].__name__]=[l,time.time()-start_time]
            print(l,time.time()-start_time)
    with open('../dump/'+res+'_untuned.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd())