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
metrics=['accuracy','recall','precision','false_alarm','f1','auc']

def _test(res='',metric=''):
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
            l = []
            importance=[]
            _, whole = learners_class[num](OrderedDict(learners_para_dic[num]), corpus
                                        , labels, corpus, labels, metric)
            for _ in range(10):
                shuffle(ranges)
                corpus,labels=corpus[ranges],labels[ranges]
                skf = StratifiedKFold(n_splits=10)
                for train_index, test_index in skf.split(corpus, labels):
                    train_data, train_labels = corpus[train_index], labels[train_index]
                    test_data, test_labels = corpus[test_index], labels[test_index]
                    v,v1=learners_class[num](OrderedDict(learners_para_dic[num]),train_data
                             , train_labels,test_data,test_labels,metric)
                    l.append(v)
                    importance.append(v1)
            temp[learners_class[num].__name__] = [l,importance, time.time() - start_time,df.columns[:-1].values.tolist(),
                                                  whole]
        else:
            l=[]
            importance = []
            ## Whole Model being checked
            _ , whole = learners_reg[num](OrderedDict(learners_para_dic[num]), corpus
                                      ,labels, corpus, labels)

            for k in range(100):
                shuffle(ranges)
                train_data, train_labels, test_data , test_labels = corpus[ranges[:int(0.8*len(ranges))]]\
                    , labels[ranges[:int(0.8 * len(ranges))]]\
                    ,corpus[ranges[int(0.8*len(ranges)):]],labels[ranges[int(0.8*len(ranges)):]]
                v, v1 = learners_reg[num](OrderedDict(learners_para_dic[num]), train_data
                                        , train_labels, test_data, test_labels)
                l.append(v)
                importance.append(v1)
            temp[learners_reg[num].__name__]=[l,importance,time.time()-start_time,df.columns[:-1].values.tolist(),whole]
    with open('../dump/'+res+"_"+metric+'_untuned.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd())