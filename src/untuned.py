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
import os

learners_class=[run_dectreeclas,run_rfclas,run_svmclas]
learners_reg=[run_dectreereg, run_rfreg,run_svmreg]
learners_para_dic=[OrderedDict(), OrderedDict(), OrderedDict()]
metrics=['accuracy','recall','precision','false_alarm','times','features']

ROOT=os.getcwd()
metric='recall'

def _test(res=''):
    seed(1)
    np.random.seed(1)

    file_names = []
    for _, _, files in os.walk(ROOT + "/../data/preprocessed_data/"):
        for file in files:
            if file.startswith(res):
                file_names.append(file.split(".csv")[0])

    if res == 'dataset1_math' or res == 'dataset1_portuguese':
        class_flag = False
    else:
        class_flag = True


    for f in file_names:
        df = pd.read_csv("../data/preprocessed_data/" + f + ".csv")
        corpus, labels = np.array(df[df.columns[:-1]].values.tolist()), np.array(df[df.columns[-1]].values.tolist())
        columns=df.columns[:-1].values.tolist()
        ranges = range(0, len(labels))
        temp = {}
        for num,i in enumerate(learners_class):

            temp[i.__name__] = {}
            temp['cols'] = columns

            if class_flag:
                for m in metrics:
                    temp[i.__name__][m] = []
                _, whole = learners_class[num](OrderedDict(learners_para_dic[num]), corpus
                                            , labels, corpus, labels, metric)
                temp[i.__name__]['whole']=whole[1]

                for _ in range(5):
                    shuffle(ranges)
                    corpus,labels=corpus[ranges],labels[ranges]
                    skf = StratifiedKFold(n_splits=5)
                    for train_index, test_index in skf.split(corpus, labels):
                        train_data, train_labels = corpus[train_index], labels[train_index]
                        test_data, test_labels = corpus[test_index], labels[test_index]

                        start_time=time.time()
                        _,val=learners_class[num](OrderedDict(learners_para_dic[num]),train_data
                                 , train_labels,test_data,test_labels,metric)
                        temp[i.__name__]['times'].append(time.time() - start_time)
                        for x in val[0]:
                            temp[i.__name__][x].append(val[0][x])
                        temp[i.__name__]['features'].append(val[1])

            else:
                ## Whole Model being checked
                _ , whole = learners_reg[num](OrderedDict(learners_para_dic[num]), corpus
                                          ,labels, corpus, labels)
                temp[i.__name__]['times'] = []
                temp[i.__name__]['MSE'] = []
                temp[i.__name__]['features'] = []
                temp[i.__name__]['whole'] = whole[1]

                for k in range(25):
                    shuffle(ranges)
                    train_data, train_labels, test_data , test_labels = corpus[ranges[:int(0.8*len(ranges))]]\
                        , labels[ranges[:int(0.8 * len(ranges))]]\
                        ,corpus[ranges[int(0.8*len(ranges)):]],labels[ranges[int(0.8*len(ranges)):]]
                    start_time=time.time()
                    t, val = learners_reg[num](OrderedDict(learners_para_dic[num]), train_data
                                            , train_labels, test_data, test_labels)
                    temp[i.__name__]['times'].append(time.time() - start_time)
                    temp[i.__name__]['MSE'].append(t)
                    temp[i.__name__]['features'].append(val[1])

        with open('../dump/'+f+'_untuned.pickle', 'wb') as handle:
            pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd())