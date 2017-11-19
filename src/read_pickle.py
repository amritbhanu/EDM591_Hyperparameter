from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True

import pickle
import numpy as np
import pandas as pd

data = ['dataset1', 'dataset2', 'dataset3']
learners = ['DT', 'RF', 'SVM']

def runtimes():
    l=[]
    l_early=[]
    l_late=[]
    for i in untuned.keys():
        l.append(untuned[i][1])
        l_early.append(early[i][2])
        l_late.append(late[i][2])

if __name__ == '__main__':
    for i in data:
        with open("../dump/"+i+".pickle", 'rb') as handle:
            early = pickle.load(handle)
        with open("../dump/"+i+"_late.pickle", 'rb') as handle:
            late = pickle.load(handle)
        with open("../dump/"+i+"_untuned.pickle", 'rb') as handle:
            untuned = pickle.load(handle)
        runtimes()

