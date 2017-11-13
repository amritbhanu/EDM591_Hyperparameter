from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True

import pickle
import numpy as np
import pandas as pd

with open("../dump/dataset1_untuned.pickle", 'rb') as handle:
    l = pickle.load(handle)

for i in l.keys():
    print(i)
    print(np.median(l[i][0]),np.percentile(l[i][0],75)-np.percentile(l[i][0],25))
    print(l[i][1])

# data=np.load("../data/dataset2/mergedata.npy")
# label=np.load("../data/dataset2/label.npy")
# df=pd.DataFrame(data)
# df['labels']=pd.DataFrame(label)
# df.to_csv("../data/dataset2.csv",index=False)
