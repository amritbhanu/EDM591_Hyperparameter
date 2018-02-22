from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pickle
import numpy as np
import os
import csv

ROOT=os.getcwd()
files_reg=['dataset1_math', 'dataset1_portuguese']
files_class=[ 'dataset2', 'dataset3']
metrics=['accuracy','recall','precision','false_alarm', 'times']
approaches=['untuned', 'early']
learners_class=['run_dectreeclas','run_rfclas','run_svmclas']

def dump_files(i='',untuned=''):
    file_names=[]
    for _, _, files in os.walk(ROOT + "/../dump/new/"):
        for file in files:
            if file.endswith(untuned+".pickle") and file.startswith(i):
                file_names.append(file)
    return file_names


def writecsvs_class(file,file_names,untuned):
    with open("../csv/"+file+untuned+".csv", "wb") as f:
        writer = csv.writer(f,delimiter=',')
        for m in metrics:
            writer.writerow([m] * 4)
            l1=[]
            cols=[]
            for f in file_names:
                with open("../dump/new/"+f, 'rb') as handle:
                    a = pickle.load(handle)
                temp=[]
                res=f.split(file)[1]
                res=res.split(untuned)[0]
                temp.append(res)
                for l,values in a.iteritems():
                    if l!='cols':
                        cols=[file]+a.keys()
                        cols.remove("cols")
                        temp.append(round(np.median(values[m]),2))

                l1.append(temp)
            writer.writerow(cols)
            writer.writerows(l1)

def writecsvs_reg(file,file_names,untuned):
    with open("../csv/"+file+untuned+".csv", "wb") as f:
        writer = csv.writer(f,delimiter=',')
        for m in ['MSE','times']:
            writer.writerow([m] * 4)
            l1=[]
            cols=[]
            for f in file_names:
                with open("../dump/new/"+f, 'rb') as handle:
                    a = pickle.load(handle)
                temp=[]
                res=f.split(file)[1]
                res=res.split(untuned)[0]
                temp.append(res)
                for l,values in a.iteritems():
                    if l!='cols':
                        cols=[file]+a.keys()
                        cols.remove("cols")
                        temp.append(round(np.median(values[m]),2))

                l1.append(temp)
            writer.writerow(cols)
            writer.writerows(l1)

def for_untuned():
    for f in files_class:
        files_names=dump_files(f,'untuned')
        writecsvs_class(f, files_names,'untuned')
    for f in files_reg:
        files_names=dump_files(f,'untuned')
        writecsvs_reg(f, files_names,'untuned')

if __name__ == '__main__':
    for_untuned()

