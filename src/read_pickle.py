from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True

import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sk import rdivDemo
data = ['dataset1', 'dataset2', 'dataset3']
learners = ['DT', 'RF', 'SVM']

def runtimes(x):
    l={}
    l_early={}
    l_late={}
    for i in untuned.keys():
        if "d" in i and "t" in i:
            l["DT"]=round(untuned[i][1]/100,2)
            l_early["DT"] = round(early[i][2]/100,2)
            l_late["DT"] = round(late[i][2]/100,2)
        elif "svm" in i:
            l["SVM"]=round(untuned[i][1]/100,2)
            l_early["SVM"] = round(early[i][2]/100,2)
            l_late["SVM"] = round(late[i][2]/100,2)
        elif "rf" in i:
            l["RF"]=round(untuned[i][1]/100,2)
            l_early["RF"] = round(early[i][2]/100,2)
            l_late["RF"] = round(late[i][2]/100,2)
    l=OrderedDict(sorted(l.items(), key=lambda t: t[0]))
    l_early = OrderedDict(sorted(l_early.items(), key=lambda t: t[0]))
    l_late = OrderedDict(sorted(l_late.items(), key=lambda t: t[0]))
    font = {
        'size': 80}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 8, 'legend.fontsize': 50, 'axes.labelsize': 100, 'legend.frameon': False,
             'figure.autolayout': True, 'axes.linewidth': 10}
    plt.rcParams.update(paras)
    fig = plt.figure(figsize=(30, 20))
    plt.plot(l.values(),color="green", label='Untuned')
    plt.plot(l_early.values(), color="red", label='Early Tuned')
    plt.plot(l_late.values(), color="blue", label='Late Tuned')
    for i, j in enumerate(l.keys()):
        if i==0:
            plt.annotate(str(l[j]), xy=(i, l[j]+5),size=50)
            plt.annotate(str(l_early[j]), xy=(i, l_early[j]+15),size=50)
            plt.annotate(str(l_late[j]), xy=(i, l_late[j]+25),size=50)
        elif i==1:
            plt.annotate(str(l[j]), xy=(i, l[j]+3),size=50)
            plt.annotate(str(l_early[j]), xy=(i, l_early[j]+5),size=50)
            plt.annotate(str(l_late[j]), xy=(i, l_late[j]+5),size=50)
        elif i==2:
            plt.annotate(str(l[j]), xy=(i-0.2, l[j]+1),size=50)
            plt.annotate(str(l_early[j]), xy=(i-0.2, l_early[j]+5),size=50)
            plt.annotate(str(l_late[j]), xy=(i-0.2, l_late[j]+12),size=50)
    plt.ylim(-5,170)
    plt.xticks(xrange(len(l.keys())), l.keys())
    plt.ylabel("Runtimes (in Secs)", labelpad=30)
    plt.xlabel("Learners", labelpad=30)
    plt.legend(bbox_to_anchor=(0.3, 0.8), loc=1, ncol=1, borderaxespad=0.1)
    plt.savefig("../results/runtimes_"+x+".png")
    plt.close(fig)

def para_change(x):
    l_early={}
    l_late={}
    for i in untuned.keys():
        if "d" in i and "t" in i:
            l_early["DT"] = pd.DataFrame(early[i][1])
            l_late["DT"] = pd.DataFrame(late[i][1])
        elif "svm" in i:
            temp=pd.DataFrame(early[i][1])
            temp.drop(["kernel"],axis=1,inplace=True)
            l_early["SVM"] = temp
            temp1 = pd.DataFrame(late[i][1])
            temp1.drop(["kernel"], axis=1,inplace=True)
            l_late["SVM"] = temp1
        elif "rf" in i:
            l_early["RF"] = pd.DataFrame(early[i][1])
            l_late["RF"] = pd.DataFrame(late[i][1])
    l_early = OrderedDict(sorted(l_early.items(), key=lambda t: t[0]))
    l_late = OrderedDict(sorted(l_late.items(), key=lambda t: t[0]))
    font = {
        'size': 50}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 50, 'legend.fontsize': 50, 'axes.labelsize': 70, 'legend.frameon': False,
             'figure.autolayout': True, 'axes.linewidth': 5}
    plt.rcParams.update(paras)

    medianprops = dict(linewidth=5, color='firebrick')

    boxprops = dict(linewidth=8)
    whiskerprops = dict(linewidth=8)
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick', markersize=20)
    for i in l_early.keys():
        fig = plt.figure(figsize=(30, 20))
        tempo=[]
        for k in l_early[i].columns:
            tempo.append(l_early[i][k].tolist())
        plt.boxplot(tempo, showmeans=False, showfliers=False, medianprops=medianprops, capprops=whiskerprops,
                    flierprops=whiskerprops, boxprops=boxprops, whiskerprops=whiskerprops,meanprops=meanpointprops)

        plt.ylabel("Boxplot", labelpad=30)
        plt.xlabel(x+' '+ i, labelpad=30)
        plt.xticks(range(1,len(l_early[i].columns)+1), l_early[i].columns,rotation=30)
        plt.legend(bbox_to_anchor=(0.3, 0.8), loc=1, ncol=1, borderaxespad=0.1)
        plt.savefig("../results/para_"+x+"_"+i+"_early.png")
        plt.close(fig)

    for i in l_late.keys():
        fig = plt.figure(figsize=(30, 20))
        tempo = []
        for k in l_late[i].columns:
            tempo.append(l_late[i][k].tolist())
        plt.boxplot(tempo, showmeans=False, showfliers=False, medianprops=medianprops, capprops=whiskerprops,
                    flierprops=whiskerprops, boxprops=boxprops, whiskerprops=whiskerprops,meanprops=meanpointprops)

        plt.ylabel("Boxplot", labelpad=30)
        plt.xlabel(x+' ' + i, labelpad=30)
        plt.xticks(range(1,len(l_late[i].columns)+1), l_late[i].columns,rotation=30)
        plt.savefig("../results/para_"+x+"_"+i+"_late.png")
        plt.close(fig)

def performance(f):
    l={}
    l_early={}
    l_late={}
    for i in untuned.keys():
        if "d" in i and "t" in i:
            l["DT"] = untuned[i][0]
            l_early["DT"] = early[i][0]
            l_late["DT"] = late[i][0]
        elif "svm" in i:
            l["SVM"] = untuned[i][0]
            l_early["SVM"] = early[i][0]
            l_late["SVM"] = late[i][0]
        elif "rf" in i:
            l["RF"] = untuned[i][0]
            l_early["RF"] = early[i][0]
            l_late["RF"] = late[i][0]
    l = OrderedDict(sorted(l.items(), key=lambda t: t[0]))
    l_early = OrderedDict(sorted(l_early.items(), key=lambda t: t[0]))
    l_late = OrderedDict(sorted(l_late.items(), key=lambda t: t[0]))

    temp=[]
    temp1=[]
    temp2=[]
    temp3 = OrderedDict()
    for i in l.keys():
        temp.append(np.median(l[i]))
        temp1.append(np.median(l_early[i]))
        temp2.append(np.median(l_late[i]))

        temp3[i] = []
        temp3[i].append(["untuned"] + l[i])
        temp3[i].append(["early"] + l_early[i])
        temp3[i].append(["late"] + l_late[i])

    font = {'size': 80}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 8, 'legend.fontsize': 50, 'axes.labelsize': 100, 'legend.frameon': False,
             'figure.autolayout': True, 'axes.linewidth': 10}
    plt.rcParams.update(paras)
    fig = plt.figure(figsize=(30, 20))

    plt.plot(temp, color="green", label='Untuned')
    plt.plot(temp1, color="red", label='Early Tuned')
    plt.plot(temp2, color="blue", label='Late Tuned')

    for i, j in enumerate(l.keys()):

        temp_dic=OrderedDict()
        ranks = rdivDemo(temp3[j])
        for _, _, z in ranks:
            temp_dic[z.name] = z.rank + 1

        plt.annotate("rk: "+str(temp_dic["untuned"]), xy=(i, temp[i]),size=50)
        plt.annotate("rk: "+str(temp_dic["early"]), xy=(i, temp1[i]),size=50)
        plt.annotate("rk: "+str(temp_dic["late"]), xy=(i, temp2[i]),size=50)

    plt.xticks(xrange(len(l.keys())), l.keys())
    plt.ylabel("Performance", labelpad=30)
    plt.xlabel("Learners", labelpad=30)
    plt.legend(bbox_to_anchor=(0.9, 1.1), loc=1, ncol=3, borderaxespad=0.1)
    plt.savefig("../results/performance_"+f+".png")
    plt.close(fig)

if __name__ == '__main__':
    for i in data:
        with open("../dump/"+i+".pickle", 'rb') as handle:
            early = pickle.load(handle)
        with open("../dump/"+i+"_late.pickle", 'rb') as handle:
            late = pickle.load(handle)
        with open("../dump/"+i+"_untuned.pickle", 'rb') as handle:
            untuned = pickle.load(handle)
        runtimes(i)

        para_change(i)

        performance(i)

