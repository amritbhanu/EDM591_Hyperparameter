from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pickle
import numpy as np
import os
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.text as mpl_text

class AnyObject(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline',
                                horizontalalignment=u'left', multialignment=None,
                                fontproperties=None, linespacing=None,
                                rotation_mode=None)
        handlebox.add_artist(patch)
        return patch

ROOT=os.getcwd()
files_reg=['dataset1_math', 'dataset1_portuguese']
files_class=[ 'dataset2', 'dataset3']
metrics=['accuracy','recall','precision','false_alarm']#, 'times']
approaches=['untuned', 'early']
learners_class={'run_dectreeclas':'DT','run_rfclas':'RF'}
learners_reg={'run_dectreereg':'DT','run_rfreg':'RF'}
# preprocess_names={'0mean_':'0meI', '0meancategorize_ii_':'0meD',
#                    'meanmean_':'memeI', 'meanmeancategorize_ii_':'memeD',
#                    'medianmean_':'medmeI', 'medianmeancategorize_ii_':'medmeD'}
preprocess_names={'0mean_':'0meI', '0meancategorize_ii_':'0meD', '0min-max_':'0miI', '0min-maxcategorize_ii_':'0minD',
                   'meanmean_':'memeI', 'meanmeancategorize_ii_':'memeD', 'meanmin-max_':'meminI', 'meanmin-maxcategorize_ii_':'meminD',
                   'medianmean_':'medmeI', 'medianmeancategorize_ii_':'medmeD', 'medianmin-max_':'medminI', 'medianmin-maxcategorize_ii_':'medminD'}

def dump_files(i='',untuned=''):
    file_names=[]
    for _, _, files in os.walk(ROOT + "/../dump/new/"):
        for file in files:
            if file.endswith(untuned+".pickle") and i in file:
                file_names.append(file)
    return file_names

def values_class(file,file_names,untuned):
    dic={}
    for m in metrics:
        dic[m]={}
        for f in file_names:
            with open("../dump/new/"+f, 'rb') as handle:
                a = pickle.load(handle)
            res=f.split(file)[1]
            res=res.split(untuned)[0]
            dic[m][preprocess_names[res]]={}
            for l,values in a.iteritems():
                if l!='run_svmclas' and l!='cols':
                    features=[round(np.median(x),5) for x in zip(*values['features'])]
                    columns=a['cols']
                    temp={x:features[i] for i,x in enumerate (columns)}
                    dic[m][preprocess_names[res]][learners_class[l]]=temp
    return dic

def values_class_tune(file,file_names,untuned):
    dic={}
    dic1 = {}
    for m in metrics:
        dic1[m] = list(filter(lambda x: m in x, file_names))
    for m in metrics:
        dic[m] = {}
        for f in dic1[m]:
            with open("../dump/new/"+f, 'rb') as handle:
                a = pickle.load(handle)
            res=f.split(m+file)[1]
            res=res.split(untuned)[0]
            if preprocess_names[res] not in dic[m]:
                dic[m][preprocess_names[res]]=[]
            for l,values in a.iteritems():
                if l!='run_svmclas' and l!='cols':
                    features=[round(np.median(x),5) for x in zip(*values['features'])]
                    columns=a['cols']
                    temp={x:features[i] for i,x in enumerate (columns)}
                    dic[m][preprocess_names[res]][learners_class[l]]=temp
    return dic

def values_reg(file, file_names, untuned):
        dic = {}
        for m in ['MSE']:
            dic[m] = {}
            for f in file_names:
                with open("../dump/new/" + f, 'rb') as handle:
                    a = pickle.load(handle)
                res = f.split(file)[1]
                res = res.split(untuned)[0]
                dic[m][preprocess_names[res]] = {}
                for l, values in a.iteritems():
                    if l != 'run_svmclas' and l != 'cols' and untuned=='untuned':
                        features = [round(np.median(x),5) for x in zip(*values['features'])]
                        columns = a['cols']
                        temp = {x: features[i] for i, x in enumerate(columns)}

                        dic[m][preprocess_names[res]][learners_class[l]] = temp
                    elif l != 'run_svmreg' and l != 'cols' and untuned=='early':
                        features = [round(np.median(x),5) for x in zip(*values['features'])]
                        columns = a['cols']
                        temp = {x: features[i] for i, x in enumerate(columns)}

                        dic[m][preprocess_names[res]][learners_reg[l]] = temp
        return dic


def draw_class(dic,tuned=''):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 70, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'axes.linewidth':8}
    plt.rcParams.update(paras)

    for i,a in enumerate(dic.keys()):
        for j,b in enumerate(dic[a].keys()):
            for k in ['DT','RF']:
                fig, ax = plt.subplots(6,2,figsize=(100, 80))
                l1 = [(x, y) for x in range(6) for y in range(2)]

                for l,c in zip(l1,list(enumerate(dic[a][b].keys()))):
                    dictionary = OrderedDict(sorted(dic[a][b][c[1]][k].items(), key=lambda x: x[1]))
                    y_pos = range(len(dictionary.keys()[-10:]))
                    ax[l[0], l[1]].barh(y_pos, dictionary.values()[-10:], align='center', color='green')
                    ax[l[0], l[1]].set_yticks(y_pos)
                    ax[l[0], l[1]].set_yticklabels(dictionary.keys()[-10:])
                    ax[l[0], l[1]].invert_yaxis()  # labels read top-to-bottom
                    if l>9:
                        ax[l[0], l[1]].set_xlabel('Feature Importance: '+c[1])
                plt.suptitle('Top 10 Features for '+a+" for "+b+" for "+k)
                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
                plt.savefig("../results/new/features_"+a+"_"+b+"_"+k+tuned+".png",bbox_inches='tight')
                plt.close(fig)

def for_untuned():
    dic={}
    for f in files_class:
        files_names = dump_files(f, 'untuned')
        dic[f]=values_class(f, files_names,'untuned')
    draw_class(dic,'')

    dic = {}
    for f in files_reg:
        files_names=dump_files(f,'untuned')
        dic[f] = values_reg(f, files_names, 'untuned')
    draw_class(dic,'')

def for_tuned():
    dic={}
    for f in files_class:
        files_names = dump_files(f, 'early')
        dic[f]=values_class(f, files_names,'early')
    draw_class(dic,'tuned')

    dic = {}
    for f in files_reg:
        files_names=dump_files(f,'early')
        dic[f] = values_reg(f, files_names, 'early')
    draw_class(dic,'tuned')

if __name__ == '__main__':
    #for_untuned()
    for_tuned()