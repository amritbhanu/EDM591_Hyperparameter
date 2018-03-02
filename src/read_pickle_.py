from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pickle
import numpy as np
import os
import csv
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
learners_class={'run_dectreeclas':'DT','run_rfclas':'RF','run_svmclas':'SVM'}
learners_reg={'run_dectreereg':'DT','run_rfreg':'RF','run_svmreg':'SVM'}
preprocess_names={'0mean_':'0meI', '0meancategorize_ii_':'0meD', '0min-max_':'0miI', '0min-maxcategorize_ii_':'0minD',
                   'meanmean_':'memeI', 'meanmeancategorize_ii_':'memeD', 'meanmin-max_':'meminI', 'meanmin-maxcategorize_ii_':'meminD',
                   'medianmean_':'medmeI', 'medianmeancategorize_ii_':'medmeD', 'medianmin-max_':'medminI', 'medianmin-maxcategorize_ii_':'medminD'}

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
            for f in file_names:
                with open("../dump/new/"+f, 'rb') as handle:
                    a = pickle.load(handle)
                temp=[]
                res=f.split(file)[1]
                res=res.split(untuned)[0]
                temp.append(preprocess_names[res])
                co = []
                for l,values in a.iteritems():
                    if l!='cols':
                        co.append(learners_class[l])
                        temp.append(round(np.median(values[m]),2))
                l1.append(temp)
            cols = [file] + co
            writer.writerow(cols)
            writer.writerows(l1)

def values(file,file_names,untuned):
    dic={}
    for m in metrics:
        dic[m]={}
        for f in file_names:
            with open("../dump/new/"+f, 'rb') as handle:
                a = pickle.load(handle)
            res=f.split(file)[1]
            res=res.split(untuned)[0]
            dic[m][preprocess_names[res]]=[]
            for l,values in a.iteritems():
                if l!='cols':
                    dic[m][preprocess_names[res]].append([learners_class[l]]+values[m])
    return dic
    ## file contains the main dateset name

def draw(dic):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 70, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9,color='black')
    colors=['cyan', 'blue', 'green']*12
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')
    #meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick',markersize=20)

    fig = plt.figure(figsize=(100, 80))
    outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.2)
    for i,a in enumerate(dic.keys()):
        inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[i], wspace=0.05, hspace=0.0)
        for j,b in enumerate(dic[a].keys()):
            ax = plt.Subplot(fig, inner[j])
            if j==0:
                ax.set_title(a)
            temp=[item[1:] for sublist in dic[a][b].values() for item in sublist]

            bplot=ax.boxplot(temp,showmeans=False,showfliers=False,medianprops=medianprops,capprops=whiskerprops,
                       flierprops=whiskerprops,boxprops=boxprops,whiskerprops=whiskerprops,
                       positions=[1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19, 21,22,23, 25,26,27, 29,30,31, 33,34,35, 37,38,39,
                                  41,42,43, 45,46,47])
            for patch, color in zip(bplot['boxes'], colors):
                patch.set(color=color)
            ax.set_xticks([2,6,10,14,18,22,26,30,34,38,42,46])
            ax.set_xticklabels(dic[a][b].keys(),rotation=45)
            ax.set_ylabel(b,labelpad=30)
            #ax.set_ylim([0,1])
            if j!=3:
                plt.setp(ax.get_xticklabels(), visible=False)
            fig.add_subplot(ax)

    # box1 = TextArea("DT", textprops=dict(color=colors[0],size='large'))
    # box2 = TextArea("RF", textprops=dict(color=colors[1],size='large'))
    # box3 = TextArea("SVM", textprops=dict(color=colors[2],size='large'))
    # box = HPacker(children=[box1, box2, box3],
    #               align="center",
    #               pad=0, sep=5)
    #
    # anchored_box = AnchoredOffsetbox(loc=3,child=box, pad=0.,frameon=True,
    #                                  bbox_to_anchor=(0., 1.02),borderpad=0.)
    #
    # plt.artist(anchored_box)
    obj_0 = AnyObject("DT", colors[0])
    obj_1 = AnyObject("RF", colors[1])
    obj_2 = AnyObject("SVM", colors[2])

    plt.legend([obj_0, obj_1,obj_2], ['Decision Tree', 'Random Forest', 'Support Vector Machine'],
               handler_map={obj_0: AnyObjectHandler(), obj_1: AnyObjectHandler(),obj_2: AnyObjectHandler()},
               loc='upper center', bbox_to_anchor=(-0.1, 4.25),
               fancybox=True, shadow=True, ncol=3)
    # plt.figtext(0.40, 0.9, 'DT', color=colors[0],size='large')
    # plt.figtext(0.50, 0.9, 'RF', color=colors[1],size='large')
    # plt.figtext(0.60, 0.9, 'SVM', color=colors[2],size='large')

    plt.savefig("../results/new/graph.png", bbox_inches='tight')
    plt.close(fig)

def writecsvs_reg(file,file_names,untuned):
    with open("../csv/"+file+untuned+".csv", "wb") as f:
        writer = csv.writer(f,delimiter=',')
        for m in ['MSE','times']:
            writer.writerow([m] * 4)
            l1=[]
            for f in file_names:
                with open("../dump/new/"+f, 'rb') as handle:
                    a = pickle.load(handle)
                temp=[]
                res=f.split(file)[1]
                res=res.split(untuned)[0]
                temp.append(preprocess_names[res])
                co=[]
                for l,values in a.iteritems():
                    if l!='cols':
                        co.append(learners_class[l])
                        temp.append(round(np.median(values[m]),2))

                l1.append(temp)
            cols=[file]+co
            writer.writerow(cols)
            writer.writerows(l1)

def for_untuned():
    dic={}
    for f in files_class:
        files_names = dump_files(f, 'untuned')
        dic[f]=values(f, files_names,'untuned')
    draw(dic)
    for f in files_reg:
        files_names=dump_files(f,'untuned')
        writecsvs_reg(f, files_names,'untuned')

if __name__ == '__main__':
    for_untuned()

