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
            if file.endswith(untuned+".pickle") and i in file:
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

def values_class_tune(file,file_names,untuned):
    dic={}
    dic1={}
    for m in metrics:
        dic1[m]=list(filter(lambda x: m in x, file_names))
    for m in metrics:
        dic[m]={}
        for f in dic1[m]:
            with open("../dump/new/"+f, 'rb') as handle:
                a = pickle.load(handle)
            res=f.split(m+file)[1]
            res=res.split(untuned)[0]
            if preprocess_names[res] not in dic[m]:
                dic[m][preprocess_names[res]]=[]
            for l,values in a.iteritems():
                if l!='cols':
                    dic[m][preprocess_names[res]].append([learners_class[l]]+values[m])
    return dic

def values_class(file,file_names,untuned):
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

def values_reg(file, file_names, untuned):
        dic = {}
        for m in ['MSE']:
            dic[m] = {}
            for f in file_names:
                with open("../dump/new/" + f, 'rb') as handle:
                    a = pickle.load(handle)
                res = f.split(file)[1]
                res = res.split(untuned)[0]
                dic[m][preprocess_names[res]] = []
                for l, values in a.iteritems():
                    if l != 'cols' and untuned=='untuned':
                        dic[m][preprocess_names[res]].append([learners_class[l]] + values[m])
                    elif l != 'cols' and untuned=='early':
                        dic[m][preprocess_names[res]].append([learners_reg[l]] + values[m])
        return dic

def draw_class(dic,tuned=''):
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
    outer = gridspec.GridSpec(1, len(dic.keys()), wspace=0.15, hspace=0.2)
    for i,a in enumerate(dic.keys()):
        inner = gridspec.GridSpecFromSubplotSpec(len(dic[a].keys()), 1, subplot_spec=outer[i], wspace=0.05, hspace=0.0)
        for j,b in enumerate(dic[a].keys()):
            ax = plt.Subplot(fig, inner[j])
            if j==0:
                ax.set_title(a)
            temp=[item[1:] for sublist in dic[a][b].values() for item in sublist]

            bplot=ax.boxplot(temp,showmeans=False,showfliers=False,medianprops=medianprops,capprops=whiskerprops,
                       flierprops=whiskerprops,boxprops=boxprops,whiskerprops=whiskerprops,
                       positions=[1,2,3, 6,7,8, 11,12,13, 16,17,18, 21,22,23, 26,27,28, 31,32,33, 36,37,38,
                                  41,42,43, 46,47,48, 51,52,53, 56,57,58])
            for patch, color in zip(bplot['boxes'], colors):
                patch.set(color=color)
            ax.set_xticks([2,7,12,17,22,27,32,37,42,47,52,57])
            ax.set_xticklabels(dic[a][b].keys(),rotation=90)
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

    plt.savefig("../results/new/graph"+tuned+".png", bbox_inches='tight')
    plt.close(fig)


def draw_reg(dic,tuned=''):
    font = {'size': 50}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 50, 'legend.fontsize': 50, 'axes.labelsize': 50, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9,color='black')
    colors=['cyan', 'blue', 'green']*12
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')
    #meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick',markersize=20)

    fig = plt.figure(figsize=(50, 40))
    outer = gridspec.GridSpec(1, len(dic.keys()), wspace=0.1, hspace=0.2)
    for i,a in enumerate(dic.keys()):
        inner = gridspec.GridSpecFromSubplotSpec(len(dic[a].keys()), 1, subplot_spec=outer[i], wspace=0.05, hspace=0.0)
        for j,b in enumerate(dic[a].keys()):
            ax = plt.Subplot(fig, inner[j])
            if j==0:
                ax.set_title(a)
            temp=[item[1:] for sublist in dic[a][b].values() for item in sublist]

            bplot=ax.boxplot(temp,showmeans=False,showfliers=False,medianprops=medianprops,capprops=whiskerprops,
                       flierprops=whiskerprops,boxprops=boxprops,whiskerprops=whiskerprops,
                       positions=[1,2,3, 6,7,8, 11,12,13, 16,17,18, 21,22,23, 26,27,28, 31,32,33, 36,37,38,
                                  41,42,43, 46,47,48, 51,52,53, 56,57,58])
            for patch, color in zip(bplot['boxes'], colors):
                patch.set(color=color)
            ax.set_xticks([2,7,12,17,22,27,32,37,42,47,52,57])
            ax.set_xticklabels(dic[a][b].keys(),rotation=90)
            ax.set_ylabel(b,labelpad=30)
            #ax.set_ylim([0,1])
            if j!=0:
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
               loc='upper center', bbox_to_anchor=(-0.1, 1.10),
               fancybox=True, shadow=True, ncol=3)
    # plt.figtext(0.40, 0.9, 'DT', color=colors[0],size='large')
    # plt.figtext(0.50, 0.9, 'RF', color=colors[1],size='large')
    # plt.figtext(0.60, 0.9, 'SVM', color=colors[2],size='large')

    plt.savefig("../results/new/graph_reg"+tuned+".png", bbox_inches='tight')
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

def draw_class_comparison(untuned,tuned):
        font = {'size': 50}
        plt.rc('font', **font)
        paras = {'lines.linewidth': 50, 'legend.fontsize': 50, 'axes.labelsize': 50, 'legend.frameon': True,
                  'axes.linewidth': 8}
        plt.rcParams.update(paras)

        boxprops = dict(linewidth=9, color='black')
        colors = ['cyan', 'cyan','blue','blue', 'green','green']
        whiskerprops = dict(linewidth=5)
        medianprops = dict(linewidth=8, color='firebrick')

        for a in untuned:
            for b in preprocess_names.values():
                fig, axes = plt.subplots(2, 2, figsize=(50, 40))
                l1 = [(x, y) for x in range(2) for y in range(2)]
                for k, c in zip(l1, metrics):
                    temp=untuned[a][c][b]
                    temp1=tuned[a][c][b]
                    data=[]
                    for i,j in enumerate(temp):
                        data.append(temp[i][1:])
                        data.append(temp1[i][1:])

                    bplot = axes[k[0],k[1]].boxplot(data, showmeans=False, showfliers=False, medianprops=medianprops,
                                   capprops=whiskerprops,
                                   flierprops=whiskerprops, boxprops=boxprops, whiskerprops=whiskerprops,
                                   positions=[1, 2, 5,6, 9,10])
                    for patch, color in zip(bplot['boxes'], colors):
                        patch.set(color=color)
                    axes[k[0], k[1]].set_xticks([1,2,5,6,9,10])
                    axes[k[0], k[1]].set_xticklabels(['untuned','tuned']*3, rotation=90)
                    axes[k[0], k[1]].set_ylabel(c, labelpad=30)
                    axes[k[0], k[1]].set_ylim([0,1])
                    if k[0] != 1:
                        plt.setp(axes[k[0], k[1]].get_xticklabels(), visible=False)
                obj_0 = AnyObject("DT", colors[0])
                obj_1 = AnyObject("RF", colors[2])
                obj_2 = AnyObject("SVM", colors[4])
                plt.suptitle(a+" - "+b)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.legend([obj_0, obj_1, obj_2], ['Decision Tree', 'Random Forest', 'Support Vector Machine'],
                           handler_map={obj_0: AnyObjectHandler(), obj_1: AnyObjectHandler(), obj_2: AnyObjectHandler()},
                           loc='upper center', bbox_to_anchor=(-0.1, 2.20),
                           fancybox=True, shadow=True, ncol=3)
                plt.savefig("../results/new/performance" + a + b+".png", bbox_inches='tight')
                plt.close(fig)


def draw_reg_comparison(untuned, tuned):
    font = {'size': 50}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 50, 'legend.fontsize': 50, 'axes.labelsize': 50, 'legend.frameon': True,
             'axes.linewidth': 8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9, color='black')
    colors = ['cyan', 'cyan', 'blue', 'blue', 'green', 'green']
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')

    for a in untuned:
        for b in preprocess_names.values():
            fig, axes = plt.subplots(1, 1, figsize=(50, 40))
            l1 = [(0, 0)]
            for k, c in zip(l1, ['MSE']):
                temp = untuned[a][c][b]
                temp1 = tuned[a][c][b]
                data = []
                for i, j in enumerate(temp):
                    data.append(temp[i][1:])
                    data.append(temp1[i][1:])

                bplot = axes.boxplot(data, showmeans=False, showfliers=False, medianprops=medianprops,
                                                 capprops=whiskerprops,
                                                 flierprops=whiskerprops, boxprops=boxprops, whiskerprops=whiskerprops,
                                                 positions=[1, 2, 5, 6, 9, 10])
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set(color=color)
                axes.set_xticks([1, 2, 5, 6, 9, 10])
                axes.set_xticklabels(['untuned', 'tuned'] * 3, rotation=90)
                axes.set_ylabel(c, labelpad=30)

            obj_0 = AnyObject("DT", colors[0])
            obj_1 = AnyObject("RF", colors[2])
            obj_2 = AnyObject("SVM", colors[4])
            plt.suptitle(a + " - " + b)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.legend([obj_0, obj_1, obj_2], ['Decision Tree', 'Random Forest', 'Support Vector Machine'],
                       handler_map={obj_0: AnyObjectHandler(), obj_1: AnyObjectHandler(), obj_2: AnyObjectHandler()},
                       loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=True, shadow=True, ncol=3)
            plt.savefig("../results/new/performance" + a + b + ".png", bbox_inches='tight')
            plt.close(fig)

def for_untuned():
    dic1={}
    for f in files_class:
        files_names = dump_files(f, 'untuned')
        dic1[f]=values_class(f, files_names,'untuned')
    #draw_class(dic1,'')

    dic = {}
    for f in files_reg:
        files_names=dump_files(f,'untuned')
        dic[f] = values_reg(f, files_names, 'untuned')
    #draw_reg(dic,'')
    return dic1,dic

def for_tuned():
    dic1={}
    for f in files_class:
        files_names = dump_files(f, 'early')
        dic1[f]=values_class_tune(f, files_names,'early')
    #draw_class(dic1,'tuned')

    dic = {}
    for f in files_reg:
        files_names=dump_files(f,'early')
        dic[f] = values_reg(f, files_names, 'early')
    #draw_reg(dic,'tuned')
    return dic1,dic

if __name__ == '__main__':
    clas_un,reg_un=for_untuned()
    clas_tu,reg_tu =for_tuned()

    draw_class_comparison(clas_un,clas_tu)
    draw_reg_comparison(reg_un,reg_tu)