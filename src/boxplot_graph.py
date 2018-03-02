# from __future__ import print_function, division
#
# __author__ = 'amrit'
#
# import sys
#
# sys.dont_write_bytecode = True
# import matplotlib.pyplot as plt
# from collections import OrderedDict
#
# import matplotlib.gridspec as gridspec
#
# font = {
#         'size': 50}
#     plt.rc('font', **font)
#     paras = {'lines.linewidth': 50, 'legend.fontsize': 50, 'axes.labelsize': 70, 'legend.frameon': False,
#              'figure.autolayout': True,'axes.linewidth':5}
#     plt.rcParams.update(paras)
#
#
#
# fig = plt.figure(figsize=(10, 8))
# outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
#
# for i in range(4):
#     inner = gridspec.GridSpecFromSubplotSpec(4, 1,
#                     subplot_spec=outer[i], wspace=0.1, hspace=0.1)
#
#     for j in range(4):
#         ax = plt.Subplot(fig, inner[j])
#         t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i,j))
#         t.set_ha('center')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         fig.add_subplot(ax)
#
# fig.show()
#     f, axarr = plt.subplots(nrows=4,ncols=1, sharex=True, sharey=True)
#     f.suptitle('Dataset1')
#     axarr[0,0].boxplot(x, y)
#     axarr[1,0].boxplot(x, y)
#     axarr[2,0].boxplot(x, 2 * y ** 2 - 1)
#     # Bring subplots close to each other.
#     f.subplots_adjust(hspace=0)
#     # Hide x labels and tick labels for all but bottom plot.
#     for ax in axarr:
#         ax.label_outer()
#
#     temp = OrderedDict()
#     medianprops = dict(linewidth=5, color='firebrick')
#
#     boxColors = ['darkkhaki', 'royalblue']
#     boxprops_svm = dict(linewidth=8,color='black')
#     boxprops_rt = dict(linewidth=8,color='red')
#     boxprops_dt = dict(linewidth=8,color='blue')
#     whiskerprops = dict(linewidth=8)
#     meanpointprops = dict(marker='D', markeredgecolor='black',
#                           markerfacecolor='firebrick',markersize=20)
#
#     plt.boxplot(data,showmeans=False,showfliers=False,medianprops=medianprops,capprops=whiskerprops,flierprops=whiskerprops,boxprops=boxprops,whiskerprops=whiskerprops) #meanprops=meanpointprops
#       positions=[1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19, 21,22,23, 25,26,27, 29,30,31, 33,34,35, 37,38,39,
#       41,42,43, 45,46,47]
