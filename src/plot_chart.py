import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data1_m = pd.read_csv('../csv/dataset1_mathuntuned.csv')
data1_p = pd.read_csv('../csv/dataset1_portugueseuntuned.csv')
data2 = pd.read_csv('../csv/dataset2untuned.csv')
data3 = pd.read_csv('../csv/dataset3untuned.csv')

def plot_dataset(data):
	plt.figure(1)

	# 0 min_max
	# plt.subplot(3,4,1)
	x = [1,2,3,4,5,6,7,8,9,10,11,12]
	y1 = list(data[1:13]['MSE.1'])
	y2 = list(data[1:13]['MSE.2'])
	y3 = list(data[1:13]['MSE.3'])
	my_ticks = data.iloc[1:13,:1].values.tolist()
	print(my_ticks)
	plt.xticks(x,my_ticks)
	plt.plot(x,y1)
	plt.plot(x,y2)
	plt.plot(x,y3)
	plt.grid()
	plt.legend(['decision tree','random forest','svm'],loc='upper left')
	plt.title('data1_math')

	plt.savefig('../csv/data1_p')



if __name__ == '__main__':
	'''
	dataset = ['data1_m','data1_p','data2','data3']
	for df in dataset:
		plot_dataset(df)
	'''
	plot_dataset(data1_p)