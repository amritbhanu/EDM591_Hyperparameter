'''
General Preprocessor:
1.Drop Column
2.Missing Value
3.Convert Categorical to Numerical
4.Normalization
'''
import csv
import numpy as np
import pandas as pd
from collections import Counter
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_int64_dtype


def mergedata():
	# this is only for merging dataset 2
	list_ = []
	frame = pd.DataFrame
	for i in range(11):
		filename = '../data/dataset2/setapProcessT' + str(i + 1) + '.csv'
		data = pd.read_csv(filename, header=63)
		list_.append(data)
	frame = pd.concat(list_)
	frame.to_csv('../data/raw/dateset2.csv')


def drop_column(df, column=[], label=''):
	# drop useless column and non labeled row
	# label_column = df[label].to_frame(name='label')
	label_column = df[label]
	df = df[pd.notnull(df[label])]
	column.append(label)
	df.drop(column, inplace=True, axis=1)
	return df, label_column


def missing_value(df, methods='0'):
	# filling missing value with 0 or avg or median
	# column name list and it's missing value list 'avg' or '0'
	nan_c = df.columns[df.isnull().any()].tolist()
	if methods == '0':
		df.fillna(0, inplace=True)
	if methods == 'median':
		for c in nan_c:
			if is_float_dtype(df[c]):
				med = df[c].median()
				df[c].fillna(med,inplace=True)
	return df

def categorical_to_numerical(df):
	# convet categorical feature to numerical feature
	df = pd.get_dummies(df)
	return df


def categorical_to_numerical_ii(df):
	# convet categorical feature to numerical feature into 1 column
	def get_item_list(l):
		item_list = Counter(l).keys()
		return item_list
	for c in df.columns:
		if is_string_dtype(df[c]):
			# only convert string type
			i_list = get_item_list(df[c])
			for n,i in enumerate(i_list):
				df[c] = df[c].apply(lambda x: int(n) if x == i else x)
			# df[c] = df[c].apply(lambda x: x.cat.codes)
	return df


def nomalization(df, methods='min-max'):
	# nomalization the preprocessed columns from dataset
	# methods contain mean, min, and max.
	for c in df.columns:
		# I only convert float type because we are not sure whether integer is categorical or numerical
		# if is_numeric_dtype(df[c]):
		if is_float_dtype(df[c]):
			if methods == 'mean':
				df[c] = (df[c] - df[c].mean()) / df[c].std()
			if methods == 'min-max':
				df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
	return df


def main(filename, drop_columns=[], label='', missing_methods='0', norm_methods='min-max'):
	# read file into pandas dataframe
	data = pd.read_csv('../data/raw/'+filename+'.csv')

	data, label_column = drop_column(df = data, column = drop_columns, label = label)
	
	data = missing_value(df = data, methods = missing_methods)
	
	data = nomalization(df = data, methods = norm_methods)

	#data = categorical_to_numerical(df = data)

	#data = pd.con([data,label_column])
	data = pd.concat([data, label_column], axis=1, join_axes=[data.index])

	data = categorical_to_numerical_ii(df = data)
	
	data.to_csv("../data/preprocessed_data/"+filename+missing_methods+norm_methods+'categorize_ii'+'.csv',index=False)

	'''
	
	
	'''
if __name__ == '__main__':
	# main(filename = 'dataset2', label = 'processLetterGrade', missing_methods = 'median', norm_methods = 'mean')
	d = {'dataset1_math':'G3','dataset1_portuguese':'G3','dataset2':'processLetterGrade','dataset3':'Class'}
	miss1 = ['0','median']
	norm2 = ['min-max','mean']
	#for k,v in d.items():
	k = 'dataset1_portuguese'
	v = 'G3'
	for m in miss1:
		for n in norm2:
			main(filename = k, label = v, missing_methods = m, norm_methods = n)
				#print(k,v,m,n)
	'''
	df = pd.read_csv('../data/raw/dataset2.csv')
	nan_c = df.columns[df.isnull().any()].tolist()
	print(df.dtypes.value_counts())
	for c in df.columns:
		if is_string_dtype(df[c]):
			print(c)
	# print(nan_c)
	'''



