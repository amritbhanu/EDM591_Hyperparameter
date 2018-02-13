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

def drop_column(df,column=[], label=''):
	# drop useless column and non labeled row
	df.drop(column, inplace=True, axis=1)
	df = df[pd.notnull(df[label])]
	return df
def missing_value(df, methods=[], column=[]):
	# filling missing value with 0 or avg or median
	# column name list and it's missing value list 'avg' or '0'
	for c,m in zip(column, methods):
		if m == '0':
			df[c] = fillna(0, inplace=True)
		if m == 'median':
			med = df[c].median()
			df[c] = fillna(med,inplace=True)
	return df

def categorical_to_numerical(df, column=[]):
	# convet categorical feature to numerical feature
	df = pd.get_dummies(df, columns=column)
	return df

def nomalization(df, column=[]):
	#nomalization the preprocessed columns from dataset
	if not column:
		return df
	for c in column:
		df[c] =  (df[c] - df[c].mean()) / (df[c].max() - df[c].min())
	return df


def main(filename, drop_column=[], label='', c_to_n_column=[], missing_column=[], methods=[], norm_column=[]):
	# read file into pandas dataframe
	data = pd.read_csv('../data/raw/'+filename)
	
	drop_column = []
	label = ''
	data = drop_column(data, drop_column, label)
	
	c_to_n_column = []
	data = categorical_to_numerical(data, c_to_n_column)
	
	missing_column = []
	methods = []
	data = missing_value(data, methods, missing_column)
	
	norm_column = []
	data = nomalization(data, norm_column)

if __name__ == '__main__':
	main()