import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

def mergedata():
	list_ = []
	frame = pd.DataFrame
	for i in range(11):
		filename = 'setapProcessT'+str(i+1)+'.csv'
		data = pd.read_csv(filename, header = 63)
		list_.append(data)
	frame = pd.concat(list_)
	frame.to_csv('mergedata.csv')

def svm(df, label, k, c):
	#k = kernel, c = C
	clf = SVC(kernel = k, C = c)
	label = label.astype(np.int)
	result = []
	kf = KFold(n_splits = 10, shuffle = True)
	for train_index, test_index in kf.split(df):
		x_train, x_test = df[train_index], df[test_index]
		y_train, y_test = label[train_index], label[test_index]
		clf.fit(x_train, y_train)
		predict = clf.predict(x_test)
		result.append(f1_score(y_test, predict))
	print('SVM 10 cross validation F1-score Results:')
	print(np.median(result))

def rf(df, label, n = 10, m_feature = 'auto', m_dep = None, m_sam_s = 2):
	clf = RandomForestClassifier(n_estimators = n, max_features = m_feature, max_depth = m_dep, min_samples_split = m_sam_s)
	# 10 cross validation
	label = label.astype(np.int)
	result = []
	kf = KFold(n_splits = 10, shuffle = True)
	for train_index, test_index in kf.split(df):
		x_train, x_test = df[train_index], df[test_index]
		y_train, y_test = label[train_index], label[test_index]
		clf.fit(x_train, y_train)
		predict = clf.predict(x_test)
		result.append(f1_score(y_test, predict))
	print('Random Forest Tree 10 cross validation F1-score Results:')
	print(np.median(result))

def main():
	#df = np.load('mergedata.npy')
	df = pd.read_csv('mergedata.csv')
	df.drop(df.columns[:6], inplace = True, axis = 1)
	df.drop(['globalLeadAdminHoursTotal','globalLeadAdminHoursAverage','globalLeadAdminHoursStandardDeviation','averageGlobalLeadAdminHoursTotalByWeek','standardDeviationGlobalLeadAdminHoursTotalByWeek','averageGlobalLeadAdminHoursAverageByWeek','standardDeviationGlobalLeadAdminHoursAverageByWeek'], inplace = True, axis = 1)
	df = df[pd.notnull(df['processLetterGrade'])]
	df = pd.get_dummies(df, columns = ['teamLeadGender','teamDistribution'])
	# normalization
	df.fillna(0, inplace = True)
	label = df['processLetterGrade']
	df.drop('processLetterGrade', inplace = True, axis = 1)
	label = label.copy()
	label[label == 'A'] = 1
	label[label == 'F'] = 0
	#fill missing value
	#df.fillna(df.mean(),inplace=True)
	print(df.head())
	np.save('mergedata',df)
	#df.to_csv('merge.csv')

	np.save('label',label)

if __name__ == '__main__':
	#mergedata()
	#main()
	data = np.load('mergedata.npy')
	label = np.load('label.npy')
	#svm(data, label, 'linear', 1)
	rf(data, label, 10)