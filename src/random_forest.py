import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

def rf_reg(df, target, n = 10, m_feature = 'auto', m_dep = None, m_sam_s = 2):
	#return MSE
	clf = RandomForestRegressor(n_estimators = n, max_features = m_feature, max_depth = m_dep, min_samples_split = m_sam_s)
	# 10 cross validation
	target = target.astype(np.float)
	result = []
	kf = StratifiedKFold(n_splits = 10, shuffle = True)
	for train_index, test_index in kf.split(df, target):
		x_train, x_test = df[train_index], df[test_index]
		y_train, y_test = target[train_index], target[test_index]
		clf.fit(x_train, y_train)
		predict = clf.predict(x_test)
		result.append(mean_squared_error(y_test, predict))
	print('Random Forest Tree 10 cross validation MSE Results:')
	print(np.median(result))

def pre():
	df = pd.read_csv('../data/dataset1.csv')
	# Drop g1 and g2 because they are highly corelated with final score.
	df.drop(['G1','G2'], inplace = True, axis = 1)
	G3 = df['G3']
	G3 = G3.copy()
	df.drop('G3', inplace = True, axis = 1)
	np.save('../data/dataset1',df)
	np.save('../data/dataset1_G3',G3)

def main():
	df = np.load('../data/dataset1.npy')
	G3 = np.load('../data/dataset1_G3.npy')
	rf_reg(df, G3)

if __name__ == '__main__':
	#pre()
	main()
