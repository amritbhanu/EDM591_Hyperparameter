import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC


def mergedata():
    list_ = []
    frame = pd.DataFrame
    for i in range(11):
        filename = '../data/dataset2/setapProcessT' + str(i + 1) + '.csv'
        data = pd.read_csv(filename, header=63)
        list_.append(data)
    frame = pd.concat(list_)
    frame.to_csv('../data/dataset2/mergedata.csv')


def svm(df, label, k, c):
    # k = kernel, c = C
    clf = SVC(kernel=k, C=c)
    label = label.astype(np.int)
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(df, label):
        x_train, x_test = df[train_index], df[test_index]
        y_train, y_test = label[train_index], label[test_index]
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        result.append(f1_score(y_test, predict))
    print('SVM 10 cross validation F1-score Results:')
    print(np.median(result))


def rf(df, label, n=10, m_feature='auto', m_dep=None, m_sam_s=2):
    clf = RandomForestClassifier(n_estimators=n, max_features=m_feature, max_depth=m_dep, min_samples_split=m_sam_s)
    # 10 cross validation
    label = label.astype(np.int)
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(df, label):
        x_train, x_test = df[train_index], df[test_index]
        y_train, y_test = label[train_index], label[test_index]
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        result.append(f1_score(y_test, predict))
    print('Random Forest Tree 10 cross validation F1-score Results:')
    print(np.median(result))


def main():
    # df = np.load('mergedata.npy')
    df = pd.read_csv('../data/dataset2/mergedata.csv')
    df.drop(df.columns[:6], inplace=True, axis=1)
    df.drop(['globalLeadAdminHoursTotal', 'globalLeadAdminHoursAverage', 'globalLeadAdminHoursStandardDeviation',
             'averageGlobalLeadAdminHoursTotalByWeek', 'standardDeviationGlobalLeadAdminHoursTotalByWeek',
             'averageGlobalLeadAdminHoursAverageByWeek', 'standardDeviationGlobalLeadAdminHoursAverageByWeek'],
            inplace=True, axis=1)
    df = df[pd.notnull(df['processLetterGrade'])]
    df = pd.get_dummies(df, columns=['teamLeadGender', 'teamDistribution'])
    # normalization
    df.fillna(0, inplace=True)
    df['processLetterGrade'] = df['processLetterGrade'].apply(lambda x: 1 if x == 'A' else 0)
    label = df['processLetterGrade']
    df.drop('processLetterGrade', inplace = True, axis = 1)
    df['label']=label
    df.to_csv("../data/dataset2.csv",index=False)
    # label = label.copy()
    # label[label == 'A'] = 1
    # label[label == 'F'] = 0
    # #fill missing value
    # #df.fillna(df.mean(),inplace=True)
    # print(df.head())
    # np.save('../data/dataset2/mergedata', df)
    # df.to_csv('merge.csv')

    # np.save('../data/dataset2/label',label)

if __name__ == '__main__':
    mergedata()
    main()
    # data = np.load('../data/dataset2/mergedata.npy')
    # np.savetxt("dataset2.csv", data, delimiter=",")
    # label = np.load('../data/dataset2/label.npy')
    # np.savetxt("dataset2_label.csv", label, delimiter=",")
    # svm(data, label, 'linear', 1)
    # rf(data, label, 10)
