from typing import List, Union, Any

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
from datetime import datetime
import seaborn as sns
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import math
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score, LeaveOneGroupOut

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

path = './data/'
from data_helper import ensure_data_and_features
foldersMIOS = ensure_data_and_features(path)
fileFeatures = 'featuresRAW-mios.csv'
featuresScalatedCut = 'featur-scale-cut-mios.csv'
maxsamples = 446 #This is the lowest number in one of activity with lower samples (and we cut all the activities with this number). This activity is for M-008 in us).

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT-CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []




def scaler(X):
    ssc = StandardScaler()
    X = ssc.fit_transform(X)
    return X

def SVN_acc():
    # instantiate the estimator
    svm = SVC()
    # fit the model
    svm.fit(X_train, y_train)
    # predict the response
    pred_svm = svm.predict(X_test)
    # accuracy score
    svm_acc = accuracy_score(y_test, pred_svm)
    return pred_svm, svm_acc
    print ("Accuracy for SVM: {}".format(svm_acc))

def NB_acc():
    # instantiate the estimator
    nb = GaussianNB()
    # fit the model
    nb.fit(X_train, y_train)
    # predict the response
    pred_nb = nb.predict(X_test)
    # accuracy score
    nb_acc = accuracy_score(y_test, pred_nb)
    print ("Accuracy for Gaussian Naive Bayes: {}".format(nb_acc))

def KNN_acc():
    # instantiate the estimator
    knn = KNeighborsClassifier()
    # fit the model
    knn.fit(X_train, y_train)
    # predict the response
    pred_knn = knn.predict(X_test)
    # accuracy score
    knn_acc = accuracy_score(y_test, pred_knn)
    print ("Accuracy for KNN: {}".format(knn_acc))

def scaleBalanced ():
    filelist=[]
    for fichero in os.listdir(path):
        #print(fichero)

        if fichero in foldersMIOS:
            filelist.append(path+fichero + '/' + fileFeatures)
            featu = pd.read_csv(path+fichero + '/' + fileFeatures)
            activities = ['r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
            #for activity in ['r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']:
                #row.append(len(featu[featu['activity'] == activity].index.to_list()))
                #featu = featu.groupby(['activity']).tail(446) #### FUNCIONA
                #featu = featu[featu['activity'] == activity].groupby(['activity']).tail(446) #### NO FUNCIONA
            #featu = featu.loc[featu['activity'].isin(activities)].groupby(['activity']).tail(446) #### FUNCIONA
            featu = featu.loc[featu['activity'].isin(activities)].groupby(['activity']).tail(646)
            featu = featu.groupby(['activity']).head(446)
            featu[['x', 'y', 'z', 'hr', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio','HRR']] = scaler(featu[['x', 'y', 'z', 'hr', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio','HRR']])
            #dfTest[['A', 'B']] = scaler.fit_transform(dfTest[['A', 'B']])
            featu.to_csv(path+fichero + '/' + featuresScalatedCut, index=False)
            #print(path + fichero)
    #print(filelist)

def split():
    filelist = []
    dataframes_list =[]
    for fichero in os.listdir(path):
        # print(fichero)
        if fichero in foldersMIOS:
            filelist.append(fichero)
            #a ="pd_"
            #exec(f'pd_{fichero} = pd.read_csv(path + fichero + \"/\" + featuresScalatedCut)' )
            #print(f'{a}{fichero}')
            #print(globals()[f"df_{fichero}"])
            #f'pd_{}.format(fichero)' = pd.read_csv(path + fichero + '/' + featuresScalatedCut)
            temp_df = pd.read_csv(path + fichero + '/' + featuresScalatedCut) #FUNCIONA, pero no veo donde guarda la varibable
            dataframes_list.append(temp_df)
            #{f'df_{fichero}'} = pd.read_csv(path + fichero + '/' + featuresScalatedCut)
            #print(globals()[f"df_{fichero}"])
            #'{}_{}'.format('pd', fichero)
            #print("Hola")
            #a = f'string{fichero}'
            #a = pd.read_csv(path + fichero + '/' + featuresScalatedCut)
            #[f'pd'+fichero] = pd.read_csv(path + fichero + '/' + featuresScalatedCut)
    loo=LeaveOneOut()
    print(filelist)
    print(dataframes_list)
    for train,test in loo.split(dataframes_list):
        test = dataframes_list[test.item(0)]
        #y_test = dataframes_list(test)
        y_test = test['activity']
        X_test = test.loc[:,['x', 'y', 'z']]
        train2 = dataframes_list[train.item(0):train.item(len(train)-1)]
        y_train = dataframes_list[train,:][: 'activity']
        X_train = dataframes_list[train, :][:, 'x','y', 'z']



        one = 1
        #featu = pd.read_csv(train)
        #pred_svn, svn_acc = SVN_acc()
        print("%s %s" % (train, test))

def modelsrun():
    logocv = LeaveOneGroupOut()
    X_train=pd.DataFrame
    dataset=[]
    y_train=pd.DataFrame
    groups=[]
    i = 0
    for fichero in os.listdir(path):

        if fichero in foldersMIOS:
            temp_df = pd.read_csv(path + fichero + '/' + featuresScalatedCut)
            dataset.append(temp_df)
            personlen = len(temp_df)
            value = i
            group = np.empty(personlen)
            group.fill(value)
            groups = np.concatenate((groups, group), axis=None)
            #groupAp.append(group)
            i += 1
            #X_train.append(temp_df)
    frame = pd.concat(dataset)
    #groups = pd.concat(groupAp)
    #frames = [dataset, temp_df]
    X_train = frame.loc[:,['smvfil', 'ratio']]
    y_train = frame.loc[:,["activity"]]

    print('hola')


    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train.values.ravel(), cv=logocv, groups=groups, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


scaleBalanced()
modelsrun()