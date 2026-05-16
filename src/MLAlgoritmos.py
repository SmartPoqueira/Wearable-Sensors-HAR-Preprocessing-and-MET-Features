import numpy as np
import pandas as pd
import os

import plotly.io
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from scipy.fft import fft, fftfreq
import numpy as np

import csv


path = './Data/'
folders = []
# F-045 es de MJ, y está mal etiquetado
#foldersMIOS = ['F-014', 'F-046', 'M-003', 'M-004', 'M-007']

# foldersMIOS = ['F-014', #Carreras
#                'F-045', #MJ
#                'F-046', #Ada
#                'F-047', #yo
#                'M-003', #Art
#                'M-004', #Oc
#                'M-007', #Simo
#                'M-008'] #Alb

foldersMIOS = ['F-014', #Carreras
               #'F-045', #MJ
               'F-046', #Ada
               'F-047', #yo
               'M-003', #Art
               'M-004', #Oc
               'M-007', #Simo
               'M-008'] #Alb

#Allactivities = ['lie', 'r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
activitySet = ['lie', 'r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
#activitySet = ['d', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']

#activitySet = ['d', 'wd', 'wr', 'ds-us']


#featuresSel = ['x', 'y', 'z', 'smv', 'smvfil', 'ratio', 'HRR', 'hr', 'smv', 'xfil', 'yfil', 'zfil']
featuresSel = ['smvfil', 'ratio', 'HRR']
#featuresSel = ['smvfil', 'ratio']
#featuresSel = ['x', 'y', 'z', 'smv']
#featuresSel = ['x', 'y', 'z']
#featuresSel = ['xfil', 'yfil', 'zfil', 'smvfil']
fileFeatures = 'featuresRAW-mios.csv'
featuresScalatedCut = 'featur-scale-cut-mios.csv'
#featuresScalatedCut = 'featur-scale-cut-mios-d-wd-wr-stairs.csv'
maxsamples = 446 #This is the lowest number in one of activity with lower samples (and we cut all the activities with this number). This activity is for M-008 in us).
botttomSamples = 646 #We take only the samples of the very end of each activity, and them take out 200 samples
# (a couple of seconds; 646-446=200). because the user spend sometime to ckick the bottom and mark the end of the activity

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT-CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []

def filtro():
    # Number of sample points
    colnames = ['x','y','z','timestamp']
    dfacc = pd.read_csv('data/F-014/ACC.csv', names=colnames, header=None)
    dfacc['x'] = dfacc['x'].astype('int')
    dfacc['y'] = dfacc['y'].astype('int')
    dfacc['z'] = dfacc['z'].astype('int')
    sample_rate = 32
    init_acc_timestamp = dfacc.iloc[(0,0)]
    dfacc.drop([0,1], inplace=True)
    dfacc.reset_index(inplace=True, drop=True)
    N = dfacc.shape[0]
    T = 1.0 / sample_rate

    yfx = np.array(dfacc['x'])
    yfx = fft(dfacc['x'])
    yfy = fft(dfacc['y'])
    yfz = fft(dfacc['z'])
    xf = fftfreq(N, T)[:N//2]

    return yfx, yfx, yfy, yfz, xf, N



def scaler(X):
    ssc = StandardScaler()
    X = ssc.fit_transform(X)
    return X

def scaleBalanced ():
    #Create a new column with the activities as numbers (to apply ML), and only takes the activities the "maxsamples":
    # number of samples per activity, because these are the samples in the user/activity with less samples,
    # which in our case is M-008 in the activity of "ds" (maybe it was "us").
    #We take this samples from the end of the activity (but leaving out 200 samples (some seconds) of the very end,
    # because the user spend sometime to ckick the bottom and mark the end of the activity.
    filelist=[]
    for fichero in os.listdir(path):
        if fichero in foldersMIOS:
            filelist.append(path+fichero + '/' + fileFeatures)
            featu = pd.read_csv(path+fichero + '/' + fileFeatures)
            activities = ['r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
            featu = featu.loc[featu['activity'].isin(activities)].groupby(['activity']).tail(botttomSamples)
            featu = featu.groupby(['activity']).head(maxsamples)
            featu[['x', 'y', 'z', 'hr', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio','HRR']] = scaler(featu[['x', 'y', 'z', 'hr', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio','HRR']])
            featu.to_csv(path+fichero + '/' + featuresScalatedCut, index=False)

def modelsrun():
    # Returns the name of the ML algorithm with the result in terms of accuracy.
    logocv = LeaveOneGroupOut()
    X_train=pd.DataFrame
    dataset=[]
    y_train=pd.DataFrame

    groups=[]
    i = 0
    # We merge all the data (from all the persons) in one file and add a new column that have a number with the number of person.
    for fichero in os.listdir(path):
        if fichero in foldersMIOS:
            temp_df = pd.read_csv(path + fichero + '/' + featuresScalatedCut)
            temp_df = temp_df.loc[temp_df['activity'].isin(activitySet)]
            dataset.append(temp_df)
            personlen = len(temp_df)
            value = i
            group = np.empty(personlen)
            group.fill(value)
            groups = np.concatenate((groups, group), axis=None)
            i += 1
    frame = pd.concat(dataset)
    X_train = frame.loc[:,featuresSel]
    y_train = frame.loc[:,["activity"]]

    for name, model in models:

        #Hacemos el cross-validation con la columna "value" que tiene el numero de cada persona, para poder hacer leave-one-group-out (dejar una persona fuera cada vez).

        cv_results = cross_val_score(model, X_train, y_train.values.ravel(), cv=logocv, groups=groups, scoring='accuracy')
        # Para varias métricas (cross_validate): scoring = ['precision_macro', 'recall_macro'], y luego scoring=scoring
        #cv_results = cross_validate(model, X_train, y_train.values.ravel(), cv=logocv, groups=groups, scoring=("roc_auc_ovo", "accuracy"),
        #                return_train_score=True)
        #print(cv_results)

            # y_pred = cross_val_predict(model, X_train, y_train.values.ravel(), cv=logocv, groups=groups)
            # conf_mat = confusion_matrix(y_train.values.ravel(), y_pred)
            # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=activitySet)
            # disp.plot()
            # disp.ax_.set_title(name)
            # plt.show()
        # print(conf_mat)
        results.append(cv_results)
        names.append(name)
        #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    return names, results

#scaleBalanced()
#
[names,resultsAv] = modelsrun()
    #
print(resultsAv)



b = open('./accuracies/output.csv', 'w')
a = csv.writer(b)
a.writerows(resultsAv)

b.close()

plt.boxplot(resultsAv)
plt.xticks(np.arange(len(names))+1, names)
plt.show()


# ESTO ES PARA VER A QUE FRECUENCIA LLEGA EL FFT.... (llega a 16Hz).
# [yfx, yfx, yfy, yfz, xf, N] = filtro()
#
# plt.plot(xf, 2.0/N * np.abs(yfx[0:N//2]))
# plt.grid()
# plt.ylim((0,2))
# plt.show()
# # plt.plot(xf, 2.0/N * np.abs(yfy[0:N//2]))
# # plt.grid()
# # plt.ylim((0,2))
# # plt.show()
# # plt.plot(xf, 2.0/N * np.abs(yfz[0:N//2]))
# # plt.grid()
# # plt.ylim((0,2))
# # plt.show()

#print(sklearn.metrics.classification_report(y_test,logreg_predictions))
