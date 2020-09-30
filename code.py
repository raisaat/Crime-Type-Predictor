# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:51:50 2020

"""
# Raisaat Rashid
# Harichandana Yeleswaram

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedKFold 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
                
def plotGraph (x, y1, y2):  
    # plotting the error against tree depth  
    plt.plot(x, y1,'r', label = 'on training data') 
    plt.plot(x, y2, 'b', label = 'on test data') 
    plt.xlabel('tree depth') 
    plt.ylabel('error')
    plt.legend() 
    plt.show()
    
def computeConfusionMatrix(true_labels, pred_labels):
   
    num_labels = len(np.unique(true_labels))
    
    conf_matrix = confusion_matrix(true_labels, pred_labels)
        
    print(conf_matrix)
    
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=( num_labels, num_labels))
    plt.show()

if __name__ == '__main__':
    colnames=["Day","crime type","Time","label"]

    X = pd.read_csv("numerical_data1.csv",names=colnames)
    
    y = X['crime type']
    X = X.drop('crime type', axis=1)
        
    X = X.to_numpy()
    y = y.to_numpy()
    
    state = 42
    #n_classes = 9
    
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=state)
    
#    print("\n\nBagging:")
#    for i in (7, 8, 9, 10, 11):
#        for j in (10, 20, 30):
#            bagging_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=i),n_estimators=j)
#            scores = cross_val_score(bagging_clf, X, y, cv=kf)
#            print ('Mean accuracy for max depth = ', i, ' num of estimators = ', j, ': {0:4.2f}%.'.format(scores.mean() * 100))
#            
    print("\n\nGradient Boosting:")    
    lr_list = [0.05, 0.075, 0.1, 0.25]
    n_estimators_list = [20, 30, 45, 60, 100]
    acc=[]
    for learning_rate in lr_list:
        for num_estimators in n_estimators_list:
            gb_clf = GradientBoostingClassifier(n_estimators=num_estimators, learning_rate = learning_rate, max_features=2, max_depth = 3, random_state = state)
            scores = cross_val_score(gb_clf, X, y, cv=kf)
            print ('Mean accuracy for learning rate = ', learning_rate, ' num of estimators = ', num_estimators, ': {0:4.2f}%.'.format(scores.mean() * 100))
            acc.append(scores.mean())
            
        
            
    maxi = acc.index(max(acc))
    best_lr = int(maxi/len(n_estimators_list))
    best_n_est = int(maxi%len(n_estimators_list))
    print("\nMaximum accuracy obtained with learning rate =",lr_list[best_lr],"and num of estimators =",n_estimators_list[best_n_est],"and the accuracy is {0:4.2f}%.".format(acc[maxi] * 100))
    
    print("\n\nSVM:")
    c_vals = [0.01, 0.1,1,10,100,1000]
    gamma_vals = [10, 100, 1000]
    acc=[]
    for c in c_vals:
        for g in gamma_vals:
            svm_clf = SVC(kernel='rbf', C = c, gamma=g)
            scores = cross_val_score(svm_clf, X, y, cv=kf)
            print ('Mean accuracy for C = ', c, 'gamma =', g, ': {0:4.2f}%.'.format(scores.mean() * 100))
            acc.append(scores.mean())
    
    maxi = acc.index(max(acc))
    best_c = int(maxi/len(gamma_vals))
    best_g = int(maxi%len(gamma_vals))
    print("\nMaximum accuracy obtained with C =",c_vals[best_c],"and gamma =",gamma_vals[best_g],"and the accuracy is {0:4.2f}%.".format(acc[maxi] * 100))
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=state)
    
    # Fit models on specific training and tetsing sets
    
    # Gradient boosting training with the best parameters
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators_list[best_n_est], learning_rate =lr_list[best_lr], max_features=2, max_depth = 3, random_state = state)
    gb_clf.fit(Xtrn, ytrn)
    y_pred = gb_clf.predict(Xtst)
    computeConfusionMatrix(ytst, y_pred)
    #SVM training with the best parameters
    svm_clf = SVC(kernel='rbf', C = c_vals[best_c], gamma=gamma_vals[best_g])
    svm_clf.fit(Xtrn, ytrn)
    y_pred = svm_clf.predict(Xtst)
    computeConfusionMatrix(ytst, y_pred)
    
    y = label_binarize(y, classes=[ 1, 2,3,4,5,6,7,8,9])
    n_classes = y.shape[1]

    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=0)
    
    classifier = OneVsRestClassifier(gb_clf)
    
    
    y_score = classifier.fit(Xtrn, ytrn).predict_proba(Xtst)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytst[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    
    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class(Gradient Boosting)')
    plt.legend(loc="lower right")
    plt.show()
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(ytst[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(ytst[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(ytst.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(ytst, y_score,
                                                         average="micro")
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Gradient Boosting Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    
    classifier = OneVsRestClassifier(SVC(kernel='rbf', probability=True,C = c_vals[best_c], gamma=gamma_vals[best_g]))
    y_score = classifier.fit(Xtrn, ytrn).predict_proba(Xtst)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytst[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    
    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class(SVM)')
    plt.legend(loc="lower right")
    plt.show()
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(ytst[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(ytst[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(ytst.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(ytst, y_score,
                                                         average="micro")
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'SVM Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    
    
    
    
    
    
    
