#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# 1% data slice
def reduce_data():
    global features_train
    global labels_train
    features_train = features_train[:len(features_train)/100]
    labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

def linear_kernel():
    clf = svm.SVC(kernel="linear")
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t1 = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t1, 3), "s"
    acc = accuracy_score(pred, labels_test)
    print 'accuracy:', acc
    return clf

def rbf_kernel():
    clf = svm.SVC(kernel="rbf")
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t1 = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t1, 3), "s"
    acc = accuracy_score(pred, labels_test)
    print 'accuracy:', acc
    return clf


def rbf_kernel_different_C():
    print 'Should work with small data to reduce running time'
    for c in [10, 100, 1000, 10000]:
        clf = svm.SVC(kernel="rbf", C=c)
        clf.fit(features_train, labels_train)
        print "C:", c, "accuracy:", clf.score(features_test, labels_test)

def rbf_kernel_optimized():
    clf = svm.SVC(kernel="rbf", C=10000)
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t1 = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t1, 3), "s"
    acc = accuracy_score(pred, labels_test)
    print 'accuracy:', acc

    return clf

#reduce_data()
clf = rbf_kernel_optimized()

#for idx in [10, 26, 50]:
#    print 'idx:', idx, 'label:', clf.predict([features_test[idx]])
pred = clf.predict(features_test)
s = pd.Series(pred)
print 'labe 1 count:', s.value_counts()[1]

#########################################################


