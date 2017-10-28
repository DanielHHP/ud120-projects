#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn.cross_validation import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(feature_train, label_train)

print 'score:', clf.score(feature_test, label_test)
print 'poi cnt in test data:', sum(label_test)
print 'test data size:', len(label_test)

predicts = clf.predict(feature_test)
tf_count = 0
for pred, label in zip(predicts, label_test):
    if pred == 1.0 and label == 1.0:
        tf_count += 1
print 'TF for overfit model:', tf_count

from sklearn.metrics import precision_score, recall_score
print 'precision:', precision_score(label_test, predicts)
print 'recall:', recall_score(label_test, predicts)

