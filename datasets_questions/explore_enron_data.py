#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'Person count:', len(enron_data)
print 'Feature count:', len(enron_data[enron_data.keys()[0]])

print 'POI count:', sum([enron_data[person]['poi']for person in enron_data])

print 'James Prentice features', enron_data['PRENTICE JAMES']
print 'Wesley Colwell features', enron_data['COLWELL WESLEY']
print 'Jeffrey K Skilling features', enron_data['SKILLING JEFFREY K']

# CEO SKILLING JEFFREY K
print '#############'
#print sorted(enron_data.keys())
print 'CEO'
print enron_data['SKILLING JEFFREY K']

# Chairman Kenneth Lay
print 'Chairman'
print enron_data['LAY KENNETH L']

# CFO Andrew Fastow
print 'CFO'
print enron_data['FASTOW ANDREW S']

# salary value count
valid_salary_count = 0
for person, feature in enron_data.iteritems():
    if feature['salary'] != 'NaN':
        valid_salary_count += 1
print 'valid_salary_count:', valid_salary_count

# email_address value count
valid_email_address_count = 0
for person, feature in enron_data.iteritems():
    if feature['email_address'] != 'NaN':
        valid_email_address_count += 1
print 'valid_email_address_count:', valid_email_address_count

# NaN total payments
nan_total_payments_count = sum([1 if feature_map['total_payments'] == 'NaN' else 0 for feature_map in enron_data.values()])
print 'NaN total_payments', nan_total_payments_count
print 'NaN total_payments ratio:', float(nan_total_payments_count) / len(enron_data)

# poid NaN total payments
poi_nan_total_payments_count = sum([1 if feature_map['total_payments'] == 'NaN' and feature_map['poi'] else 0 for feature_map in enron_data.values()])
print 'poi NaN total_payments', poi_nan_total_payments_count
print 'poi NaN total_payments ratio:', float(poi_nan_total_payments_count) / sum([enron_data[person]['poi']for person in enron_data])