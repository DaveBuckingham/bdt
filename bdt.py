#!/usr/bin/python

import numpy as np
import sys
from sklearn import tree
from sklearn import linear_model

from sklearn.externals.six import StringIO

if (len(sys.argv) < 2):
    print "usage: draw.py <input_file>"
    sys.exit(1)

input_index = 1

while (input_index < len(sys.argv)):


    training_dates = []
    training_aswes = []
    training_pswes = []
    training_toys = []

    testing_dates = []
    testing_aswes = []
    testing_pswes = []
    testing_toys = []




    input_file = sys.argv[input_index]
    input_index = input_index + 1

    in_fh = open(input_file, 'r')

    line = in_fh.readline()
    if (line != "TRAINING\n"):
	print ("error. expected TRAINING, read " + line) 

    line = in_fh.readline()
    if (line != "DATE,AREAL_SWE,POINT_SWE,TOY\n"):
	print ("error. expected header, read " + line) 

    line = in_fh.readline()
    while (line != "\n"):
	line.rstrip('\n')
	date, areal_swe, point_swe, toy = line.split(",")
	training_dates.append(date)
	training_aswes.append(float(areal_swe))
	training_pswes.append(float(point_swe))
	training_toys.append(int(toy))
	line = in_fh.readline()

    line = in_fh.readline()
    if (line != "TEST\n"):
	print ("error. expected TEST, read " + line) 

    line = in_fh.readline()
    if (line != "DATE,AREAL_SWE,POINT_SWE,TOY\n"):
	print ("error. expected header, read " + line) 

    line = in_fh.readline()
    while (line != "\n"):
	line.rstrip('\n')
	date, areal_swe, point_swe, toy = line.split(",")
	testing_dates.append(date)
	testing_aswes.append(float(areal_swe))
	testing_pswes.append(float(point_swe))
	testing_toys.append(int(toy))
	line = in_fh.readline()


    #X = np.matrix((training_pswes)).T
    #X_test = np.matrix((testing_pswes)).T

    #X = np.matrix((training_toys)).T
    #X_test = np.matrix((testing_toys)).T

    X = np.matrix((training_pswes, training_toys)).T
    X_test = np.matrix((testing_pswes, testing_toys)).T

    y = np.matrix(training_aswes).T


    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf.fit(X, y)

    from StringIO import StringIO
    out = StringIO()
    out = tree.export_graphviz(clf, out_file='tree.dot')

    reg = linear_model.LinearRegression()
    reg.fit(X, y)
#   print('Coefficients: \n', reg.intercept_, reg.coef_)



    bdt_predictions = clf.predict(X_test).T
    lin_predictions = reg.predict(X_test).T

    bdt_error = abs(testing_aswes - bdt_predictions)
    lin_error = abs(testing_aswes - lin_predictions)


#   print "station: "
#   print testing_pswes
#   print "ground_truth: "
#   print testing_aswes
#   print "predicted: "
#   print lin_predictions
#   print "difference: "
#   print abs(testing_aswes - lin_predictions)
#   print "mean: "

#   print np.mean(bdt_error), np.mean(lin_error)

    print np.mean(bdt_error)


#   import matplotlib.pyplot as plt

#   plt.figure()
#   plt.scatter(training_pswes, training_aswes, c="w", marker="p", s=120, label="train_data")
#   plt.scatter(testing_pswes, testing_aswes, c="w", s=40, marker="s", label="test_data")
#   plt.scatter(testing_pswes, predicted, c="k", marker="o", s=8, label="predicted")

#   plt.xlabel("point_swe")
#   plt.ylabel("snowcloud_swe")
#   plt.title(input_file)
#   plt.legend()
#   plt.show()
