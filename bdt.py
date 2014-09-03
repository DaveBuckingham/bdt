#!/usr/bin/python

import numpy as np
import sys
from sklearn import tree
from sklearn import linear_model

from sklearn.externals.six import StringIO

VALIDATION  = True
USE_TOY     = True
USE_PILLOW  = True

TREE_DEPTH  = 3   # only used if not VALIDATION
MAX_DEPTH   = 10  # only used if VALIDATION


VERBOSE     = False
PLOT        = False
EXPORT_TREE = False

if (len(sys.argv) < 2):
    print "usage: draw.py <input_file>"
    sys.exit(1)

input_index = 1

while (input_index < len(sys.argv)):

    training_dates = []
    training_aswes = []
    training_pswes = []
    training_toys = []

    validation_dates = []
    validation_aswes = []
    validation_pswes = []
    validation_toys = []

    testing_dates = []
    testing_aswes = []
    testing_pswes = []
    testing_toys = []

    trees = []

    input_file = sys.argv[input_index]
    input_index = input_index + 1
    in_fh = open(input_file, 'r')


####################################################
#                READ TRAINING DATA                #
####################################################

    line = in_fh.readline()
    if (line != "TRAINING\n"):
	print ("error. expected TRAINING, read " + line) 
	sys.exit(1)

    line = in_fh.readline()
    if (line != "DATE,AREAL_SWE,POINT_SWE,TOY\n"):
	print ("error. expected header, read " + line) 
	sys.exit(1)

    line = in_fh.readline()
    while (line != "\n"):
	line.rstrip('\n')
	date, areal_swe, point_swe, toy = line.split(",")
	training_dates.append(date)
	training_aswes.append(float(areal_swe))
	training_pswes.append(float(point_swe))
	training_toys.append(int(toy))
	line = in_fh.readline()


####################################################
#              READ VALIDATION DATA                #
####################################################

    if (VALIDATION):

	line = in_fh.readline()
	if (line != "VALIDATION\n"):
	    print ("error. expected VALIDATION, read " + line) 
	    sys.exit(1)

	line = in_fh.readline()
	if (line != "DATE,AREAL_SWE,POINT_SWE,TOY\n"):
	    print ("error. expected header, read " + line) 
	    sys.exit(1)

	line = in_fh.readline()
	while (line != "\n"):
	    line.rstrip('\n')
	    date, areal_swe, point_swe, toy = line.split(",")
	    validation_dates.append(date)
	    validation_aswes.append(float(areal_swe))
	    validation_pswes.append(float(point_swe))
	    validation_toys.append(int(toy))
	    line = in_fh.readline()



####################################################
#                 READ TESTING DATA                #
####################################################

    line = in_fh.readline()
    if (line != "TEST\n"):
	print ("error. expected TEST, read " + line) 
	sys.exit(1)

    line = in_fh.readline()
    if (line != "DATE,AREAL_SWE,POINT_SWE,TOY\n"):
	print ("error. expected header, read " + line) 
	sys.exit(1)

    line = in_fh.readline()
    while (line != "\n"):
	line.rstrip('\n')
	date, areal_swe, point_swe, toy = line.split(",")
	testing_dates.append(date)
	testing_aswes.append(float(areal_swe))
	testing_pswes.append(float(point_swe))
	testing_toys.append(int(toy))
	line = in_fh.readline()


####################################################
#                 CREATE MODELS                    #
####################################################

    if (USE_PILLOW and USE_TOY):
	X = np.matrix((training_pswes, training_toys)).T
	X_test = np.matrix((testing_pswes, testing_toys)).T
    elif (USE_PILLOW):
	X = np.matrix((training_pswes)).T
	X_test = np.matrix((testing_pswes)).T
    elif (USE_TOY):
	X = np.matrix((training_toys)).T
	X_test = np.matrix((testing_toys)).T
    else:
	print "no independent variables!"
	sys.exit(1)


    y = np.matrix(training_aswes).T



    if (VALIDATION):
	for tree_size in range(1,MAX_DEPTH):

    else:
    clf = tree.DecisionTreeRegressor(max_depth=TREE_DEPTH)
    clf.fit(X, y)


    reg = linear_model.LinearRegression()
    reg.fit(X, y)


    bdt_predictions = clf.predict(X_test).T
    lin_predictions = reg.predict(X_test).T

    bdt_error = abs(testing_aswes - bdt_predictions)
    lin_error = abs(testing_aswes - lin_predictions)


####################################################
#                       OUTPUT                     # 
####################################################

    if (VERBOSE):
	print('Coefficients: \n', reg.intercept_, reg.coef_)
	print "station: "
	print testing_pswes
	print "ground_truth: "
	print testing_aswes
	print "predicted: "
	print lin_predictions
	print "difference: "
	print abs(testing_aswes - lin_predictions)
	print "mean: "

    #print np.mean(bdt_error), np.mean(lin_error)
    print np.mean(bdt_error)

    if (PLOT):
	import matplotlib.pyplot as plt

	predicted = bdt_predictions
	plt.figure()
	plt.scatter(training_pswes, training_aswes, c="w", marker="p", s=120, label="train_data")
	plt.scatter(testing_pswes, testing_aswes, c="w", s=40, marker="s", label="test_data")
	plt.scatter(testing_pswes, predicted, c="k", marker="o", s=8, label="predicted")

	plt.xlabel("point_swe")
	plt.ylabel("snowcloud_swe")
	plt.title(input_file)
	plt.legend()
	plt.show()

    if (EXPORT_TREE):
	from StringIO import StringIO
	out = StringIO()
	out = tree.export_graphviz(clf, out_file='tree.dot')
