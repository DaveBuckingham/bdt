#!/usr/bin/python

import numpy as np
import sys
from sklearn import tree
from sklearn import linear_model

from sklearn.externals.six import StringIO

VALIDATION  = True
#VALIDATION  = False

#USE_TOY     = True
USE_TOY     = False

USE_PILLOW  = True
#USE_PILLOW  = False

TREE_DEPTH  = 0   # only used if not VALIDATION
MAX_DEPTH   = 15  # only used if VALIDATION


VERBOSE     = False
PLOT        = False
EXPORT_TREE = True

PRINT_SIZE  = True

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
	X_train = np.matrix((training_pswes, training_toys)).T
	X_test = np.matrix((testing_pswes, testing_toys)).T
	X_validate = np.matrix((validation_pswes, validation_toys)).T
    elif (USE_PILLOW):
	X_train = np.matrix((training_pswes)).T
	X_test = np.matrix((testing_pswes)).T
	X_validate = np.matrix((validation_pswes)).T
    elif (USE_TOY):
	X_train = np.matrix((training_toys)).T
	X_test = np.matrix((testing_toys)).T
	X_validate = np.matrix((validation_toys)).T
    else:
	print "no independent variables!"
	sys.exit(1)


    y = np.matrix(training_aswes).T



    best_model = None
    best_error = 9999
    best_size = 0
    old_size = 0
    if (VALIDATION):
	limit = 1
	while (True):
	    model = tree.DecisionTreeRegressor(max_depth=limit)
	    limit = limit + 1
	    model.fit(X_train, y)
	    my_tree = model.tree_
	    size = my_tree.node_count
	    if (size == old_size):
		break
            old_size = size
	    #print dir(my_tree)
	    validation_predictions = model.predict(X_validate).T
	    error = np.mean(abs(validation_aswes - validation_predictions))
	    #print (size, error)
	    #print(tree_size, error, best_error)
	    if (error < best_error):
		best_size = size
		best_model = model
		best_error = error

    else:
	best_model = tree.DecisionTreeRegressor(max_depth=TREE_DEPTH)
	best_model.fit(X_train, y)



    test_predictions = best_model.predict(X_test).T
    test_error = abs(testing_aswes - test_predictions)


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

    #print (best_size, best_error, np.mean(test_error))
    print (best_size, np.mean(test_error))

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
	out = tree.export_graphviz(best_model, out_file='tree.dot')
