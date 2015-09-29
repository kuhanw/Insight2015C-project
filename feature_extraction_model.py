#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import *

from sklearn import cross_validation, datasets, linear_model
from sklearn.feature_extraction import text 
from sklearn.utils import resample
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import sys,getopt
import matplotlib.pyplot as plt
from time import time
import csv
import numpy as np

from read_json import *
from post_processing import *

import argparse


def make_binary(y):

	binary_y_pre = []

	for i in range(len(y)):
		if y[i]>0: binary_y_pre.append(1)
		else: binary_y_pre.append(0)

	return binary_y_pre

def get_args():

	'''This function parses and return arguments passed in'''
	# Assign description to the help doc
	parser = argparse.ArgumentParser(
	description='Execute Text Analysis Pipeline')
	# Add arguments
	parser.add_argument(
	'-w', '--widget_selection', type=str, help='Widget type [str]', required=True)
	parser.add_argument(
	'-nL', '--Ngram_Range_Low', type=int, help='Lower n-gram range [int]', required=True)
	parser.add_argument(
	'-nH', '--Ngram_Range_High', type=int, help='Upper n-gram range [int]', required=True)
	parser.add_argument(
	'-mDF', '--Min_DF', type=float, help='Minimum term document frequency between 0 to 1 [float]', required=True)
	parser.add_argument(
	'-pLoad', '--PageLoaded', type=int, help='Minimum page views [int]', required=True)
	parser.add_argument(
	'-wView', '--WidgetViewed', type=int, help='Minimum widget views [int]', required=True)
	parser.add_argument(
	'-Seed', '--ite', type=int, help='Random seed [int]', required=True)
	# Array for all arguments passed to script
	args = parser.parse_args()
	# Assign args to variables
	widget_selection = args.widget_selection
	Ngram_Range_Low = args.Ngram_Range_Low
	Ngram_Range_High = args.Ngram_Range_High
	Min_DF = args.Min_DF
	PageLoaded = args.PageLoaded
	WidgetViewed = args.WidgetViewed
	ite = args.ite
	# Return all variable values
	return widget_selection, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite

def main(arvg):

	widget_selection, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite = get_args()

	figures_folder = "figures/"+widget_selection + "/"

	#NLP knobs
	Decode_Error='ignore'

	#CV metrics
	CV=3

	#Modeling metrics
	Max_Iter=100
	Fit_Intercept=False
	Return_Models=False
	Positive=True
	Verbose=False
	N_Jobs=-1
	N_Alphas=1000
	Normalize=False
	Alphas=[0]
	Tol=0.001

	N_Estimators=10

	DSampling=False
	DSampling_Rate=0.50

	Scoring_1="mean_squared_error"
	Scoring_2 = 'f1'

	RSeed=ite

	input_json = "web_text_v12_data_set_1_2.json"

	print "%s, %s, %s, %s, %s, %s, %s" % (widget_selection, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite)

	corpus, engagement_rate, page_stats = read_json(input_json, widget_selection, PageLoaded, WidgetViewed)

	print "size of corpus:%d" % len(corpus)
	print "size of corpus target:%d" % len(engagement_rate)
	print len(engagement_rate)/4.

	#Test_Size=1./(CV-1)
	Test_Size=0.50
	#Test_Size=0

	print "Relative test data size:%.3g" % Test_Size
	#ADDITIONAL STOPWORDS
	my_words = ["0", "2015", "considering", "proper","agree", "soon", "changing", "wish", "flickr", "protect","including", 
			"example", "want", "concept", "photo", "like" ,"comes", "things", "com", "don", "help"] 

	my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

	#build vocabulary matrix

	vectorizer = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error=Decode_Error, 
					ngram_range=(Ngram_Range_Low,Ngram_Range_High),  min_df=Min_DF)#, max_df=0.85)

	X = vectorizer.fit_transform(corpus)

	corpus_array = X.toarray()

	print "number of zeros:%d" % engagement_rate.count(0)
	print "total number of engagements:%d" % len(engagement_rate)
	print "total number of non-zero:%d" % (len(engagement_rate) - engagement_rate.count(0))
	zero_rate = float(engagement_rate.count(0))/float(len(engagement_rate))

	print "zero rate:%.3g" % zero_rate

	#######DOWNSAMPLING BEGIN############
	if zero_rate>(1-DSampling_Rate): DSampling=True

	training_matrix = np.array(corpus_array)
	engagement_matrix = np.array(engagement_rate)

	print "training matrix:%d, engagement matrix:%d" % (len(training_matrix), len(engagement_matrix))
	#print training_matrix
	#print engagement_matrix
	total_matrix =  np.column_stack((training_matrix, engagement_matrix))
	matrix_of_zeros = []
	matrix_of_nonzeros = []

	for i in range(len(total_matrix)):
		if total_matrix[i][len(total_matrix[i])-1]>0:
			matrix_of_nonzeros.append(total_matrix[i])
		else: matrix_of_zeros.append(total_matrix[i])

	#print matrix_of_zeros

	#print matrix_of_nonzeros
	if DSampling==True:
		target_downsampling = DSampling_Rate;
		downsampling=int(np.round((len(matrix_of_nonzeros)/target_downsampling)*(1-target_downsampling)))
		#print downsampling
		downsampled_nonzeros=resample(matrix_of_zeros, n_samples=downsampling, random_state=0, replace = False)

		print len(downsampled_nonzeros)

		downsampled_total = np.concatenate((downsampled_nonzeros,matrix_of_nonzeros))
		downsampled_engagement = downsampled_total[:,(len(downsampled_total[0])-1):(len(downsampled_total[0]))]
		downsampled_training = downsampled_total[:,:-1]
		corpus_array = downsampled_training
	
		temp_y = []
		for i in range(len(downsampled_engagement)):
	#		print downsampled_engagement[i][0]
			temp_y.append(downsampled_engagement[i][0])	
		engagement_rate = temp_y
		print "resampled engagement length %d" % len(engagement_rate)

	####NORMALIZATION AND TDIDF WEIGHTING BEGIN#######
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(corpus_array)
	tfidf_array = tfidf.toarray()

	corpus_array = np.array(tfidf_array)

	########NORMALIZATION AND TDIDF WEIGHTING END#########

	#####SPLITTING TRAINING AND TEST DATASETS BEING##########
	x_train, x_test, y_train, y_test = train_test_split(corpus_array, engagement_rate, test_size=Test_Size, random_state=ite)

	print x_train.shape
	print y_train.shape

	print "Zeroes in y_train:%d" % list(y_train).count(0)
	print "Zeroes in y_test:%d" % list(y_test).count(0)
	print "total in y_train:%d" % len(list(y_train))
	print "total in y_test:%d" % len(list(y_test))

	X = x_train

	y = y_train

	print "size of training X:%d, training y:%d, test X:%d, test y:%d" % (x_train.shape[0], y_train.shape[0], x_test.shape[0], y_test.shape[0])
	#####SPLITTING TRAINING AND TEST DATASETS END##########

	number_of_features = len(vectorizer.get_feature_names())
	list_of_features = vectorizer.get_feature_names()
	print "number of features :%d" % number_of_features
	print "#######vocabulary########"

	binary_y = np.array(make_binary(y))

	coef_path_linear_cv = LinearRegression(normalize=Normalize,fit_intercept=Fit_Intercept) 
	coef_path_SGD_cv = SGDClassifier(loss='hinge', penalty='elasticnet') 
	coef_path_logistic_cv = LogisticRegression(penalty='l2', tol=Tol)
	coef_path_forest_cv = RandomForestClassifier(n_estimators = N_Estimators, random_state=ite, criterion='entropy', max_features=number_of_features)


	coef_path_forest_cv.fit(X,binary_y)
	coef_path_SGD_cv.fit(X,binary_y)
	coef_path_logistic_cv.fit(X,binary_y)

	forest_cv_score = cross_validation.cross_val_score(coef_path_forest_cv, X, binary_y, n_jobs=2, cv=CV, scoring=Scoring_2)
	SGD_cv_score = cross_validation.cross_val_score(coef_path_SGD_cv, X, binary_y, n_jobs=2, cv=CV, scoring=Scoring_2)
	logistic_cv_score = cross_validation.cross_val_score(coef_path_logistic_cv, X, binary_y, n_jobs=2, cv=CV, scoring=Scoring_2)

	forest_prediction_training = coef_path_forest_cv.predict(X)

	forest_results_parameters = [ forest_prediction_training, coef_path_forest_cv.get_params, coef_path_forest_cv.feature_importances_, 
					 coef_path_forest_cv.predict(x_test), np.array(make_binary(y_test)), coef_path_forest_cv.classes_] 
	forest_scores = [forest_cv_score, classification_report(binary_y, forest_results_parameters[0]), 'forest',
	  				 precision_score(np.array(make_binary(y)), forest_prediction_training),
			                recall_score(np.array(make_binary(y)), forest_prediction_training),
			                accuracy_score(np.array(make_binary(y)), forest_prediction_training),
			                confusion_matrix(np.array(make_binary(y)), forest_prediction_training)
					]
	SGD_prediction_training = coef_path_SGD_cv.predict(X)

	SGD_results_parameters = [ SGD_prediction_training, coef_path_SGD_cv.get_params, coef_path_SGD_cv.coef_, 
					 coef_path_SGD_cv.predict(x_test), np.array(make_binary(y_test))] 

	SGD_scores = [SGD_cv_score, classification_report(binary_y, SGD_results_parameters[0]), 'SGD',
					 precision_score(np.array(make_binary(y)), SGD_prediction_training),
			                recall_score(np.array(make_binary(y)), SGD_prediction_training),
			                accuracy_score(np.array(make_binary(y)), SGD_prediction_training),
			                confusion_matrix(np.array(make_binary(y)), SGD_prediction_training)
					]

	logistic_prediction_training = coef_path_logistic_cv.predict(X)

	logistic_results_parameters = [logistic_prediction_training, coef_path_logistic_cv.get_params, coef_path_logistic_cv.coef_, 
					coef_path_logistic_cv.predict(x_test), np.array(make_binary(y_test)), coef_path_logistic_cv.predict_proba(x_test)]

	logistic_scores = [logistic_cv_score, classification_report(binary_y, logistic_results_parameters[0]), 'logistic'
					, precision_score(np.array(make_binary(y)), logistic_prediction_training), 
					recall_score(np.array(make_binary(y)), logistic_prediction_training), 
					accuracy_score(np.array(make_binary(y)), logistic_prediction_training),
					confusion_matrix(np.array(make_binary(y)), logistic_prediction_training)
					]
	print "TTTTTTTTTTTTT"
	print logistic_cv_score
	print "TTTTTTTTTTTTT"
	linear_reg=0
	if linear_reg==1:
	##LINEAR REGRESSION METHOD BEGIN
		reduced_feature_matrix_logistic = []
		print "list of features from logistic regression:%d" % len(logistic_results_parameters[2][0])
		print len(X[0])
		transpose_training_X = zip(*X)
		reduced_feature_list_logistic = []

		for i in range(len(logistic_results_parameters[2][0])):
			if float(logistic_results_parameters[2][0][i])>0.01:
				reduced_feature_matrix_logistic.append(transpose_training_X[i])
				reduced_feature_list_logistic.append(list_of_features[i])

		print "reduced_feature_matrix_logistic before transpose:%d" % len(reduced_feature_matrix_logistic)	
		reduced_feature_matrix_logistic = zip(*reduced_feature_matrix_logistic)
		print "reduced_feature_matrix_logistic after transpose:%d" % len(reduced_feature_matrix_logistic)

		coef_path_linear_cv.fit(reduced_feature_matrix_logistic,y)

		linear_cv_score = cross_validation.cross_val_score(coef_path_linear_cv, reduced_feature_matrix_logistic, y, n_jobs=2, cv=CV, scoring=Scoring_1)

		linear_results_parameters = [ coef_path_linear_cv.predict(reduced_feature_matrix_logistic), coef_path_linear_cv.get_params,reduced_feature_list_logistic, coef_path_linear_cv.coef_]

		linear_scores = [linear_cv_score, r2_score(y, linear_results_parameters[0]), 'linear']

		print "reduced_feature_list_logistic length:%d" % len(reduced_feature_list_logistic)
		print "linear_coefficient length:%d" % len(coef_path_linear_cv.coef_)
		linear_word_results = []

		for i in range(len(coef_path_linear_cv.coef_)):
			temp_list = [reduced_feature_list_logistic[i], coef_path_linear_cv.coef_[i]]
			linear_word_results.append(temp_list)

	###LINEAR REGRESSION END

	model_results = [forest_results_parameters, SGD_results_parameters, logistic_results_parameters]

	model_scores = [forest_scores, SGD_scores, logistic_scores]

	post_processing(model_results, model_scores, X, y, widget_selection, list_of_features, 
			Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test)

if __name__ == "__main__":
	main(sys.argv[1:])

