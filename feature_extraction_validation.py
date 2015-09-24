from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

from sklearn import cross_validation, datasets, linear_model
from sklearn.feature_extraction import text 
from sklearn.utils import resample

import sys
import matplotlib.pyplot as plt
from time import time
import csv
import numpy as np
import re

from read_json import *
from post_processing import *
import ROOT

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)

#Widget selection
widget_selection = sys.argv[1]
figures_folder = "figures/"+widget_selection + "/"

#NLP knobs
Decode_Error='ignore'

#CV metrics
Scoring="mean_squared_error"
CV=3
Ngram_Range_Low=int(sys.argv[2])
Ngram_Range_High=int(sys.argv[3])

#Modeling metrics
Max_Iter=1000
Fit_Intercept=False
Return_Models=False
Positive=True
Verbose=False
N_Jobs=-1
N_Alphas=1000
Normalize=False
Alphas=[0]
Tol=0.001
Min_DF=float(sys.argv[4])

N_Estimators=1000
DSampling=False
DSampling_Rate=0.50

PageLoaded = int(sys.argv[5])
WidgetViewed = int(sys.argv[6])

print "%s, %s, %s, %s, %s, %s" % (widget_selection, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed)
corpus, engagement_rate, page_stats = read_json("web_text_v9c.json",widget_selection, PageLoaded, WidgetViewed)

##DOWNSAMPLING TEST


#for i in range(len(engagement_rate)):
#	print engagement_rate[i]
my_words = ["considering","proper","agree", "soon", "changing", "wish", "flickr", "protect","including", "example", "want", "concept", "photo", "like" ,"comes", "things", "com", "don", "help"]#, "improve wisegeek", "related article", "u'improve wisegeek"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

#build vocabulary matrix
print "size of corpus:%d" % len(corpus)

vectorizer = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error=Decode_Error, 
				ngram_range=(Ngram_Range_Low,Ngram_Range_High),  min_df=Min_DF)#, max_df=0.85)
vectorizer_binary = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error=Decode_Error, 
			ngram_range=(Ngram_Range_Low,Ngram_Range_High), binary="True",  min_df=Min_DF)#, max_df=0.85)

X = vectorizer.fit_transform(corpus)

corpus_array = X.toarray()
#######DOWNSAMPLING BEGIN############
print "number of zeros:%d" % engagement_rate.count(0)
print "total number of engagements:%d" % len(engagement_rate)
zero_rate = float(engagement_rate.count(0))/float(len(engagement_rate))
print "zero rate:%.3g" % zero_rate
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
	#print downsampled_nonzeros

	downsampled_total = np.concatenate((downsampled_nonzeros,matrix_of_nonzeros))
#print matrix_of_nonzeros
	downsampled_engagement = downsampled_total[:,(len(downsampled_total[0])-1):(len(downsampled_total[0]))]
#print "downsampled engagement %s: " % downsampled_engagement
	downsampled_training = downsampled_total[:,:-1]
	#print downsampled_training
	#print downsampled_training.shape
	corpus_array = downsampled_training
#######DOWNSAMPLING END############
number_of_features = len(vectorizer.get_feature_names())
list_of_features = vectorizer.get_feature_names()
print "number of features :%d" % number_of_features
print "#######vocabulary########"
#print vectorizer.vocabulary_


transformer = TfidfTransformer()
tfidf = transformer.fit_transform(corpus_array)

tfidf_array = tfidf.toarray()

#print tfidf_array

#print engagement_rate
#print tfidf_array.shape

#print len(engagement_rate)



X = np.array(tfidf_array)
y = np.array(engagement_rate)

if DSampling==True: 
	temp_y = []
	for i in range(len(downsampled_engagement)):
#		print downsampled_engagement[i][0]
		temp_y.append(downsampled_engagement[i][0])	
#y = downsampled_engagement

	y = np.array(temp_y)
print "data points of training set:%d" % len(X)
print "data points of target set:%d" % len(y)
#print y
binary_y_pre = []

for i in range(len(y)):
	if y[i]>0: binary_y_pre.append(1)
	else: binary_y_pre.append(0)
binary_y = np.array(binary_y_pre)

coef_path_linear_cv = LinearRegression(normalize=Normalize,fit_intercept=Fit_Intercept) 
coef_path_lasso_cv = LassoCV(normalize=Normalize, max_iter=Max_Iter, copy_X=True, cv=CV, verbose=Verbose, fit_intercept=Fit_Intercept, tol=Tol)#, alphas=Alphas) 
coef_path_elastic_cv = ElasticNetCV(normalize=Normalize,max_iter=Max_Iter, tol=Tol)#,alphas=Alphas)
coef_path_logistic_cv = LogisticRegression( tol=Tol)
#coef_path_binary_x_logistic_cv = LogisticRegression( tol=Tol)
coef_path_forest_cv = RandomForestClassifier(n_estimators = N_Estimators, max_features='auto')

#if DSampling==True: binary_X = vectorizer_binary.fit_transform(downsampled_training)

#else: binary_X = vectorizer_binary.fit_transform(corpus)
#coef_path_forest_cv.fit(X,binary_y)
coef_path_lasso_cv.fit(X,y)
#coef_path_binary_x_logistic_cv.fit(binary_X,binary_y)
coef_path_logistic_cv.fit(X,binary_y)
coef_path_elastic_cv.fit(X,y)

#forest_cv_score = cross_validation.cross_val_score(coef_path_forest_cv, X, binary_y, n_jobs=2, cv=CV, scoring='f1')
print "before lassso X:%d" % len(X)
#print X
print "before lassso y:%d" % len(y)
lasso_cv_score = cross_validation.cross_val_score(coef_path_lasso_cv, X, y, n_jobs=2, cv=CV, scoring=Scoring)
elastic_cv_score = cross_validation.cross_val_score(coef_path_elastic_cv, X, y, n_jobs=2, cv=CV, scoring=Scoring)
logistic_cv_score = cross_validation.cross_val_score(coef_path_logistic_cv, X, binary_y, n_jobs=2, cv=CV, scoring='f1')
#binary_x_logistic_cv_score = cross_validation.cross_val_score(coef_path_binary_x_logistic_cv, binary_X, binary_y, n_jobs=2, cv=CV, scoring='f1')

#forest_results_parameters = [ coef_path_forest_cv.predict(X), coef_path_forest_cv.get_params, coef_path_forest_cv.feature_importances_, 
#				coef_path_forest_cv.classes_, coef_path_forest_cv.n_classes_]
#forest_scores = [forest_cv_score, classification_report(binary_y, forest_results_parameters[0]), 'forest']

lasso_results_parameters = [coef_path_lasso_cv.predict(X), coef_path_lasso_cv.get_params, coef_path_lasso_cv.alphas_, coef_path_lasso_cv.coef_]  

lasso_scores = [lasso_cv_score, r2_score(y,lasso_results_parameters[0]), 'lasso']

elastic_results_parameters = [ coef_path_elastic_cv.predict(X), coef_path_elastic_cv.get_params, coef_path_elastic_cv.alphas_ ,
				coef_path_elastic_cv.coef_]
elastic_scores = [elastic_cv_score, r2_score(y,elastic_results_parameters[0]), 'elastic']

logistic_results_parameters = [coef_path_logistic_cv.predict(X), coef_path_logistic_cv.get_params, coef_path_logistic_cv.coef_]

logistic_scores = [logistic_cv_score, classification_report(binary_y, logistic_results_parameters[0]), 'logistic']

#binary_x_logistic_results_parameters = [coef_path_binary_x_logistic_cv.predict(X), coef_path_binary_x_logistic_cv.get_params, coef_path_binary_x_logistic_cv.coef_]

#binary_x_logistic_scores = [binary_x_logistic_cv_score, classification_report(binary_y, binary_x_logistic_results_parameters[0]), 'binary_logistic']

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

linear_cv_score = cross_validation.cross_val_score(coef_path_linear_cv, reduced_feature_matrix_logistic, y, n_jobs=2, cv=CV, scoring=Scoring)

linear_results_parameters = [ coef_path_linear_cv.predict(reduced_feature_matrix_logistic), coef_path_linear_cv.get_params,reduced_feature_list_logistic, coef_path_linear_cv.coef_]

linear_scores = [linear_cv_score, r2_score(y, linear_results_parameters[0]), 'linear']

print "reduced_feature_list_logistic length:%d" % len(reduced_feature_list_logistic)
print "linear_coefficient length:%d" % len(coef_path_linear_cv.coef_)
linear_word_results = []
###LINEAR REGRESSION END

for i in range(len(coef_path_linear_cv.coef_)):
	temp_list = [reduced_feature_list_logistic[i], coef_path_linear_cv.coef_[i]]
	linear_word_results.append(temp_list)

#word_priority_linear = sorted (linear_word_results, key= lambda x: float(x[1]), reverse=True)

#model_results = [forest_results_parameters, lasso_results_parameters, elastic_results_parameters, logistic_results_parameters, binary_x_logistic_results_parameters, linear_results_parameters]

#model_scores = [forest_scores, lasso_scores, elastic_scores, logistic_scores, binary_x_logistic_scores, linear_scores]

model_results = [0, lasso_results_parameters, elastic_results_parameters, logistic_results_parameters, 0, linear_results_parameters]

model_scores = [0, lasso_scores, elastic_scores, logistic_scores, 0, linear_scores]
post_processing(model_results, model_scores, X, y, widget_selection, list_of_features, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed)

