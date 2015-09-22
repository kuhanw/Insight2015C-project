from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

from sklearn import cross_validation, datasets, linear_model
from sklearn.feature_extraction import text 

import sys
import matplotlib.pyplot as plt
from time import time
import csv
import numpy as np
import re
import json

from read_json import *

import ROOT

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)

#NLP knobs
Decode_Error='ignore'

#Modeling metrics
Max_Iter=10000
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
#CV metrics
Scoring="mean_squared_error"
CV=3
Ngram_Range_Low=1
Ngram_Range_High=2

#Widget selection
widget_selection = 'budgetcalculator'
#widget_selection = 'assetallocationcalculator'
#widget_selection = 'careercalculator'
figures_folder = "figures/"+widget_selection + "/"
corpus, engagement_rate, page_stats = read_json("web_text_v9c.json",widget_selection)

my_words = ["considering","proper","agree", "soon", "changing", "wish", "flickr", "protect","including", "example", "want", "concept", "photo", "like" ,"comes", "things", "com", "don", "help"]#, "improve wisegeek", "related article", "u'improve wisegeek"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

#build vocabulary matrix
print "size of corpus:%d" % len(corpus)

vectorizer = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error=Decode_Error, 
				ngram_range=(Ngram_Range_Low,Ngram_Range_High),  min_df=0.05)#, max_df=0.85)
vectorizer_binary = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error=Decode_Error, 
			ngram_range=(Ngram_Range_Low,Ngram_Range_High), binary="True",  min_df=0.05)#, max_df=0.85)

X = vectorizer.fit_transform(corpus)

corpus_array = X.toarray()
number_of_features = len(vectorizer.get_feature_names())
print "list of features:%d" % number_of_features
print "#######vectorizer stop words############"
print vectorizer.get_stop_words()
print "#######vocabulary########"
print vectorizer.vocabulary_
print corpus_array

#with open(figures_folder+'diagonostic_stop_'+widget_selection+'words.csv', 'wb') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(vectorizer.get_stop_words())
#csvfile.close()
#with open(igures_folder+'diagonostic_vocabulary'+widget_selection+'.csv', 'wb') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(vectorizer.vocabulary_)
#csvfile.close()

##reweight usingTf-idf term weighting

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(corpus_array)

tfidf_array = tfidf.toarray()

print tfidf_array

print engagement_rate
print tfidf_array.shape

print len(engagement_rate)

X = np.array(tfidf_array)
y = np.array(engagement_rate)
print X

coef_path_linear_cv = LinearRegression(normalize=Normalize,fit_intercept=Fit_Intercept) 
coef_path_lasso_cv = LassoCV(normalize=Normalize, max_iter=Max_Iter, copy_X=True, cv=CV, verbose=Verbose, fit_intercept=Fit_Intercept, tol=Tol)#, alphas=Alphas) 
coef_path_elastic_cv = ElasticNetCV(normalize=Normalize,max_iter=Max_Iter, tol=Tol)#,alphas=Alphas)
coef_path_logistic_cv = LogisticRegression( tol=Tol)
coef_path_binary_x_logistic_cv = LogisticRegression( tol=Tol)
coef_path_forest_cv = RandomForestClassifier(n_estimators = N_Estimators, max_features=number_of_features)
 
binary_y_pre = []

for i in range(len(y)):
	if y[i]>0: binary_y_pre.append(1)
	else: binary_y_pre.append(0)
binary_y = np.array(binary_y_pre)
print "binary y"
print binary_y

binary_X = vectorizer_binary.fit_transform(corpus)
coef_path_forest_cv.fit(X,binary_y)
coef_path_lasso_cv.fit(X,y)
coef_path_binary_x_logistic_cv.fit(binary_X,binary_y)
coef_path_logistic_cv.fit(X,binary_y)
coef_path_elastic_cv.fit(X,y)

forest_cv_score = cross_validation.cross_val_score(coef_path_forest_cv, X, binary_y, n_jobs=2, cv=CV, scoring='roc_auc')
lasso_cv_score = cross_validation.cross_val_score(coef_path_lasso_cv, X, y, n_jobs=2, cv=CV, scoring=Scoring)
elastic_cv_score = cross_validation.cross_val_score(coef_path_elastic_cv, X, y, n_jobs=2, cv=CV, scoring=Scoring)
logistic_cv_score = cross_validation.cross_val_score(coef_path_logistic_cv, X, binary_y, n_jobs=2, cv=CV, scoring='roc_auc')
binary_x_logistic_cv_score = cross_validation.cross_val_score(coef_path_binary_x_logistic_cv, binary_X, binary_y, n_jobs=2, cv=CV, scoring='roc_auc')

forest_results_parameters = [ coef_path_forest_cv.predict(X), coef_path_forest_cv.get_params, coef_path_forest_cv.feature_importances_, 
				coef_path_forest_cv.classes_, coef_path_forest_cv.n_classes_]
forest_scores = [forest_cv_score, classification_report(binary_y, forest_results_parameters[0])]

lasso_results_parameters = [coef_path_lasso_cv.predict(X), coef_path_lasso_cv.get_params, coef_path_lasso_cv.alphas_, coef_path_lasso_cv.coef_]  

lasso_scores = [lasso_cv_score, r2_score(y,lasso_results_parameters[0])]

elastic_results_parameters = [ coef_path_elastic_cv.predict(X), coef_path_elastic_cv.get_params, coef_path_elastic_cv.alphas_ ,
				coef_path_elastic_cv.coef_]
elastic_scores = [elastic_cv_score, r2_score(y,elastic_results_parameters[0])]

logistic_results_parameters = [coef_path_logistic_cv.predict(X), coef_path_logistic_cv.get_params, coef_path_logistic_cv.coef_]

logistic_scores = [logistic_cv_score, classification_report(binary_y, logistic_results_parameters[0])]

binary_x_logistic_results_parameters = [coef_path_binary_x_logistic_cv.predict(X), coef_path_binary_x_logistic_cv.get_params, coef_path_binary_x_logistic_cv.coef_]

binary_x_logistic_scores = [binary_x_logistic_cv_score, classification_report(binary_y, binary_x_logistic_results_parameters[0])]

##LINEAR REGRESSION METHOD BEGIN
reduced_feature_matrix_logistic = []
print "list of features from logistic regression:%d" % len(logistic_results_parameters[2][0])
print len(X[0])
transpose_training_X = zip(*X)
print "transpose_training_x: %d" % len(transpose_training_X)
print "feature list length:%d" % len(vectorizer.get_feature_names())
reduced_feature_list_logistic = []

for i in range(len(logistic_results_parameters[2][0])):
	if float(logistic_results_parameters[2][0][i])>0.01:
#		print "logistic beta:%.3g, word:%s" % (logistic_results_parameters[2][0][i], vectorizer.get_feature_names()[i])
		#temp_list = [vectorizer.get_feature_names()[i],transpose_training_X[i]]
		reduced_feature_matrix_logistic.append(transpose_training_X[i])
		reduced_feature_list_logistic.append(vectorizer.get_feature_names()[i])

print "reduced_feature_matrix_logistic before transpose:%d" % len(reduced_feature_matrix_logistic)	
reduced_feature_matrix_logistic = zip(*reduced_feature_matrix_logistic)
print "reduced_feature_matrix_logistic after transpose:%d" % len(reduced_feature_matrix_logistic)

coef_path_linear_cv.fit(reduced_feature_matrix_logistic,y)

linear_cv_score = cross_validation.cross_val_score(coef_path_linear_cv, reduced_feature_matrix_logistic, y, n_jobs=2, cv=CV, scoring=Scoring)

linear_results_parameters = [ coef_path_linear_cv.predict(reduced_feature_matrix_logistic), coef_path_linear_cv.get_params, coef_path_linear_cv.coef_]

linear_scores = [linear_cv_score, r2_score(y, linear_results_parameters[0])]

print "reduced_feature_list_logistic length:%d" % len(reduced_feature_list_logistic)
print "linear_coefficient length:%d" % len(coef_path_linear_cv.coef_)
linear_word_results = []
###LINEAR REGRESSION END
for i in range(len(coef_path_linear_cv.coef_)):
	temp_list = [reduced_feature_list_logistic[i], coef_path_linear_cv.coef_[i]]
	linear_word_results.append(temp_list)
word_priority_linear = sorted (linear_word_results, key= lambda x: float(x[1]), reverse=True)

hist_0 = ROOT.TH2D("hist_0","hist_0",2,-0.5,1.5,2,-0.5,1.5)
hist_3 = ROOT.TH2D("hist_3","hist_3",2,-0.5,1.5,2,-0.5,1.5)
hist_4 = ROOT.TH2D("hist_4","hist_4",2,-0.5,1.5,2,-0.5,1.5)

hist_1 = ROOT.TH2D("hist_1","hist_1",100,0,1,100,0,1)
hist_2 = ROOT.TH2D("hist_2","hist_2",100,0,1,100,0,1)
hist_5 = ROOT.TH2D("hist_5","hist_5",100,0,1,100,0,1)

for i in range(len(y)):
	hist_0.Fill(y,forest_results_parameters[0][i])
	hist_1.Fill(y,lasso_results_parameters[0][i])
	hist_2.Fill(y,elastic_results_parameters[0][i])
	hist_3.Fill(y,logistic_results_parameters[0][i])
	hist_4.Fill(y,binary_logistic_results_parameters[0][i])
	hist_5.Fill(y,linear_results_parameters[0][i])


c0 = ROOT.TCanvas("c0","c0",0,0,600,600)
hist_0.GetXaxis().SetTitle("Truth Target")
hist_0.GetYaxis().SetTitle("Predicted Target")
hist_0.Draw("COLZ")
c1 = ROOT.TCanvas("c1","c1",0,0,600,600)
hist_1.GetXaxis().SetTitle("Truth Target")
hist_1.GetYaxis().SetTitle("Predicted Target")
hist_1.Draw("COLZ")
c2 = ROOT.TCanvas("c2","c2",0,0,600,600)
hist_2.GetXaxis().SetTitle("Truth Target")
hist_2.GetYaxis().SetTitle("Predicted Target")
hist_2.Draw("COLZ")
c3 = ROOT.TCanvas("c3","c3",0,0,600,600)
hist_3.GetXaxis().SetTitle("Truth Target")
hist_3.GetYaxis().SetTitle("Predicted Target")
hist_3.Draw("COLZ")
c4 = ROOT.TCanvas("c4","c4",0,0,600,600)
hist_4.GetXaxis().SetTitle("Truth Target")
hist_4.GetYaxis().SetTitle("Predicted Target")
hist_4.Draw("COLZ")
c5 = ROOT.TCanvas("c5","c5",0,0,600,600)
hist_5.GetXaxis().SetTitle("Truth Target")
hist_5.GetYaxis().SetTitle("Predicted Target")
hist_5.Draw("COLZ")
c0.SaveAs(figures_folder+ "forest_correlation.pdf")
c1.SaveAs(figures_folder+ "lasso_correlation.pdf")
c2.SaveAs(figures_folder+ "elastic_correlation.pdf")
c3.SaveAs(figures_folder+ "logistic_correlation.pdf")
c4.SaveAs(figures_folder+ "binary_logistic_correlation.pdf")
c5.SaveAs(figures_folder+ "linear_correlation.pdf")
print "%d-FOLD VALIDATION" % CV

for i in range(len(summary_scoring_metrics)):
		print "model:%d " % (i)
		print summary_scoring_metrics[i]

word_priority = []
for i in range(len(vectorizer.get_feature_names())):
	word_priority_list = [vectorizer.get_feature_names()[i], coef_path_lasso_cv.coef_[i], coef_path_elastic_cv.coef_[i], 
				coef_path_logistic_cv.coef_[0][i], coef_path_binary_x_logistic_cv.coef_[0][i], coef_path_forest_cv.feature_importances_[i],0] 
			#	coef_path_linear_cv.coef_[i]]
#	word_priority_list = [vectorizer.get_feature_names()[i], coef_path_lasso_cv.coef_[i], coef_path_elastic_cv.coef_[i], 
#				coef_path_logistic_cv.coef_[0][i], coef_path_binary_x_logistic_cv.coef_[0][i], coef_path_forest_cv.feature_importances_[i], 
#				0]
	word_priority.append(word_priority_list)
#	print "b_lasso:%.2g, b_elastic_net:%.2g, b_logistic:%.2g, word:%s" % (coef_path_lasso_cv.coef_[i], coef_path_elastic_cv.coef_[i], coef_path_logistic_cv.coef_[0][i], vectorizer.get_feature_names()[i])

word_priority_lasso = sorted (word_priority, key= lambda x: float(x[1]), reverse=True)
word_priority_elastic = sorted (word_priority, key= lambda x: float(x[2]), reverse=True)
word_priority_logistic = sorted (word_priority, key= lambda x: float(x[3]), reverse=True)
word_priority_binary_logistic = sorted (word_priority, key= lambda x: float(x[4]), reverse=True)
word_priority_forest = sorted (word_priority, key= lambda x: float(x[5]), reverse=True)
#word_priority_linear = sorted (word_priority, key= lambda x: float(x[6]), reverse=True)

print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

ranked_key_words = []

for i in range(len(word_priority_lasso)):
	try:
		ranked_key_words_list = [word_priority_lasso[i][1], word_priority_lasso[i][0],
		word_priority_elastic[i][2], word_priority_elastic[i][0],
		word_priority_logistic[i][3], word_priority_logistic[i][0],
		word_priority_binary_logistic[i][4], word_priority_binary_logistic[i][0],
		word_priority_forest[i][5], word_priority_forest[i][0],
		word_priority_linear[i][0], word_priority_linear[i][1]]
	except:
		ranked_key_words_list = [word_priority_lasso[i][1], word_priority_lasso[i][0],
		word_priority_elastic[i][2], word_priority_elastic[i][0],
		word_priority_logistic[i][3], word_priority_logistic[i][0],
		word_priority_binary_logistic[i][4], word_priority_binary_logistic[i][0],
		word_priority_forest[i][5], word_priority_forest[i][0],
		0, 0]
	ranked_key_words.append(ranked_key_words_list)
#	print ranked_key_words[i]

ranked_words_header = [["lasso rank"],["lasso word"],["elastic rank"],["elastic word"],["logistic rank"],["logistic word"],["b-logistic rank"],["b-logistic word"],["forest rank"],["forest word"],["linear rank"],["linear word"]]
#ranked_words_header = ["lasso rank","lasso word","elastic rank","elastic word","logistic rank","logistic word","b-logistic rank","b-logistic word","forest rank","forest word"]
with open(figures_folder+"ranked_words_"+widget_selection+"v01.csv", 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(ranked_words_header)
    for i in range(len(ranked_key_words)):
       	spamwriter.writerow(ranked_key_words[i])

csvfile.close()	
