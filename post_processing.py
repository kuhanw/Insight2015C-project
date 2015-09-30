import ROOT
import csv

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)
ROOT.gROOT.SetBatch(1)

def post_processing(model_results, model_scores, training_set, target, widget, list_of_features, 
			Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test, Find):

	print "###POST PROCESSING###"

	PageLoaded = str(PageLoaded)
	WidgetViewed = str(WidgetViewed)	

	widget_selection = widget

	figures_folder = "figures/"+widget_selection + "/"

	X = training_set
	y = target

	forest_results_parameters = model_results[0]
	SGD_results_parameters = model_results[1]
	logistic_results_parameters = model_results[2]

	forest_scores = model_scores[0]
	SGD_scores = model_scores[1]
	logistic_scores = model_scores[2]

	summary_scoring_metrics = [ forest_scores, SGD_scores, logistic_scores ]
	score_metrics = open(figures_folder +'validation_' + str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF) \
				 +"_"+PageLoaded+"_"+WidgetViewed+'_iteration'+str(ite)+"_"+str(Find)+'.txt', 'w')
	for i in range(len(summary_scoring_metrics)):
		if i==0: print 'FOREST'
		if i==1: print 'SGD'
		if i==2: print 'LOGISTIC'
		score_metrics.write(str(summary_scoring_metrics[i][2]) + ',' + str(summary_scoring_metrics[i][0]) 
						+ ',' + str(summary_scoring_metrics[i][3])
						+ ',' + str(summary_scoring_metrics[i][4])
						+ ',' + str(summary_scoring_metrics[i][5])
						+'\n' )
		conf_matrix = confusion_matrix(model_results[i][4], model_results[i][3]) 
		try: score_metrics.write("Test Precision :%.5g, Recall :%.5g, Accuracy :%.5g, Confusion:%d,%d,%d,%d,\n:" % (
				precision_score(model_results[i][4], model_results[i][3])
				,recall_score(model_results[i][4], model_results[i][3])
				,accuracy_score(model_results[i][4], model_results[i][3])
				,conf_matrix[0][0]
				,conf_matrix[0][1]
				,conf_matrix[1][0]
				,conf_matrix[1][1]
				)
				)
		except: continue
		print summary_scoring_metrics[i][0]
		score_metrics.write('@@@@@@@ \n')
	score_metrics.close()
	word_priority = []
	print len(list_of_features)
	
	for i in range(len(list_of_features)):
		word_priority_list = [list_of_features[i], i, forest_results_parameters[2][i], SGD_results_parameters[2][0][i], logistic_results_parameters[2][0][i] ] 
		word_priority.append(word_priority_list)

	word_priority_forest = sorted (word_priority, key= lambda x: float(x[2]), reverse=True)
	word_priority_SGD = sorted(word_priority, key= lambda x: float(x[3]), reverse=True)
	word_priority_logistic = sorted (word_priority, key= lambda x: float(x[4]), reverse=True)

	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

	ranked_key_words = []

	for i in range(len(word_priority_logistic)):
		ranked_key_words_list = [
		word_priority_forest[i][1], word_priority_forest[i][2], word_priority_forest[i][0],
		word_priority_SGD[i][1], word_priority_SGD[i][3], word_priority_SGD[i][0],
		word_priority_logistic[i][1], word_priority_logistic[i][4], word_priority_logistic[i][0],
		]
		ranked_key_words.append(ranked_key_words_list)

	ranked_words_header = [["forest feature index"],["forest rank"],["forest word"],
				["SGD feature index"],["SGD rank"],["SGD word"],
				["logistic feature index"],["logistic rank"],["logistic word"]
				]
	
	with open(figures_folder+"ranked_words_ngram_" + str(Ngram_Range_Low) +"_" + str(Ngram_Range_High)+ \
			"_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+"_"+str(Find)+".csv", 'wb') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    spamwriter.writerow(ranked_words_header)
	    for i in range(len(ranked_key_words)):
	       	spamwriter.writerow(ranked_key_words[i])

	csvfile.close()	
