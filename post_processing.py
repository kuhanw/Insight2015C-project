import sys
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
			Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test):

	PageLoaded = str(PageLoaded)
	WidgetViewed = str(WidgetViewed)	

	widget_selection = widget

	figures_folder = "figures/"+widget_selection + "/"

	X = training_set
	y = target

	#training_correlation = np.corrcoef(zip(*X))
	#testing_correlation = np.corrcoef(zip(*x_test))

	#print len(X)
	#print len(X[0])

#	print "training correlation size %d" % len(training_correlation)
#	corr_hist = ROOT.TH2D("corr_hist","corr_hist",len(training_correlation),0,len(training_correlation),len(training_correlation[0]),0,len(training_correlation[0]))
#	for i in range(len(training_correlation)):
#		for j in range(len(training_correlation[i])):
#			corr_hist.Fill(i,j, training_correlation[i][j])
			

#	c_corr_hist = ROOT.TCanvas("c_corr_hist","c_corr_hist",0,0,800,800)
#	corr_hist.GetXaxis().SetTitle("Features")
#	corr_hist.GetYaxis().SetTitle("Features")
#	corr_hist.SetTitle("Correlation Matrix")
#	corr_hist.Draw("COLZ")
#	c_corr_hist.SaveAs(figures_folder+ "logistic_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+"_"+PageLoaded+"_"+WidgetViewed+ "_correlation_training.svg")
	
	forest_results_parameters = model_results[0]
	lasso_results_parameters = model_results[1]
	elastic_results_parameters = model_results[2]
	logistic_results_parameters = model_results[2]
	binary_x_logistic_results_parameters = model_results[2]
	linear_results_parameters = model_results[2]

#	print "probabilities: %s" % logistic_results_parameters[5]

	logistic_prob_1 = ROOT.TH1D('logistic_prob_1','logistic_prob_1',100,0,1)
	logistic_prob_0 = ROOT.TH1D('logistic_prob_0','logistic_prob_0',100,0,1)
	for i in range(len(logistic_results_parameters[5])):
		logistic_prob_1.Fill(logistic_results_parameters[5][i][1])
		logistic_prob_0.Fill(logistic_results_parameters[5][i][0])

	c_logistic_prob_1 = ROOT.TCanvas('c_logistic_prob_1','c_logistic_prob_1',0,0,600,600)
	c_logistic_prob_1.SetLogy()
	logistic_prob_1.Draw()
	c_logistic_prob_1.SaveAs(figures_folder+ "logistic_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+"_"+PageLoaded+"_"+WidgetViewed+ "_logistic_prob_1.svg")

	c_logistic_prob_0 = ROOT.TCanvas('c_logistic_prob_0','c_logistic_prob_0',0,0,600,600)
	c_logistic_prob_0.SetLogy()
	logistic_prob_0.Draw()
	c_logistic_prob_0.SaveAs(figures_folder+ "logistic_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+"_"+PageLoaded+"_"+WidgetViewed+ "_logistic_prob_0.svg")

	forest_scores = model_scores[0]
	lasso_scores = model_scores[1]
	logistic_scores = model_scores[2]
	elastic_scores = model_scores[2]
	binary_x_logistic_scores = model_scores[2]
	linear_scores = model_scores[2]

	summary_scoring_metrics = [ forest_scores, lasso_scores, logistic_scores ]
	score_metrics = open(figures_folder +'validation_' + str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)
				 +"_"+PageLoaded+"_"+WidgetViewed+'_iteration'+str(ite)+'.txt', 'w')
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
		score_metrics.write("Test Precision :%.5g, Recall :%.5g, Accuracy :%.5g, Confusion:%d,%d,%d,%d,\n:" % (
				precision_score(model_results[i][4], model_results[i][3])
				,recall_score(model_results[i][4], model_results[i][3])
				,accuracy_score(model_results[i][4], model_results[i][3])
				,conf_matrix[0][0]
				,conf_matrix[0][1]
				,conf_matrix[1][0]
				,conf_matrix[1][1]
				)
				)
		print summary_scoring_metrics[i][0]
		score_metrics.write('@@@@@@@ \n')
	score_metrics.close()
	word_priority = []
	print len(list_of_features)
	
	for i in range(len(list_of_features)):
		word_priority_list = [list_of_features[i], i, forest_results_parameters[2][i], lasso_results_parameters[2][0][i], logistic_results_parameters[2][0][i] ] 
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
	
	with open(figures_folder+"ranked_words_ngram_" + str(Ngram_Range_Low) +"_" + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".csv", 'wb') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    spamwriter.writerow(ranked_words_header)
	    for i in range(len(ranked_key_words)):
	       	spamwriter.writerow(ranked_key_words[i])

	csvfile.close()	
