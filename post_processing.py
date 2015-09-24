import ROOT
import csv

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)

def post_processing(model_results, model_scores, training_set, target, widget, list_of_features, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed):

	PageLoaded = str(PageLoaded)
	WidgetViewed = str(WidgetViewed)	

	widget_selection = widget

	figures_folder = "figures/"+widget_selection + "/"

	X = training_set
	y = target

	forest_results_parameters = model_results[0]
	lasso_results_parameters = model_results[1]
	elastic_results_parameters = model_results[2]
	logistic_results_parameters = model_results[3]
	binary_x_logistic_results_parameters = model_results[4]
	linear_results_parameters = model_results[5]

	forest_scores = model_scores[0]
	lasso_scores = model_scores[1]
	elastic_scores = model_scores[2]
	logistic_scores = model_scores[3]
	binary_x_logistic_scores = model_scores[4]
	linear_scores = model_scores[5]

	linear_word_results = []

	for i in range(len(linear_results_parameters[2])):
		#temp_list = [reduced_feature_list_logistic[i], coef_path_linear_cv.coef_[i]]
		temp_list = [linear_results_parameters[2][i], linear_results_parameters[3][i]]
		linear_word_results.append(temp_list)

	#print linear_word_results

	hist_0 = ROOT.TH2D("hist_0","hist_0",2,-0.5,1.5,2,-0.5,1.5)
	hist_3 = ROOT.TH2D("hist_3","hist_3",2,-0.5,1.5,2,-0.5,1.5)
	hist_4 = ROOT.TH2D("hist_4","hist_4",2,-0.5,1.5,2,-0.5,1.5)

	hist_1 = ROOT.TH2D("hist_1","hist_1",100,0,1,100,0,1)
	hist_2 = ROOT.TH2D("hist_2","hist_2",100,0,1,100,0,1)
	hist_5 = ROOT.TH2D("hist_5","hist_5",100,0,1,100,0,1)

	for i in range(len(y)):
		#hist_0.Fill(y[i],forest_results_parameters[0][i])
		hist_1.Fill(y[i]/len(y),lasso_results_parameters[0][i]/len(y))
		hist_2.Fill(y[i]/len(y),elastic_results_parameters[0][i]/len(y))
		hist_3.Fill(y[i]/len(y),logistic_results_parameters[0][i]/len(y))
		#hist_4.Fill(y[i],binary_x_logistic_results_parameters[0][i])
		hist_5.Fill(y[i]/len(y),linear_results_parameters[0][i]/len(y))


	c0 = ROOT.TCanvas("c0","c0",0,0,600,600)
	c0.SetLogy()
	hist_0.GetXaxis().SetTitle("Truth Target")
	hist_0.GetYaxis().SetTitle("Predicted Target")
	hist_0.Draw("COLZ")
	c1 = ROOT.TCanvas("c1","c1",0,0,600,600)
	c1.SetLogy()
	hist_1.GetXaxis().SetTitle("Truth Target")
	hist_1.GetYaxis().SetTitle("Predicted Target")
	hist_1.Draw("COLZ")
	c2 = ROOT.TCanvas("c2","c2",0,0,600,600)
	c2.SetLogy()
	hist_2.GetXaxis().SetTitle("Truth Target")
	hist_2.GetYaxis().SetTitle("Predicted Target")
	hist_2.Draw("COLZ")
	c3 = ROOT.TCanvas("c3","c3",0,0,600,600)
	c3.SetLogy()
	hist_3.GetXaxis().SetTitle("Truth Target")
	hist_3.GetYaxis().SetTitle("Predicted Target")
	hist_3.Draw("COLZ")
	c4 = ROOT.TCanvas("c4","c4",0,0,600,600)
	c4.SetLogy()
	hist_4.GetXaxis().SetTitle("Truth Target")
	hist_4.GetYaxis().SetTitle("Predicted Target")
	hist_4.Draw("COLZ")
	c5 = ROOT.TCanvas("c5","c5",0,0,600,600)
	c5.SetLogy()
	hist_5.GetXaxis().SetTitle("Truth Target")
	hist_5.GetYaxis().SetTitle("Predicted Target")
	hist_5.Draw("COLZ")
	c0.SaveAs(figures_folder+ "forest_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+"_"+PageLoaded+"_"+WidgetViewed+ ".pdf")
	c1.SaveAs(figures_folder+ "lasso_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".pdf")
	c2.SaveAs(figures_folder+ "elastic_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".pdf")
	c3.SaveAs(figures_folder+ "logistic_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".pdf")
	c4.SaveAs(figures_folder+ "binary_logistic_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".pdf")
	c5.SaveAs(figures_folder+ "linear_correlation"+str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".pdf")

#	summary_scoring_metrics = [forest_scores, lasso_scores, elastic_scores, logistic_scores, binary_x_logistic_scores, linear_scores  ]
	summary_scoring_metrics = [0, lasso_scores, elastic_scores, logistic_scores, 0, linear_scores  ]
	score_metrics = open(figures_folder +'validation_' + str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+ "_" + str(Min_DF) +"_"+PageLoaded+"_"+WidgetViewed+'.txt', 'w')
	for i in range(len(summary_scoring_metrics)):
		#try:
		if i==0 or i==4: continue
		else:
	#		print summary_scoring_metrics[i]
			score_metrics.write(str(summary_scoring_metrics[i][2]) + "," + str(summary_scoring_metrics[i][0]) + "," + str(summary_scoring_metrics[i][1])+ "\n" )
			print summary_scoring_metrics[i][0]
			plt.figure()
			plt.plot(np.arange(0,len(summary_scoring_metrics[i][0]),1), summary_scoring_metrics[i][0],'ro', markersize=10)
			plt.title(summary_scoring_metrics[i][2])
			plt.xlabel("K-Fold")
			plt.ylabel("Score")
			plt.savefig(figures_folder+'k_fold_cv_ngram_'+ str(Ngram_Range_Low) +'_' + str(Ngram_Range_High)+'_model_' + summary_scoring_metrics[i][2] + "_" + str(Min_DF)  +"_"+PageLoaded+"_"+WidgetViewed +'.pdf') 

	score_metrics.close()
	word_priority = []
	print len(list_of_features)
	print len(lasso_results_parameters[3])


	for i in range(len(list_of_features)):
#		word_priority_list = [list_of_features[i], lasso_results_parameters[3][i], elastic_results_parameters[3][i], 
		word_priority_list = [list_of_features[i], lasso_results_parameters[3][i], elastic_results_parameters[3][i], 
					logistic_results_parameters[2][0][i], 0, 0] 

		word_priority.append(word_priority_list)

	#print word_priority

	word_priority_lasso = sorted (word_priority, key= lambda x: float(x[1]), reverse=True)
	word_priority_elastic = sorted (word_priority, key= lambda x: float(x[2]), reverse=True)
	word_priority_logistic = sorted (word_priority, key= lambda x: float(x[3]), reverse=True)
	#word_priority_binary_logistic = sorted (word_priority, key= lambda x: float(x[4]), reverse=True)
#	word_priority_forest = sorted (word_priority, key= lambda x: float(x[5]), reverse=True)
	word_priority_linear = sorted (linear_word_results, key= lambda x: float(x[1]), reverse=True)
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
			0,0,0,0,
			0, 0]
		ranked_key_words.append(ranked_key_words_list)
	#	print ranked_key_words[i]

	ranked_words_header = [["lasso rank"],["lasso word"],["elastic rank"],["elastic word"],["logistic rank"],["logistic word"],["b-logistic rank"],["b-logistic word"],["forest rank"],["forest word"],["linear rank"],["linear word"]]
	
	with open(figures_folder+"ranked_words_ngram_" + str(Ngram_Range_Low) +"_" + str(Ngram_Range_High)+ "_" + str(Min_DF)+ "_"+PageLoaded+"_"+WidgetViewed+".csv", 'wb') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    spamwriter.writerow(ranked_words_header)
	    for i in range(len(ranked_key_words)):
	       	spamwriter.writerow(ranked_key_words[i])

	csvfile.close()	
