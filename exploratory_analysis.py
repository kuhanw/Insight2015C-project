from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
import logging
import sys
import matplotlib.pyplot as plt
from time import time
from operator import add
import csv
import numpy as np

import re
import json

import ROOT

from read_json import *

#build vocabulary matrix

Decode_Error = "ignore"

#widget_selection = "assetallocationcalculator"
widget_selection = "budgetcalculator"
#widget_selection = "careercalculator"

if widget_selection=="budgetcalculator": hist_title="Asset Type 1"
if widget_selection=="careercalculator": hist_title="Asset Type 2"
if widget_selection=="homeaffordability": hist_title="Asset Type 3"
if widget_selection=="assetallocationcalculator": hist_title="Asset Type 4"
figures_folder = "figures/"+widget_selection + "/"
corpus, engagement_rate, page_stats = read_json("web_text_v12_data_set_1_2.json",widget_selection, 0, 0)

vectorizer = CountVectorizer(analyzer="word", stop_words="english", decode_error=Decode_Error, ngram_range=(1,1))#, min_df=0.15)
vectorizer_bigram = CountVectorizer(analyzer="word", stop_words="english", decode_error=Decode_Error, ngram_range=(2,2))#, min_df=0.15)

X = vectorizer.fit_transform(corpus)
X_2 = vectorizer_bigram.fit_transform(corpus)

corpus_array = X.toarray()
corpus_array_bigram = X_2.toarray()

switch=1
setoptstat=1111
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetStatX(0.85)
ROOT.gStyle.SetStatY(0.90)
ROOT.gStyle.SetPalette(1)
#ROOT.gStyle.SetStyle('Plain')
print len(corpus_array)
hist_document_stats = ROOT.TH1D("hist_document_stats","hist_document_stats",5,-0.5,9.5)
hist_widgetviewed = ROOT.TH1D("hist_widgetviewed","hist_widgetvieweed",100,0,500)
hist_widgetclicked = ROOT.TH1D("hist_widgetclicked","hist_widgetclicked",100,0,500)
hist_engagementrate = ROOT.TH1D("hist_engagementrate","hist_engagement",100,0,1)
hist_pagesloaded = ROOT.TH1D("hist_pagesloaded","hist_pagesloaded",100,0,500)
hist_widgetviewed_pagesloaded = ROOT.TH2D("hist_widgetviewed_pagesloaded","hist_widegtviewed_pagesloaded",100,0,100,100,0,100)
hist_widgetclicked_pagesloaded  = ROOT.TH2D("hist_widgetclicked_pagesloaded ","hist_widgetclicked_pagesloaded ",100,0,100,100,0,100)
hist_engagementrate_int_df  = ROOT.TH2D("hist_engagementrate_int_df","hist_engagementrate_int_df ",100,0,1,100,0,1)
hist_df = ROOT.TH1D("hist_df","hist_df",100,0,1)
hist_int_df = ROOT.TH1D("hist_int_df","hist_int_df",100,0,1)
if switch==1:
	transpose_corpus_array = zip(*corpus_array)

	for i in range(len(transpose_corpus_array)):
		count=0
		for j in range(len(transpose_corpus_array[i])):
			if transpose_corpus_array[i][j]>0: count=count+1
#		print "%d, feature:%s, non-zero occurence:%d, percentage:%.3g" % (i,vectorizer.get_feature_names()[i],count,float(count)/float(len(transpose_corpus_array[i])))
		hist_df.Fill(float(count)/float(len(transpose_corpus_array[i])))
#		hist_engagementrate_int_df.Fill(i,engagement_rate[i],hist_df.Integral(i,100)/hist_df.Integral(0,100))		

for i in range(len(page_stats[0])):
	hist_engagementrate.Fill(engagement_rate[i])
	hist_pagesloaded.Fill(page_stats[0][i])
	hist_widgetviewed.Fill(page_stats[1][i])
	hist_widgetclicked.Fill(page_stats[2][i])
	hist_widgetviewed_pagesloaded.Fill(page_stats[1][i], page_stats[0][i])
	hist_widgetclicked_pagesloaded.Fill(page_stats[2][i], page_stats[0][i])
                  
c_hist_engagementrate = ROOT.TCanvas("c_hist_engagementrate","c_hist_engagementrate",0,0,3000,2000)
c_hist_engagementrate.SetLogy()
hist_engagementrate.GetXaxis().SetTitle("Frequency of Engagement")
hist_engagementrate.GetYaxis().SetTitle("Entries/0.01 bins")
hist_engagementrate.SetTitle("Asset Type 1")
hist_engagementrate.Draw("COLZ")
c_hist_engagementrate.SaveAs(figures_folder+ "engagementRate.pdf")


c_hist_widgetviewed_pagesloaded = ROOT.TCanvas("c_hist_widgetviewed_pagesloaded","c_hist_widgetviewed_pagesloaded",0,0,3000,2000)
hist_widgetviewed_pagesloaded.GetXaxis().SetTitle("WidgetsViewed")
hist_widgetviewed_pagesloaded.GetYaxis().SetTitle("PagesLoaded")
hist_widgetviewed_pagesloaded.SetTitle(hist_title)
hist_widgetviewed_pagesloaded.Draw("COLZ")
c_hist_widgetviewed_pagesloaded.SaveAs(figures_folder+ "widgetViewed_pagesLoaded.pdf")

c_hist_widgetclicked_pagesloaded = ROOT.TCanvas("c_hist_widgetclicked_pagesloaded","c_hist_widgetclicked_pagesloaded",0,0,3000,2000)
hist_widgetclicked_pagesloaded.GetXaxis().SetTitle("WidgetsClicked")
hist_widgetclicked_pagesloaded.GetYaxis().SetTitle("PagesLoaded")
hist_widgetclicked_pagesloaded.SetTitle(hist_title)
hist_widgetclicked_pagesloaded.Draw("COLZ")
c_hist_widgetclicked_pagesloaded.SaveAs(figures_folder+ "widgetClicked_pagesLoaded.pdf")

legend = ROOT.TLegend(0.6,0.65,0.88,0.85);
legend.SetTextFont(72);
legend.SetTextSize(0.04);
legend.AddEntry(hist_pagesloaded,"Page Loaded","l");
legend.AddEntry(hist_widgetviewed,"Asset Viewed","l");
legend.AddEntry(hist_widgetclicked,"Asset Clicked","l");
c_hist_pagesloaded = ROOT.TCanvas("c_hist_pagesloaded","c_hist_pagesloaded",0,0,3000,2000)
c_hist_pagesloaded.SetLogy()
hist_pagesloaded.GetYaxis().SetLabelOffset(999);
hist_pagesloaded.GetYaxis().SetLabelSize(0);
hist_pagesloaded.GetXaxis().SetLabelOffset(999);
hist_pagesloaded.GetXaxis().SetLabelSize(0);
hist_pagesloaded.GetXaxis().SetTitle("Occurrences")
hist_pagesloaded.GetYaxis().SetTitle("Entries/1 bin")
hist_pagesloaded.SetTitle(hist_title)
hist_pagesloaded.SetMinimum(0.1001)
hist_pagesloaded.Draw()
hist_widgetviewed.SetLineColor(2)
hist_widgetclicked.SetLineColor(3)
hist_widgetviewed.Draw("same")
hist_widgetclicked.Draw("same")
legend.Draw();
c_hist_pagesloaded.SaveAs(figures_folder+ "pagesLoaded.pdf")

c_hist_df = ROOT.TCanvas("c_hist_df","c_hist_df",0,0,3000,2000)
c_hist_df.SetLogy()
hist_df.GetXaxis().SetTitle("Document Feature Frequency")
hist_df.GetYaxis().SetTitle("Features")
hist_df.SetTitle(hist_title)
hist_df.Draw()

c_hist_df.SaveAs(figures_folder+ "totalDocumentFrequency.pdf")
for i in range(100):
	hist_int_df.SetBinContent(i,hist_df.Integral(i,100)/hist_df.Integral(0,100))		

c_int_hist_df = ROOT.TCanvas("c_int_hist_df","c_int_hist_df",0,0,1500,1000)
c_int_hist_df.SetLogy()
hist_int_df.SetMinimum(0.0000101)
hist_int_df.GetXaxis().SetTitle("Feature Frequency/Documents")
hist_int_df.GetYaxis().SetTitle("Relative Number of Documents [%]")
hist_int_df.SetTitle(hist_title)
hist_int_df.Draw()
c_int_hist_df.SaveAs(figures_folder+ "totalDocumentFrequencyInt2.pdf")

c_engagementrate_int_hist_df = ROOT.TCanvas("c_engagementrate_int_hist_df","c_engagementrate_int_hist_df",0,0,1500,1000)
hist_engagementrate_int_df.GetYaxis().SetTitle("Feature Frequency/Documents [%]")
hist_engagementrate_int_df.GetXaxis().SetTitle("Engagement Rate [%]")
hist_engagementrate_int_df.SetTitle(hist_title)
hist_engagementrate_int_df.Draw("COLZ")
c_engagementrate_int_hist_df.SaveAs(figures_folder+ "engagementRate_totalDocumentFrequencyInt2.pdf")
if switch==2:
	total_abs_word_frequency = []
	total_corpus_array= np.sum(corpus_array,axis=0)
	for i in range(len(vectorizer.get_feature_names())):
		temp_list = [ i, vectorizer.get_feature_names()[i], total_corpus_array[i]]
		total_abs_word_frequency.append(temp_list)
	sorted_word_frequency = sorted (total_abs_word_frequency, key= lambda x: int(x[2]))
	x = np.array(np.arange(0,len(sorted_word_frequency),1))
	print "standard_word........."
	sorted_word_frequency_transformed = zip(*sorted_word_frequency)[2]
	y = np.array(sorted_word_frequency_transformed)
	
	my_xticks = vectorizer.get_feature_names()
	plt.figure()
	plt.plot(x, y)
	plt.xlabel("Individual Words per Bin")
	plt.ylabel("Absolute Raw Word Count")
	plt.savefig("figures/wordCountAbs_xlabel.pdf")

	with open("FrequencyOfWords.csv", 'wb') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    for i in range(len(sorted_word_frequency)):
		print sorted_word_frequency[i]
	        spamwriter.writerow(sorted_word_frequency[i])
	csvfile.close()
	for i in xrange(0,len(corpus_array),1):
       	 plt.figure()
       	 plt.hist(corpus_array[i],np.arange(0,100,1),log="True")
      	 plt.ylabel("Entries/1 bin")
       	 plt.xlabel("Frequency of Words Absolute")
       	 plt.title(page_stats[3][i])
       	 plt_save_name = "figures/wordFrequencyAbsRow_%d.pdf" % i
       	 plt.savefig(plt_save_name)
         plt.close()


	hist_1 = ROOT.TH1D("hist_1","hist_1",450,0,450)
	hist_2 = ROOT.TH1D("hist_2","hist_2",450,0,450)

	for i in range(len(corpus_array)):
		if i%50==0: print "%d/%d" % (i,len(corpus_array))
		for j in range(len(corpus_array[i])):
			hist_1.Fill(corpus_array[i][j])

	c0 = ROOT.TCanvas("c0","c0",0,0,3000,2000)
	c0.SetLogy()
	hist_1.GetYaxis().SetTitle("Unique Unigrams")
	hist_1.GetXaxis().SetTitle("Absolute Frequency of Words")
	hist_1.Draw()
	c0.SaveAs(figures_folder+ "histTotalCorpusUnigram.pdf")

	for i in range(len(corpus_array_bigram)):
		if i%50==0: print "%d/%d" % (i,len(corpus_array_bigram))
		for j in range(len(corpus_array_bigram[i])):
			hist_2.Fill(corpus_array_bigram[i][j])

	c1 = ROOT.TCanvas("c1","c1",0,0,3000,2000)
	c1.SetLogy()
	hist_2.GetYaxis().SetTitle("Unique Bigrams")
	hist_2.GetXaxis().SetTitle("Absolute Frequency of Words")
	hist_2.Draw()
	c1.SaveAs(figures_folder+ "histTotalCorpusBigram.pdf")
	
	plt.figure()
	plt.hist(page_stats[0],np.arange(0,200,1),log="True")
	plt.ylabel("Entries/1 bin")
	plt.xlabel("# Pages Loaded")
	plt_save_name = "figures/pagesLoaded.pdf"
	plt.savefig(plt_save_name)
	plt.figure()
	plt.hist(page_stats[1],np.arange(0,500,1),log="True")
	plt.ylabel("Entries/1 bin")
	plt.xlabel("# Widget Visible")
	plt_save_name = "figures/widgetsVisible.pdf"
	plt.savefig(plt_save_name)
	plt.figure()
	plt.hist(page_stats[2],np.arange(0,100,1),log="True")
	plt.ylabel("Entries/1 bin")
	plt.xlabel("# Widget Used")
	plt_save_name = "figures/widgetsUsed.pdf"
	plt.savefig(plt_save_name)
	plt.figure()
	plt.hist(engagement_rate,np.arange(0,1,0.01),log="True")
	plt.ylabel("Entries/0.01 bin")
	plt.xlabel("Frequency of Engagement Rates")
	plt.title(widget_selection)
	plt_save_name = "figures/engageFrequencyAbs.pdf"
	plt.savefig(plt_save_name)
