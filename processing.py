import re
import sys
import Quandl
import matplotlib.pyplot as plt
import datetime
import numpy as np
import random
import math
import time
from matplotlib.collections import LineCollection
from joblib import Parallel, delayed
import csv

data_set = "data_set_1"

engagement_path = "data/" + data_set + "/engagement.csv"
captivate_path = "data/" + data_set + "/captivate_keywords.csv"
scores_path = "data/" + data_set + "/scores.csv"

def processing(output_name):

	engagement_values = []

	with open(engagement_path) as engagement_list:
		next(engagement_list)
		next(engagement_list)
		next(engagement_list)
		for line in engagement_list:
			engagement_list = [str(line.split(",")[0]), int(line.split(",")[1]), int(line.split(",")[2]), int(line.split(",")[3])]
			engagement_values.append(engagement_list)		

	captivate_keywords = [] 
	widget_count = [] 

	with open(captivate_path) as captivate_keywords_list:
		next(captivate_keywords_list)
		for line in captivate_keywords_list:
			captive_keywords_list = [line.split(",")[0], line.split(",")[1], line.split(",")[2]]
			captivate_keywords.append(captive_keywords_list)		

	#scores.csv
	score_ranks = []
	a = []
	count = 0
	one_rank_scores = []
	two_rank_scores = []
	three_rank_scores = []
	widget_scores = [0]*31
	widget_placements = [0]*31

	with open(scores_path) as scores_list:
		for line in scores_list:
			score_ranks.append(line.split(",")[3])
			one_rank_scores_list = [-1,-1]
			two_rank_scores_list = [-1,-1]
			three_rank_scores_list = [-1,-1]

			try: 
	#			print line.split(",")[3].split(";")[0].split("[")[1]
				one_rank_scores_list= [ str(line.split(",")[3].split(";")[0].split("[")[0]), 
				float(line.split(",")[3].split(";")[0].split("[")[1].split("]")[0]),
				str(line.split(",")[2]) ]

				for i in range(len(captivate_keywords)): 
					if line.split(",")[3].split(";")[0].split("[")[0] in captivate_keywords[i][1]: 
						widget_scores[i]=widget_scores[i]+float(line.split(",")[3].split(";")[1].split("[")[1].split("]")[0]) 
						widget_placements[i]=widget_placements[i]+1 
								
				try: 
					two_rank_scores_list= [ line.split(",")[3].split(";")[1].split("[")[0],
								 float(line.split(",")[3].split(";")[1].split("[")[1].split("]")[0]),
								 str(line.split(",")[2]) ]
					try: 
						three_rank_scores_list= [ line.split(",")[3].split(";")[2].split("[")[0],
									   float(line.split(",")[3].split(";")[2].split("[")[1].split("]")[0]), 
								      	   str(line.split(",")[2]) ]
					except: 
						three_rank_scores_list= [ line.split(",")[3].split(";")[2].split("[")[0], -1.0, str(line.split(",")[2]) ]
				except: 
					two_rank_scores_list= [ line.split(",")[3].split(";")[1].split("[")[0], -1.0, str(line.split(",")[2]) ]
			except: 
	#			print "empty"
				one_rank_scores_list= [ line.split(",")[3].split(";")[0].split("[")[0], -1.0, str(line.split(",")[2]) ]
	#			continue

			one_rank_scores.append(one_rank_scores_list)
			two_rank_scores.append(two_rank_scores_list)
			three_rank_scores.append(three_rank_scores_list)
			count=count+1		

	#w_file = open('one_rank_correlation_v7.csv','wb')
	one_rank_correlation_pageload_score = []
	one_rank_correlation_widget_score = []
	one_rank_correlation_engagement_score = []

	one_ranked_correlation_matched_indices = []
	print "number of urls in engagement.csv:%d" % len(engagement_values)

	for i in range(len(engagement_values)):
		#print "url:%d, %s" % (i, engagement_values[i][0])
		for j in range(len(one_rank_scores)):
			if engagement_values[i][0]==one_rank_scores[j][2]:
				#print "MATCH:%s, %s" % (engagement_values[i][0], one_rank_scores[j][2])
				temp_list = [engagement_values[i][0],engagement_values[i][1],engagement_values[i][2], engagement_values[i][3],
				one_rank_scores[j][1], one_rank_scores[j][0]]
		                #print temp_list
				#if (temp_list[4]>0): 
				one_ranked_correlation_matched_indices.append(temp_list)

	print one_ranked_correlation_matched_indices
	output_csv_name = output_name+'_' + data_set + '.csv', 'wb'
	with open(output_csv_name) as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',',
		                    quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    for i in range(len(one_ranked_correlation_matched_indices)):
		print one_ranked_correlation_matched_indices[i]
		spamwriter.writerow(one_ranked_correlation_matched_indices[i])

	csvfile.close()

	return output_csv_name
