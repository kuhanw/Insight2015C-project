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

data_set = "data_set_2"

engagement_path = "data/" + data_set + "/engagement.csv"
captivate_path = "data/" + data_set + "/captivate_keywords.csv"
scores_path = "data/" + data_set + "/scores.csv"

#engagement.csv

engagement_values = []

with open(engagement_path) as engagement_list:
	next(engagement_list)
	next(engagement_list)
	next(engagement_list)
        for line in engagement_list:
		engagement_list = [str(line.split(",")[0]), int(line.split(",")[1]), int(line.split(",")[2]), int(line.split(",")[3])]
		engagement_values.append(engagement_list)		

#print engagement_values

#captivate_keywords.csv

captivate_keywords = [] 
widget_count = [] 

with open(captivate_path) as captivate_keywords_list:
	next(captivate_keywords_list)
        for line in captivate_keywords_list:
		captive_keywords_list = [line.split(",")[0], line.split(",")[1], line.split(",")[2]]
		captivate_keywords.append(captive_keywords_list)		

#for i in range(len(captivate_keywords)):
#	print captivate_keywords[i] 
print captivate_keywords

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
#		if line.split(",")[3].split(";")[0]<0.0001: continue
#		print line
		# current_buffer=line.split(",")[0]
		score_ranks.append(line.split(",")[3])
#		print "url:%s, line:%d, value:%s" % (line.split(",")[2], count, line.split(",")[3].split(";")[0])
#		print re.findall("\s+", line.split(",")[3].split(";")[0].split("]")[0])
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
					#print "%s %s" % (line.split(",")[3].split(";")[0].split("[")[0], captivate_keywords[i][1]) 
								
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

#		print line.split(",")[3]

		one_rank_scores.append(one_rank_scores_list)
		two_rank_scores.append(two_rank_scores_list)
		three_rank_scores.append(three_rank_scores_list)
		count=count+1		

#w_file = open('one_rank_correlation_v7.csv','wb')
one_rank_correlation_pageload_score = []
one_rank_correlation_widget_score = []
one_rank_correlation_engagement_score = []

one_ranked_correlation_matched_indices = []
print one_rank_scores[4]
for i in range(len(engagement_values)):
	for j in range(len(one_rank_scores)):
		if engagement_values[i][0]==one_rank_scores[j][2]:
#			text="%s,%d,%d,%d,%.2g,%s " % (engagement_values[i][0],engagement_values[i][1],engagement_values[i][2], engagement_values[i][3],
#			 one_rank_scores[j][1], one_rank_scores[j][0])
#			print text
			temp_list = [engagement_values[i][0],engagement_values[i][1],engagement_values[i][2], engagement_values[i][3],
                         one_rank_scores[j][1], one_rank_scores[j][0]]
			if (temp_list[4]>0): one_ranked_correlation_matched_indices.append(temp_list)
			#	w_file.write(text)	
#				spamwriter.writerow(text)
#		#print "%s %d %d %d %.3g" % (engagement_values[i][0], engagement_values[i][1],engagement_values[i][2],engagement_values[i][3],one_rank_scores[j][1])
#			temp_list = [engagement_values[i][1],one_rank_scores[j][1]]
#			one_rank_correlation_pageload_score.append(temp_list)
#			temp_list = [engagement_values[i][2],one_rank_scores[j][1]]
#			one_rank_correlation_widget_score.append(temp_list)
#			temp_list = [engagement_values[i][3],one_rank_scores[j][1]]
#			one_rank_correlation_engagement_score.append(temp_list)
#	if engagement_values[i][0] in one_rank_scores
	      #  else: continue
print one_ranked_correlation_matched_indices
with open('one_rank_correlation_v10_' + data_set + '.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(one_ranked_correlation_matched_indices)):
	print one_ranked_correlation_matched_indices[i]
	spamwriter.writerow(one_ranked_correlation_matched_indices[i])

csvfile.close()

#w_file.close()
#csvfile.close()

#correlation = [zip(*one_rank_scores)[1],zip(*two_rank_scores)[1],zip(*three_rank_scores)[2]]

#w_file = open('histograms_data.csv','wb')
#for i in range(len(zip(*one_rank_scores)[1])):	
#	#print zip(*one_rank_scores)[1][i]
#	text="%.2g,%.2g,%.2g \r\n " % (zip(*one_rank_scores)[1][i],zip(*two_rank_scores)[1][i],zip(*three_rank_scores)[1][i])
#	w_file.write(text)

#w_file.close()

#print widget_scores
#print zip(*three_rank_scores)[1]
#plt.figure(1)
#plt.hist(zip(*one_rank_scores)[1],np.arange(-1,20,0.1),log="True", label="1 Rank")
#plt.hist(zip(*two_rank_scores)[1],np.arange(-1,20,0.1),log="True", label="2 Rank")
#plt.hist(zip(*three_rank_scores)[1],np.arange(-1,20,0.1),log="True", label="3 Rank")
#plt.legend(loc='upper right')
#plt.show()
#print score_ranks
plt.figure(1)
plt.subplot(1,2,1)
plt.xlabel("Widgets")
plt.ylabel("Total Score")
plt.bar(np.arange(0,31,1),widget_scores,1)
plt.subplot(1,2,2)
plt.bar(np.arange(0,31,1),widget_placements,1)
plt.xlabel("Widgets")
plt.ylabel("Total Placement")
plt.show()
plt.savefig("figures/WidgetStats.pdf")
