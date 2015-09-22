import sys
import Quandl
import matplotlib.pyplot as plt
import datetime
import numpy as np
import random
import math
import time
from sklearn import cluster, covariance, manifold
from matplotlib.collections import LineCollection
from joblib import Parallel, delayed

correlation_list = [[],[],[],[],[],[],[]]

with open("one_rank_correlation_v1.csv", "rb") as read_list:
	for line in read_list:
		if float(line.split(",")[4])<0 : continue
		if float(line.split(",")[1])<5 : continue
		if float(line.split(",")[2])<5 : continue
		temp_list = [line.split(",")[1],line.split(",")[4]]
		correlation_list[0].append(temp_list)
		temp_list = [line.split(",")[2],line.split(",")[4]]
		correlation_list[1].append(temp_list)
		if float(line.split(",")[2])!=0 and float(line.split(",")[3])!=0: 
			if float(line.split(",")[3])/float(line.split(",")[2])==1: print line.split(",")[0]
			temp_list = [float(line.split(",")[3])/float(line.split(",")[2]),line.split(",")[4]]
			correlation_list[2].append(temp_list)
		temp_list = [float(line.split(",")[2])/float(line.split(",")[1]),line.split(",")[4]]
		correlation_list[3].append(temp_list)
		temp_list = [float(line.split(",")[2])/float(line.split(",")[1]),line.split(",")[3]]
		correlation_list[4].append(temp_list)
		if float(line.split(",")[2])!=0 and float(line.split(",")[3])!=0: 
			temp_list = [float(line.split(",")[2])/float(line.split(",")[1]),float(line.split(",")[3])/float(line.split(",")[2])]
			correlation_list[5].append(temp_list)
		if float(line.split(",")[3])/float(line.split(",")[2]) <0.05 and float(line.split(",")[2])>10 and float(line.split(",")[1])>100: print "engage:%.2g url:%s pageloaded:%.2g" % (float(line.split(",")[3])/float(line.split(",")[2]), line.split(",")[0], float(line.split(",")[1]))	
		temp_list = [int(line.split(",")[2]),int(line.split(",")[3])]
		correlation_list[6].append(temp_list)


#print correlation_list[0]

plt.figure(1)
plt.subplot(3,3,1)
plt.scatter(zip(*correlation_list[0])[0],zip(*correlation_list[0])[1])
plt.ylabel('Top Ranked Score')
plt.xlabel('Times Page Loaded')
plt.subplot(3,3,2)
plt.scatter(zip(*correlation_list[1])[0],zip(*correlation_list[1])[1])
plt.ylabel('Top Ranked Score')
plt.xlabel('Times Widget Visible')
plt.subplot(3,3,3)
plt.scatter(zip(*correlation_list[2])[0],zip(*correlation_list[2])[1])
plt.ylabel('Top Ranked Score')
plt.xlabel('Total Engagement/Times Widget Visible')
plt.subplot(3,3,4)
plt.scatter(zip(*correlation_list[3])[0],zip(*correlation_list[3])[1])
plt.ylabel('Top Ranked Score')
plt.xlabel('Times Widget Visible/Times Page Loaded')
plt.subplot(3,3,5)
plt.scatter(zip(*correlation_list[4])[0],zip(*correlation_list[4])[1])
plt.ylabel('Total Engagement')
plt.xlabel('Times Widget Visible/Times Page Loaded')
plt.subplot(3,3,6)
plt.scatter(zip(*correlation_list[5])[0],zip(*correlation_list[5])[1])
plt.ylabel('Total Engagement/Times Widget Visible')
plt.xlabel('Times Widget Visible/Times Page Loaded')
plt.subplot(3,3,7)
plt.scatter(zip(*correlation_list[6])[0],zip(*correlation_list[6])[1])
plt.ylabel('Times Widget Engaged')
plt.xlabel('Times Widget Visible')
plt.show()
