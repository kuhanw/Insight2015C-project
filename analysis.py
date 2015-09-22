import numpy as np
import csv
import ROOT
ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)
import matplotlib.pyplot as plt

widgets = ["budgetcalculator", "homeaffordability", "assetallocationcalculator", "careercalculator"]

data_sources = ["figures/"+widgets[0] + "/", "figures/"+widgets[1] + "/","figures/"+widgets[2] + "/","figures/"+widgets[3] + "/"]

widget_results = [ [], [] ,[] ,[] ]

with open(data_sources[0]+"ranked_words_ngram_" + ngram+ ".csv", 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
	for row in spamreader:
		widget_results[0].append(row)



data_points = []
y_test =  zip(*widget_results[0])

for i in range(1,len(y_test),1):
	if i<8: 
		temp_list = [y_test[0][i], y_test[1][i]]
		data_points.append(temp_list)
	#	print data_points[i]

plt.figure()
x = np.array(np.arange(0,len(zip(*data_points)[0]),1))
y = zip(*data_points)[0]
plt.xticks(x, zip(*data_points)[1])
plt.ylabel("Ranking Coefficient")
plt.title(widgets[0]+" " + y_test[0][0])
plt.plot(x, y, 'ro',markersize=12)
plt.show()
