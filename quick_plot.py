import ROOT

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)

all_lines = []

toys = 3 
Min_DF = 0.02
PagesLoaded = 0
#widget = 'budgetcalculator'
#widget = 'homeaffordability'
#widget = 'assetallocationcalculator'
widget = 'careercalculator'

for i in range(2,toys,1):
	with open('figures/' + widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1_iteration'+str(i)+'.txt') as values:
		count=0
		line_total = []
       		for line in values:
			line_total.append(line)
			count=count+1
	all_lines.append(line_total)

cv_values = []
test_values = []
hist_1 = ROOT.TH2D("hist_1", "hist_1", 100, 0, 1, 100, 0, 1)

total_line_value_0 = []

#for i in range(len(all_lines[0])):
#	print "%d, %s" % (i, all_lines[0][i])
print len(all_lines)
#print all_lines
for i in range(0,len(all_lines),1):
#	print all_lines[i][13]
	try:
		print all_lines[i][13].split(',')[1][1:-1]
		line_value_0 = str(all_lines[i][13].split(',')[1][1:-1])		
		
		class_values = []
		for j in range(len(all_lines[i][24].split(' '))):
			try: 
		#		print "%d, %.3g" % (j, float(all_lines[i][27].split(' ')[j]))
				class_values.append(float(all_lines[i][24].split(' ')[j]))
			except: continue #print "novalue"
		print "%d, class_value:%.3g" % (i, class_values[1])
		#print all_lines[i][16].split(',')[1][1:-1]		
		line_value_0_length = len(line_value_0.split(' '))	
		line_value_0_temp_list = []
		for i in range(line_value_0_length):
			try: 
				line_value_0_float = float(line_value_0.split(' ')[i])
		#		print line_value_0_float
				line_value_0_temp_list.append(line_value_0_float)
			except: continue #print "nothing"
		total_line_value_0.append(line_value_0_temp_list)
		print line_value_0_temp_list
                line_value_0_temp_list_avg = (line_value_0_temp_list[0]+line_value_0_temp_list[1]+line_value_0_temp_list[2])/3.
		print "%d, avg of listed:%.3g" % (i, line_value_0_temp_list_avg)
                hist_1.Fill(line_value_0_temp_list_avg, class_values[1])
	#	f1_tested_value = float(all_lines[i][9].split(':')[1].split(',')[0])
#                hist_1.Fill(line_value_0_temp_list_avg, f1_tested_value)
	except: continue


text_hist_1 = ROOT.TPaveText(.15,.1,.75,.2)
text_hist_1.SetFillColor(10)
text_hist_1.SetLineColor(10)
text_hist_1.SetTextSize(0.03)
text_hist_1.AddText(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
c_hist_1 = ROOT.TCanvas("c_hist_1", "c_hist_1", 0, 0, 2400, 2000)
hist_1.GetXaxis().SetTitle('Avg of 3-fold Training f1')
hist_1.GetYaxis().SetTitle('Test Sample f1')
hist_1.Draw('COLZ')
text_hist_1.Draw()
c_hist_1.SaveAs('k_fold_valid_f1_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

