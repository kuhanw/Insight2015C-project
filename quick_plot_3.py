import ROOT

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)

all_lines = []

toys = 100 
Min_DF = 0.02
PagesLoaded = 0
#widget = 'budgetcalculator'
#widget = 'homeaffordability'
#widget = 'assetallocationcalculator'
widget = 'careercalculator'
print 'figures/' + widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1_iteration'+str(2)+'.txt'
for i in range(1,toys,1):
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
hist_2 = ROOT.TH2D("hist_2", "hist_2", 100, 0, 1, 100, 0, 1)
hist_3 = ROOT.TH2D("hist_3", "hist_3", 100, 0, 1, 100, 0, 1)
hist_4 = ROOT.TH2D("hist_4", "hist_4", 100, 0, 1, 300, 0, 3)
hist_5 = ROOT.TH2D("hist_5", "hist_5", 100, 0, 1, 100, 0, 1)
hist_6 = ROOT.TH2D("hist_6", "hist_6", 100, 0, 1, 300, 0, 3)
hist_7 = ROOT.TH2D("hist_7", "hist_7", 100, 0, 1, 300, 0, 3)

total_line_value_0 = []
print "length of all_lines:%d" % len(all_lines)
for i in range(0,len(all_lines),1):
	print all_lines[i][3]
#	print all_lines[i][3].split(',')[1][1:-1]
	line_value_0 = str(all_lines[i][3].split(',')[1][1:-1])		
#	print line_value_0
#	print all_lines[i][4]
	class_values = float(all_lines[i][4].split(',')[0].split(':')[1])
	class_values_2 = float(all_lines[i][4].split(',')[1].split(':')[1])
	class_values_3 = float(all_lines[i][4].split(',')[2].split(':')[1])
	conf_a = float(all_lines[i][4].split(',')[3].split(':')[1])
	conf_b = float(all_lines[i][4].split(',')[4])
	conf_c = float(all_lines[i][4].split(',')[5])
	conf_d = float(all_lines[i][4].split(',')[6])
#	print conf_b
#	print conf_c
#	print conf_d
	lift = ((conf_a/(conf_a+conf_b)))/((conf_a+conf_c)/(conf_a+conf_b+conf_c+conf_d))
	line_value_0_list = []
	for j in range(len(line_value_0.split(' '))):
		try: 
			#print float(line_value_0.split(' ')[j])
			line_value_0_list.append(float(line_value_0.split(' ')[j]))
		except: continue
	print "%d, class_value:%.3g" % (i, class_values)
	print "%d, class_value_2:%.3g" % (i, class_values_2)
	print "%d, class_value_3:%.3g" % (i, class_values_3)
	print "%d, %s" % (i, line_value_0_list)
  	line_value_0_temp_list_avg=(line_value_0_list[0]+line_value_0_list[1]+line_value_0_list[2])/3.
	hist_1.Fill(line_value_0_temp_list_avg, class_values)
	hist_2.Fill(class_values, class_values_2)
	hist_3.Fill(class_values, class_values_3)
	hist_4.Fill(class_values, lift)
	hist_5.Fill(class_values_2, class_values_3)
	hist_6.Fill(class_values_2, lift)
	hist_7.Fill(class_values_3, lift)
	#	f1_tested_value = float(all_lines[i][9].split(':')[1].split(',')[0])
#               hist_1.Fill(line_value_0_temp_list_avg, f1_tested_value)


c_hist_1 = ROOT.TCanvas("c_hist_1", "c_hist_1", 0, 0, 2400, 2000)
hist_1.GetXaxis().SetTitle('Avg of 3-fold Training Precision')
hist_1.GetYaxis().SetTitle('Test Sample Precision')
hist_1.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_1.Draw('COLZ')
c_hist_1.SaveAs('k_fold_valid_p_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_2 = ROOT.TCanvas("c_hist_2", "c_hist_2", 0, 0, 2400, 2000)
hist_2.GetXaxis().SetTitle('Precision')
hist_2.GetYaxis().SetTitle('Recall')
hist_2.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_2.Draw('COLZ')
c_hist_2.SaveAs('k_fold_valid_pvsr_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_3 = ROOT.TCanvas("c_hist_3", "c_hist_3", 0, 0, 2400, 2000)
hist_3.GetXaxis().SetTitle('Precision')
hist_3.GetYaxis().SetTitle('Accuracy')
hist_3.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_3.Draw('COLZ')
c_hist_3.SaveAs('k_fold_valid_pvsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_4 = ROOT.TCanvas("c_hist_4", "c_hist_4", 0, 0, 2400, 2000)
hist_4.GetXaxis().SetTitle('Precision')
hist_4.GetYaxis().SetTitle('Lift')
hist_4.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_4.Draw('COLZ')
c_hist_4.SaveAs('k_fold_valid_pvsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_5 = ROOT.TCanvas("c_hist_5", "c_hist_5", 0, 0, 2400, 2000)
hist_5.GetXaxis().SetTitle('Recall')
hist_5.GetYaxis().SetTitle('Accuracy')
hist_5.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_5.Draw('COLZ')
c_hist_5.SaveAs('k_fold_valid_rvsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_6 = ROOT.TCanvas("c_hist_6", "c_hist_6", 0, 0, 2400, 2000)
hist_6.GetXaxis().SetTitle('Recall')
hist_6.GetYaxis().SetTitle('Lift')
hist_6.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_6.Draw('COLZ')
c_hist_6.SaveAs('k_fold_valid_rvsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_7 = ROOT.TCanvas("c_hist_7", "c_hist_7", 0, 0, 2400, 2000)
hist_7.GetXaxis().SetTitle('Accuracy')
hist_7.GetYaxis().SetTitle('Lift')
hist_7.SetTitle(widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1')
hist_7.Draw('COLZ')
c_hist_7.SaveAs('k_fold_valid_avsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')
