import ROOT

#ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(1)

all_lines = []

toys = 500 
Min_DF = 0.02
PagesLoaded = 0
widget = 'budgetcalculator'
#widget = 'homeaffordability'
#widget = 'assetallocationcalculator'
#widget = 'careercalculator'

x_ticks = []
widget_counts = []
	
with open('widget_placement_data.csv') as widget_placement:
	for line_values in widget_placement:
		print line_values
		x_ticks.append(line_values.split(',')[3])
		widget_counts.append(line_values.split(',')[2])


hist_0 = ROOT.TH1D("hist_0", "hist_0", 12, -0.5, 11.5)
for i in range(len(widget_counts)):
	hist_0.SetBinContent(i+1,int(widget_counts[i]))

c_hist_0 = ROOT.TCanvas("c_hist_0", "c_hist_0", 0, 0, 2400, 2000)
c_hist_0.SetLogy()
hist_0.GetXaxis().SetTitle("Asset Type")
hist_0.GetYaxis().SetTitle("Placement Count")
hist_0.SetTitle("Asset Placement Counts")
hist_0.Draw()
c_hist_0.SaveAs("widgetPlacementCounts.pdf")

print x_ticks
#print widget_counts

if widget=="budgetcalculator": hist_title="Asset Type 1"
if widget=="careercalculator": hist_title="Asset Type 2"
if widget=="homeaffordability": hist_title="Asset Type 3"
if widget=="assetallocationcalculator": hist_title="Asset Type 4"


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
hist_4 = ROOT.TH2D("hist_4", "hist_4", 100, 0, 1, 200, 0, 2)
hist_5 = ROOT.TH2D("hist_5", "hist_5", 100, 0, 1, 100, 0, 1)
hist_6 = ROOT.TH2D("hist_6", "hist_6", 100, 0, 1, 200, 0, 2)
hist_7 = ROOT.TH2D("hist_7", "hist_7", 100, 0, 1, 200, 0, 2)
hist_8 = ROOT.TH2D("hist_8", "hist_8", 100, 0, 1, 100, 0, 1)
hist_9 = ROOT.TH2D("hist_9", "hist_9", 100, 0, 1, 200, 0, 2)
hist_10 = ROOT.TH2D("hist_10", "hist_10", 100, 0, 1, 100, 0, 1)

total_line_value_0 = []
print "length of all_lines:%d" % len(all_lines)
for i in range(0,len(all_lines),1):
	#print all_lines[i][3]
#	print all_lines[i][3].split(',')[1][1:-1]
	line_value_0 = str(all_lines[i][3].split(',')[1][1:-1])		
#	print line_value_0
#	print all_lines[i][4]
	class_values = float(all_lines[i][4].split(',')[0].split(':')[1]) #precision
	class_values_2 = float(all_lines[i][4].split(',')[1].split(':')[1]) #recall
	class_values_3 = float(all_lines[i][4].split(',')[2].split(':')[1]) #accuracy
	f1 = 2*class_values*class_values_2/(class_values+class_values_2)
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
	#print "%d, class_value:%.3g" % (i, class_values) #precision
	#print "%d, class_value_2:%.3g" % (i, class_values_2) #recall
	#print "%d, class_value_3:%.3g" % (i, class_values_3) #accuracy
	#print "%d, %s" % (i, line_value_0_list)
  	line_value_0_temp_list_avg=(line_value_0_list[0]+line_value_0_list[1]+line_value_0_list[2])/3.
	hist_1.Fill(line_value_0_temp_list_avg, class_values)
	hist_2.Fill(class_values, class_values_2)
	hist_3.Fill(class_values, class_values_3)
	hist_4.Fill(class_values, lift)
	hist_5.Fill(class_values_2, class_values_3)
	hist_6.Fill(class_values_2, lift)
	hist_7.Fill(class_values_3, lift)
	hist_8.Fill(f1,class_values_3)
	hist_9.Fill(f1,lift)
	hist_10.Fill(f1,class_values_2)

c_hist_1 = ROOT.TCanvas("c_hist_1", "c_hist_1", 0, 0, 2400, 2000)
hist_1.GetXaxis().SetTitle('Avg of 3-fold Training Precision')
hist_1.GetYaxis().SetTitle('Test Sample Precision')
hist_1.SetTitle(hist_title)
hist_1.Draw('COLZ')
c_hist_1.SaveAs('k_fold_valid_p_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_2 = ROOT.TCanvas("c_hist_2", "c_hist_2", 0, 0, 2400, 2000)
hist_2.GetXaxis().SetTitle('Precision')
hist_2.GetYaxis().SetTitle('Recall')
hist_2.SetTitle(hist_title)
hist_2.Draw('COLZ')
c_hist_2.SaveAs('k_fold_valid_pvsr_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_3 = ROOT.TCanvas("c_hist_3", "c_hist_3", 0, 0, 2400, 2000)
hist_3.GetXaxis().SetTitle('Precision')
hist_3.GetYaxis().SetTitle('Accuracy')
hist_3.SetTitle(hist_title)
hist_3.Draw('COLZ')
c_hist_3.SaveAs('k_fold_valid_pvsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_4 = ROOT.TCanvas("c_hist_4", "c_hist_4", 0, 0, 2400, 2000)
hist_4.GetXaxis().SetTitle('Precision')
hist_4.GetYaxis().SetTitle('Lift')
hist_4.SetTitle(hist_title)
hist_4.Draw('COLZ')
c_hist_4.SaveAs('k_fold_valid_pvsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_5 = ROOT.TCanvas("c_hist_5", "c_hist_5", 0, 0, 2400, 2000)
hist_5.GetXaxis().SetTitle('Recall')
hist_5.GetYaxis().SetTitle('Accuracy')
hist_5.SetTitle(hist_title)
hist_5.Draw('COLZ')
c_hist_5.SaveAs('k_fold_valid_rvsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_6 = ROOT.TCanvas("c_hist_6", "c_hist_6", 0, 0, 2400, 2000)
hist_6.GetXaxis().SetTitle('Recall')
hist_6.GetYaxis().SetTitle('Lift')
hist_6.SetTitle(hist_title)
#hist_6.SetTitle("Asset 1")
hist_6.Draw('COLZ')
c_hist_6.SaveAs('k_fold_valid_rvsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_7 = ROOT.TCanvas("c_hist_7", "c_hist_7", 0, 0, 2400, 2000)
hist_7.GetXaxis().SetTitle('Accuracy')
hist_7.GetYaxis().SetTitle('Lift')
hist_7.SetTitle(hist_title)
hist_7.Draw('COLZ')
c_hist_7.SaveAs('k_fold_valid_avsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_8 = ROOT.TCanvas("c_hist_8", "c_hist_8", 0, 0, 2400, 2000)
hist_8.GetXaxis().SetTitle('f1')
hist_8.GetYaxis().SetTitle('Accuracy')
hist_8.SetTitle(hist_title)
hist_8.Draw('COLZ')
c_hist_8.SaveAs('k_fold_valid_f1vsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_9 = ROOT.TCanvas("c_hist_9", "c_hist_9", 0, 0, 2400, 2000)
hist_9.GetXaxis().SetTitle('f1')
hist_9.GetYaxis().SetTitle('Lift')
hist_9.SetTitle(hist_title)
hist_9.Draw('COLZ')
c_hist_9.SaveAs('k_fold_valid_f1vsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')

c_hist_10 = ROOT.TCanvas("c_hist_10", "c_hist_10", 0, 0, 2400, 2000)
hist_10.GetXaxis().SetTitle('f1')
hist_10.GetYaxis().SetTitle('Recall')
hist_10.SetTitle("Asset Type 1")
#hist_10.SetTitle(hist_title)
hist_10.Draw('COLZ')
c_hist_10.SaveAs('k_fold_valid_f1vsr_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1'+ '.pdf')
