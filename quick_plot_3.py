import ROOT

ROOT.gStyle.SetOptStat(1111)
#ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetStatX(0.33)
ROOT.gStyle.SetStatY(0.37)
ROOT.gStyle.SetTitleSize(.05, "XY")
all_lines = []
toys_begin = 1 
toys = 100
Min_DF = 0.02
#Min_DF = 0.05
PagesLoaded = 5
Find = 0.25
widget = 'budgetcalculator'
#widget = 'homeaffordability'
#widget = 'assetallocationcalculator'
#widget = 'careercalculator'
algo = 'logistic'
#algo = 'SGD'


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
hist_0.SetMinimum(0.1001)
hist_0.GetXaxis().SetTitle("Ad Type")
hist_0.GetYaxis().SetTitle("Placement Count")
hist_0.SetTitle("Ad Placement Counts")
hist_0.SetFillColor(4)
#hist_0.SetLineColor(1)
hist_0.Draw('bar2')
c_hist_0.SaveAs("widgetPlacementCounts.pdf")

print x_ticks
#print widget_counts

if widget=="budgetcalculator": hist_title="Ad Type 1"
if widget=="careercalculator": hist_title="Ad Type 2"
if widget=="homeaffordability": hist_title="Ad Type 3"
if widget=="assetallocationcalculator": hist_title="Ad Type 4"


for i in range(toys_begin,toys,1):
	with open('figures/' + widget + '/validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_1_iteration'+str(i)+'_'+str(Find)+'.txt') as values:
		count=0
		line_total = []
       		for line in values:
			line_total.append(line)
			count=count+1
	all_lines.append(line_total)

cv_values = []
test_values = []
hist_1 = ROOT.TH2D("Stats", "hist_1", 100, 0, 1, 100, 0, 1)
hist_2 = ROOT.TH2D("Stats", "hist_2", 100, 0, 1, 100, 0, 1)
hist_3 = ROOT.TH2D("Stats", "hist_3", 100, 0, 1, 100, 0, 1)
hist_4 = ROOT.TH2D("Stats", "hist_4", 100, 0, 1, 200, 0, 2)
hist_5 = ROOT.TH2D("Stats", "hist_5", 100, 0, 1, 100, 0, 1)
hist_6 = ROOT.TH2D("Stats", "hist_6", 100, 0, 1, 200, 0, 2)
hist_7 = ROOT.TH2D("Stats", "hist_7", 100, 0, 1, 200, 0, 2)
hist_8 = ROOT.TH2D("Stats", "hist_8", 100, 0, 1, 100, 0, 1)
hist_9 = ROOT.TH2D("Stats", "hist_9", 100, 0, 1, 200, 0, 2)
hist_10 = ROOT.TH2D("Stats", "hist_10", 100, 0, 1, 100, 0, 1)
hist_11 = ROOT.TH2D("Stats", "hist_11", 100, 0, 1, 100, 0, 1)

total_line_value_0 = []

if algo=='logistic': line_parse=6
if algo=='SGD': line_parse=3
if algo=='forest': line_parse=0
print "algo:%s" %algo

print "length of all_lines:%d" % len(all_lines)
for i in range(0,len(all_lines),1):
	try:
		print "%d, %s" % (i, all_lines[i][line_parse])
		line_value_0 = str(all_lines[i][line_parse].split(',')[1][1:-1])		
	#	print line_value_0
	#	print all_lines[i][4]
		class_values = float(all_lines[i][line_parse+1].split(',')[0].split(':')[1]) #precision
		class_values_2 = float(all_lines[i][line_parse+1].split(',')[1].split(':')[1]) #recall
		class_values_3 = float(all_lines[i][line_parse+1].split(',')[2].split(':')[1]) #accuracy
		try: f1 = 2*class_values*class_values_2/(class_values+class_values_2)
		except: f1=0
		conf_a = float(all_lines[i][line_parse+1].split(',')[3].split(':')[1])
		conf_b = float(all_lines[i][line_parse+1].split(',')[4])
		conf_c = float(all_lines[i][line_parse+1].split(',')[5])
		conf_d = float(all_lines[i][line_parse+1].split(',')[6])
	#	print conf_b
	#	print conf_c
	#	print conf_d
		try: lift = ((conf_a/(conf_a+conf_b)))/((conf_a+conf_c)/(conf_a+conf_b+conf_c+conf_d))
		except: lift = 0
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
	
	except: continue


pt = ROOT.TPaveText(.15,.7,.25,.8);
pt.SetFillColor(10)
pt.SetLineColor(10)


c_hist_1 = ROOT.TCanvas("c_hist_1", "c_hist_1", 0, 0, 2400, 2000)
hist_1.GetXaxis().SetTitle('Avg of 3-fold Training Precision')
hist_1.GetYaxis().SetTitle('Test Sample Precision')
hist_1.SetTitle(hist_title)
hist_1.Draw('COLZ')
c_hist_1.SaveAs('k_fold_valid_p_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_2 = ROOT.TCanvas("c_hist_2", "c_hist_2", 0, 0, 2400, 2000)
hist_2.SetAxisRange(0.25,0.95,"X")
hist_2.SetAxisRange(0.25,0.95,"Y")
hist_2.GetXaxis().SetTitle('Precision')
hist_2.GetYaxis().SetTitle('Recall')
hist_2.SetTitle('Validation on 1000 Randomly Sampled Test Sets')
#hist_2.SetTitleSize(4)
hist_2.Draw('COLZ')
precision_text = "#langle P #rangle="+str(hist_2.GetMean(0))
print hist_2.GetMean(0)
print hist_2.GetMean(1)
#pt.AddText("#langle P #rangle="+str(hist_2.GetMean(1)));
#pt.Draw()
c_hist_2.SaveAs('k_fold_valid_pvsr_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_3 = ROOT.TCanvas("c_hist_3", "c_hist_3", 0, 0, 2400, 2000)
hist_3.GetXaxis().SetTitle('Precision')
hist_3.GetYaxis().SetTitle('Accuracy')
hist_3.SetTitle(hist_title)
hist_3.Draw('COLZ')
c_hist_3.SaveAs('k_fold_valid_pvsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_4 = ROOT.TCanvas("c_hist_4", "c_hist_4", 0, 0, 2400, 2000)
hist_4.GetXaxis().SetTitle('Precision')
hist_4.GetYaxis().SetTitle('Lift')
hist_4.SetTitle(hist_title)
hist_4.Draw('COLZ')
c_hist_4.SaveAs('k_fold_valid_pvsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_5 = ROOT.TCanvas("c_hist_5", "c_hist_5", 0, 0, 2400, 2000)
hist_5.GetXaxis().SetTitle('Recall')
hist_5.GetYaxis().SetTitle('Accuracy')
hist_5.SetTitle(hist_title)
hist_5.Draw('COLZ')
c_hist_5.SaveAs('k_fold_valid_rvsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_6 = ROOT.TCanvas("c_hist_6", "c_hist_6", 0, 0, 2400, 2000)
hist_6.GetXaxis().SetTitle('Recall')
hist_6.GetYaxis().SetTitle('Lift')
hist_6.SetTitle(hist_title)
#hist_6.SetTitle("Ad 1")
hist_6.Draw('COLZ')
c_hist_6.SaveAs('k_fold_valid_rvsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_7 = ROOT.TCanvas("c_hist_7", "c_hist_7", 0, 0, 2400, 2000)
hist_7.GetXaxis().SetTitle('Accuracy')
hist_7.GetYaxis().SetTitle('Lift')
hist_7.SetTitle(hist_title)
hist_7.Draw('COLZ')
c_hist_7.SaveAs('k_fold_valid_avsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_8 = ROOT.TCanvas("c_hist_8", "c_hist_8", 0, 0, 2400, 2000)
hist_8.GetXaxis().SetTitle('f1')
hist_8.GetYaxis().SetTitle('Accuracy')
hist_8.SetTitle(hist_title)
hist_8.Draw('COLZ')
c_hist_8.SaveAs('k_fold_valid_f1vsa_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_9 = ROOT.TCanvas("c_hist_9", "c_hist_9", 0, 0, 2400, 2000)
hist_9.GetXaxis().SetTitle('f1')
hist_9.GetYaxis().SetTitle('Lift')
hist_9.SetTitle(hist_title)
hist_9.Draw('COLZ')
c_hist_9.SaveAs('k_fold_valid_f1vsl_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')

c_hist_10 = ROOT.TCanvas("c_hist_10", "c_hist_10", 0, 0, 2400, 2000)
hist_10.GetXaxis().SetTitle('f1')
hist_10.GetYaxis().SetTitle('Recall')
hist_10.SetTitle("Ad Type 1")
#hist_10.SetTitle(hist_title)
hist_10.Draw('COLZ')
c_hist_10.SaveAs('k_fold_valid_f1vsr_result_' + widget  + '_validation_1_2_' + str(Min_DF) + '_' + str(PagesLoaded) + '_' + str(algo) + '_1'+'_'+str(Find)+ '.pdf')
