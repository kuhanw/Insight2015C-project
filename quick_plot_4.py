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
results_csv_1 = []
	
plot_results = ROOT.TH2D('2d_results','2d_results',4,-0.5,3.5,2416,0,2416)

with open('/home/euix/Dropbox/insight_project/project/figures/budgetcalculator/ranked_words_ngram_1_2_0.02_0_1.csv') as results_1:
	for line_values in results_1:
		try: 
			plot_results.SetBinContent(1, int(line_values.split(',')[0])+1, 10.*float(line_values.split(',')[1]) )
		except: continue

with open('/home/euix/Dropbox/insight_project/project/figures/careercalculator/ranked_words_ngram_1_2_0.02_0_1.csv') as results_1:
	for line_values in results_1:
		try: 
			plot_results.SetBinContent(2, int(line_values.split(',')[0])+1, 10.*float(line_values.split(',')[1]) )
		except: continue

with open('/home/euix/Dropbox/insight_project/project/figures/homeaffordability/ranked_words_ngram_1_2_0.02_0_1.csv') as results_1:
	for line_values in results_1:
		try: 
			plot_results.SetBinContent(3, int(line_values.split(',')[0])+1, 10.*float(line_values.split(',')[1]) )
		except: continue

with open('/home/euix/Dropbox/insight_project/project/figures/assetallocationcalculator/ranked_words_ngram_1_2_0.02_0_1.csv') as results_1:
	for line_values in results_1:
		try: 
			plot_results.SetBinContent(4, int(line_values.split(',')[0])+1, 10.*float(line_values.split(',')[1]) )
		except: continue
c0 = ROOT.TCanvas('c0','c0',0,0,800,1600)
c0.SetLogz()
plot_results.GetXaxis().SetTitle('Asset Type')
plot_results.GetYaxis().SetTitle('Features')
plot_results.SetTitle('Feature Importance')
plot_results.Draw('COLZ')
c0.SaveAs('plot_results.svg')
