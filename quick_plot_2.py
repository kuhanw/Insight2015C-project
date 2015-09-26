import ROOT

ROOT.gStyle.SetOptStat(11111)
ROOT.gStyle.SetPalette(1)

hist_1 = ROOT.TH2D("hist_1", "hist_1", 100, 0, 1, 100, 0, 1)
with open('validation_stats.csv') as values:
	for line in values:
		print line
		hist_1.Fill(float(line.split(',')[3]),float(line.split(',')[5]))

c_hist_1 = ROOT.TCanvas("c_hist_1", "c_hist_1", 0, 0, 2400, 2000)
hist_1.GetXaxis().SetTitle('Avg of 3-fold Training f1')
hist_1.GetYaxis().SetTitle('Test Sample f1')
hist_1.Draw('COLZ')
c_hist_1.SaveAs('k_fold_valid_f1_result.png')
