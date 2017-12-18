###FeatureRank######
###Kuhan Wang, October 1st, 2015
###This is the main driver script.#####

from feature_extraction_model import *
import argparse
import os, sys

if __name__ == "__main__":

        parser = argparse.ArgumentParser(
        description='Execute text analysis pipeline FeatureRank.')

        parser.add_argument(
        '-json', '--Input', type = str, help = 'Input JSON file [string]', required = True)
        parser.add_argument(
        '-w', '--widget_selection', type = str, help = 'Widget type [string]', required = True)
        parser.add_argument(
        '-nL', '--Ngram_Range_Low', type = int, help = 'Lower n-gram range [int]', required = True)
        parser.add_argument(
        '-nH', '--Ngram_Range_High', type = int, help = 'Upper n-gram range [int]', required = True)
        parser.add_argument(
        '-mDF', '--Min_DF', type = float, help = 'Minimum term document frequency between 0 and 1 [float]', required = True)
        parser.add_argument(
        '-pLoad', '--PageLoaded', type = int, help = 'Minimum page views [int]', required = True)
        parser.add_argument(
        '-wView', '--WidgetViewed', type = int, help = 'Minimum widget views [int]', required = True)
        parser.add_argument(
        '-Seed', '--ite', type = int, help = 'Random seed [int]', required = False)
        parser.add_argument(
        '-f', '--Find', type = float, help = 'Find keywords on relative percentage of data set between 0 and 1 [float]', required = False)
        
	args = parser.parse_args()
        
	Input = args.Input
        widget_selection = args.widget_selection
        Ngram_Range_Low = args.Ngram_Range_Low
        Ngram_Range_High = args.Ngram_Range_High
        Min_DF = args.Min_DF
        PageLoaded = args.PageLoaded
        WidgetViewed = args.WidgetViewed
        ite = args.ite
        Find = args.Find

	#Throw an error if the JSON file is nonesense

	try:
        	if os.stat(Input).st_size > 0:
       			print "JSON file is:%s" % Input
        	else:
        		print "JSON file is empty!"
			sys.exit(0)
        except OSError:
    		print "JSON file does not exist!" 	
		sys.exit(0)
		
	if os.stat(Input).st_size > 0:

		##Model text against engagement
		model_results, model_scores, X, y, widget_selection, list_of_features, \
			Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test = \
				feature_extraction_model(widget_selection, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, Find, Input)

		##Extract relevant keywords and output to file 
		post_processing(model_results, model_scores, X, y, widget_selection, list_of_features,  
		                Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test, Find)


