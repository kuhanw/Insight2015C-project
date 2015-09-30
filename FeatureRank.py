from processing import *
from feature_extraction_model import *
import argparse
import os, sys
def main(argv):

        '''This function parses and return arguments passed in'''
        # Assign description to the help doc
        parser = argparse.ArgumentParser(
        description='Execute text analysis pipeline.')
        # Add arguments
        parser.add_argument(
        '-json', '--Input', type=str, help='Input JSON file [string]', required=True)
        parser.add_argument(
        '-w', '--widget_selection', type=str, help='Widget type [str]', required=True)
        parser.add_argument(
        '-nL', '--Ngram_Range_Low', type=int, help='Lower n-gram range [int]', required=True)
        parser.add_argument(
        '-nH', '--Ngram_Range_High', type=int, help='Upper n-gram range [int]', required=True)
        parser.add_argument(
        '-mDF', '--Min_DF', type=float, help='Minimum term document frequency between 0 and 1 [float]', required=True)
        parser.add_argument(
        '-pLoad', '--PageLoaded', type=int, help='Minimum page views [int]', required=True)
        parser.add_argument(
        '-wView', '--WidgetViewed', type=int, help='Minimum widget views [int]', required=True)
        parser.add_argument(
        '-Seed', '--ite', type=int, help='Random seed [int]', required=True)
        parser.add_argument(
        '-f', '--Find', type=float, help='Find keywords on relative percentage of data set between 0 and 1 [float]', required=False)
        # Array for all arguments passed to script
        args = parser.parse_args()
        # Assign args to variables
        Input = args.Input
        widget_selection = args.widget_selection
        Ngram_Range_Low = args.Ngram_Range_Low
        Ngram_Range_High = args.Ngram_Range_High
        Min_DF = args.Min_DF
        PageLoaded = args.PageLoaded
        WidgetViewed = args.WidgetViewed
        ite = args.ite
        Find = args.Find
        # Return all variable values

	try:
        	if os.stat(Input).st_size > 0:
       			print "json file is:%s" % Input
        	else:
        		print "json file is empty."
			sys.exit(0)
        except OSError:
    		print "json file does not exist." 	
		sys.exit(0)

	model_results, model_scores, X, y, widget_selection, list_of_features, \
		Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test = \
			feature_extraction_model(widget_selection, Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, Find, Input)

        post_processing(model_results, model_scores, X, y, widget_selection, list_of_features, 
                        Ngram_Range_Low, Ngram_Range_High, Min_DF, PageLoaded, WidgetViewed, ite, x_test, Find)

if __name__ == "__main__":
        main(sys.argv[1:])

