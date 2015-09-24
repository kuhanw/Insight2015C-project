#!/usr/bin/python
# Filename: read_json.py
import re
import json
import sys

reload(sys)
sys.setdefaultencoding('utf8')

#Use this add stop words that are bi-grams and/or strings, lower case only

processing_list = ["good idea", "photo credit", "photo credit: flickr", "\u2026", "learn something new every day \nmore info...\n   by email\n", "related article", "One of our editors will review your suggestion and make changes if warranted.\n      Note that depending on the number of suggestions we receive,\n      this can take anywhere from a few hours to a few days.\n      Thank you for helping to improve wiseGEEK!\n".lower()]

def replace_bigrams(input_paragraph):
        output_paragraph = input_paragraph.lower()

        for i in range(len(processing_list)):
                output_paragraph = output_paragraph.replace(processing_list[i],"")

        return output_paragraph

def read_json(json_file, widget, page_loaded_cut, widget_viewed_cut):
	corpus = []
	engagement_rate = []
	page_stats = [[], [] ,[], []]

	with open(json_file, 'r') as handle:
		text_data = handle.read()
	        text_data = '[' + re.sub(r'\}\s\{', '},{', text_data) + ']'
  	 	json_data = json.loads(text_data)

	for line in range(len(json_data)):
		if json_data[line]["pageloaded"]<page_loaded_cut: continue 
		if json_data[line]["widgetviewed"]<widget_viewed_cut: continue 
    		if widget not in str(json_data[line]["type"]): continue
     		#print "url:%s, type:%s" % (str(json_data[line]["title"]), str(json_data[line]["type"]))
     		if json_data[line]["widgetviewed"]!=0: engagement_rate.append(json_data[line]["widgetused"]/json_data[line]["widgetviewed"])
        	else: engagement_rate.append(0)
        	join_body="".join(json_data[line]["body"])
      	        #print "######"
		#print replace_bigrams(join_body)
      	        #print "######"
        	corpus.append(replace_bigrams(join_body))
        	page_stats[0].append(json_data[line]["pageloaded"])
        	page_stats[1].append(json_data[line]["widgetviewed"])
        	page_stats[2].append(json_data[line]["widgetused"])
        	page_stats[3].append(str(json_data[line]["title"]))

	return corpus, engagement_rate, page_stats

