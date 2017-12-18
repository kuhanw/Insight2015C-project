#!/usr/bin/python
###Kuhan Wang, October 1st, 2015

import re
import json
import sys

reload(sys)
sys.setdefaultencoding('utf8')

#Use this to add stop words that are bi-grams and/or strings, lower case only

processing_list = ["2013", "Sign up for the email list and get you finances in shape!".lower(), "notify me of follow-up comments by email. notify me of new posts by email.get free access to our money dominating tool kit, and 2 chapters of my best selling book, soldier of finance. these resources will give you the tools you need to start building real wealth and positively impact your life today.let's kick butt together.".lower(), "november 14, 2013 by jeff rose 54 commentsmay 22, 2014 by jeff rose 45 commentsjune 28, 2013 by jeff rose 35 commentsjune 6, 2012 by jeff rose 27 commentsnovember 30, 2011 by jeff rose 25 commentsmarch 12, 2012 by jeff rose 24 commentsnovember 18, 2013 by jeff rose 19 commentsdecember 27, 2013 by jeff rose 17 commentsmay 23, 2012 by jeff rose 16 commentsjanuary 8, 2013 by jeff rose 12 commentssign up for the email list and get you finances in shape!over 35,000 awesome financial people have joined the good financial cents community - and you're awesome, too!".lower(), "notify me of followup comments via e-mail".lower(), "good idea", "photo credit", "photo credit: flickr", "\u2026", "learn something new every day \nmore info...\n   by email\n", "related article", "One of our editors will review your suggestion and make changes if warranted.\n      Note that depending on the number of suggestions we receive,\n      this can take anywhere from a few hours to a few days.\n      Thank you for helping to improve wiseGEEK!\n".lower()]

def replace_bigrams(input_paragraph):
        output_paragraph = input_paragraph.lower()

        for i in range(len(processing_list)):
                output_paragraph = output_paragraph.replace(processing_list[i],"")

        return output_paragraph

def read_json(json_file, widget, page_loaded_cut, widget_viewed_cut):
	print json_file
	corpus = []
	engagement_rate = []
	page_stats = [[], [] ,[], []]

	with open(json_file, 'r') as handle:
		text_data = handle.read()
	        text_data = '[' + re.sub(r'\}\s\{', '},{', text_data) + ']'
  	 	json_data = json.loads(text_data)

	with open("body_text_" + widget + ".txt", 'wb') as txtfile:
		for line in range(len(json_data)):
			if json_data[line]["pageloaded"]<page_loaded_cut: continue 
			if json_data[line]["widgetviewed"]<widget_viewed_cut: continue 
    			if widget not in str(json_data[line]["type"]): continue
     			#print "url:%s, type:%s" % (str(json_data[line]["title"]), str(json_data[line]["type"]))
     			if json_data[line]["widgetviewed"]!=0: engagement_rate.append(json_data[line]["widgetused"]/json_data[line]["widgetviewed"])
        		else: engagement_rate.append(0)
        		join_body="".join(json_data[line]["body"])
			pruned_body = replace_bigrams(join_body)
        		corpus.append(pruned_body)
        		txtfile.write('URL:%s \n' % str(json_data[line]["url"]) )
        		txtfile.write(pruned_body)
        		txtfile.write('\n')
        		page_stats[0].append(json_data[line]["pageloaded"])
        		page_stats[1].append(json_data[line]["widgetviewed"])
        		page_stats[2].append(json_data[line]["widgetused"])
        		page_stats[3].append(str(json_data[line]["title"]))
	return corpus, engagement_rate, page_stats
	txtfile.close()

