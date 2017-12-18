###Kuhan Wang, October 1st, 2015

import json
import urllib2
from bs4 import BeautifulSoup

def url_pull(url_string):

	paragraph_list = []
	url_info = [0, 0]
	req = urllib2.Request(url_string, headers={'User-Agent' : "Magic Browser"})
	response = urllib2.urlopen(req)
	soup = BeautifulSoup(response.read())
	title_value = soup.title.string

	##CRAWL FOR PARAGRAPH DATA

	for link in soup.find_all('p'):
		paragraph_list.append(link.get_text())
	
	url_info = [title_value, paragraph_list]

	return url_info


start_urls = [[],[],[],[],[],[]]

##Input data_set_%d
data_set = "data_set_2"

source_csv = 'processed_' + data_set + '.csv'

with open(source_csv) as correlation_list:
        for line in correlation_list:
		if "www" not in line: continue
		start_urls[0].append(str(line.split(",")[0]))
      		start_urls[1].append(float(line.split(",")[1]))
      		start_urls[2].append(float(line.split(",")[2]))
		start_urls[3].append(float(line.split(",")[3]))
		start_urls[4].append(float(line.split(",")[4]))
		start_urls[5].append(str(line.split(",")[5]))

print len(start_urls) 

with open('web_text_' + data_set + '.json', "w") as outfile:
	for i in range(len(start_urls[0])):
		try:	
			url_result = url_pull(start_urls[0][i])
			print url_result[0]   
	    		json.dump({"title":url_result[0], "body":url_result[1], "pageloaded":start_urls[1][i]
					,"widgetviewed":start_urls[2][i], "widgetused":start_urls[3][i], "onerank":start_urls[4][i], 
					"type":start_urls[5][i], "url":start_urls[0][i]}, outfile, indent=4)
			outfile.write("\n")
			print "ACCESSED URL:%s" % start_urls[0][i]
		except: 
			print "COULD NOT ACCESS URL:%s" % start_urls[0][i]

outfile.close()
