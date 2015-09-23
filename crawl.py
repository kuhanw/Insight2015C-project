import json
import urllib2
from bs4 import BeautifulSoup

def url_pull(url_string):

	paragraph_list = []
	url_info = [0,0]
	response = urllib2.urlopen(url_string)
	soup = BeautifulSoup(response.read())
#	soup = soup.encode('ascii')
	title_value = soup.title.string

	for link in soup.find_all('p'):
#		paragraph_value = link.get_text()
		paragraph_list.append(link.get_text())
	
	url_info = [title_value, paragraph_list]

	return url_info


start_urls = [[],[],[],[],[],[]]

data_set = "data_set_2"

#source_csv = "one_rank_correlation_v10.csv"
source_csv = 'one_rank_correlation_v10_'+data_set+'.csv'

with open(source_csv) as correlation_list:
        for line in correlation_list:
		if "www" not in line: continue
		print line
#		try: 
		start_urls[0].append(str(line.split(",")[0]))
      		start_urls[1].append(float(line.split(",")[1]))
      		start_urls[2].append(float(line.split(",")[2]))
		start_urls[3].append(float(line.split(",")[3]))
		start_urls[4].append(float(line.split(",")[4]))
		start_urls[5].append(str(line.split(",")[5]))
#		except: continue
print "#####"
#print start_urls


print len(start_urls) 

with open('web_text_v10c_' + data_set + '.json', "w") as outfile:
	for i in range(len(start_urls[0])):
		try: 
			url_result = url_pull(start_urls[0][i])
			print url_result[0]   
	    		json.dump({"title":url_result[0], "body":url_result[1], "pageloaded":start_urls[1][i]
					,"widgetviewed":start_urls[2][i], "widgetused":start_urls[3][i], "onerank":start_urls[4][i], 
					"type":start_urls[5][i], "url":start_urls[0][i]}, outfile, indent=4)
			outfile.write("\n")
		except: 
			print "Couldn't Access:%s " % start_urls[0][i]
			continue


outfile.close()
