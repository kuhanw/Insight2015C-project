with open("result.csv") as engagement_list:
	for line in engagement_list:
		print line.split(",")[0]

