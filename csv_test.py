import csv

abc = ["abc",1,3]

with open('eggs.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(abc)
    spamwriter.writerow(abc)
    spamwriter.writerow(abc)
    spamwriter.writerow(abc)

