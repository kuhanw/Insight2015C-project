from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, datasets, linear_model
from sklearn.feature_extraction import text 
import logging
import sys
import matplotlib.pyplot as plt
from time import time
import csv
import numpy as np
import re
import json

###LOAD JSON

#json_file = "web_text_v9_budgetcalculator.json"
#json_file = "web_text_v9.json"
json_file = "web_text_v9c.json"

with open(json_file, 'r') as handle:
    text_data = handle.read()
    text_data = '[' + re.sub(r'\}\s\{', '},{', text_data) + ']'
    json_data = json.loads(text_data)

corpus = [] 
engagement_rate = []

page_stats = [[], [] ,[], []]

for line in range(len(json_data)):
        if json_data[line]["pageloaded"]<2: continue
#        if json_data[line]["pageloaded"]>1500: continue
#        if json_data[line]["widgetviewed"]<1: continue
	if "budgetcalculator" not in str(json_data[line]["type"]): continue
	print "url:%s, type:%s" % (str(json_data[line]["title"]), str(json_data[line]["type"]))
	if json_data[line]["widgetviewed"]!=0: engagement_rate.append(json_data[line]["widgetused"]/json_data[line]["widgetviewed"])
	else: engagement_rate.append(0)
	join_body="".join(json_data[line]["body"])
	corpus.append(join_body)
	page_stats[0].append(json_data[line]["pageloaded"])
	page_stats[1].append(json_data[line]["widgetviewed"])
	page_stats[2].append(json_data[line]["widgetused"])
	page_stats[3].append(str(json_data[line]["title"]))

my_words = ["concept", "photo", "four", "four eyes", "like" ,"comes", "things", "com","don","help"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

#build vocabulary matrix
print "size of corpus:%d" % len(corpus)
#vectorizer = CountVectorizer(analyzer="word", stop_words="english", decode_error="strict", ngram_range=(1,3))

vectorizer = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error="strict", ngram_range=(1,2),  min_df=0.05)#, max_df=0.85)
vectorizer_binary = CountVectorizer(analyzer="word", stop_words=set(my_stop_words), decode_error="strict", ngram_range=(1,2),  min_df=0.05, binary="True")#, max_df=0.85)

X = vectorizer.fit_transform(corpus)

corpus_array = X.toarray()
print "#######vectorizer stop words############"
print vectorizer.get_stop_words()
print "#######vocabulary########"
print vectorizer.vocabulary_
print corpus_array

#with open('diagonostic_stop_words.csv', 'wb') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(vectorizer.get_stop_words())
#csvfile.close()
#with open('diagonostic_vocabulary.csv', 'wb') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(vectorizer.vocabulary_)
#csvfile.close()

##reweight usingTf-idf term weighting

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(corpus_array)

tfidf_array = tfidf.toarray()

print tfidf_array

print engagement_rate
print tfidf_array.shape

print len(engagement_rate)

X = np.array(tfidf_array)
y = np.array(engagement_rate)
print X
#print "engagement rate:"
#for i in range(len(y)) :
#	print y[i]

Max_Iter=10000
Fit_Intercept=False
Return_Models=False
Positive=True
Verbose=False
N_Jobs=-1
N_Alphas=1000
Normalize=False
CV=3
Alphas=[0]
Tol=0.001

coef_path_lasso_cv = LassoCV(normalize=Normalize, max_iter=Max_Iter, copy_X=True, cv=CV, verbose=Verbose, fit_intercept=Fit_Intercept, tol=Tol)#, alphas=Alphas) 
coef_path_elastic_cv = ElasticNetCV(normalize=Normalize,max_iter=Max_Iter, tol=Tol)#,alphas=Alphas)
coef_path_logistic_cv = LogisticRegression( tol=Tol)
coef_path_binary_x_logistic_cv = LogisticRegression( tol=Tol)

binary_y_pre = []

for i in range(len(y)):
	if y[i]>0: binary_y_pre.append(1)
	else: binary_y_pre.append(0)
binary_y = np.array(binary_y_pre)
print "binary y"
print binary_y

binary_X = vectorizer_binary.fit_transform(corpus)

print "########LASSO######"
coef_path_lasso_cv.fit(X,y)
print coef_path_lasso_cv.get_params
print "alphas:" 
print  coef_path_lasso_cv.alphas_
print "coef_:"
print coef_path_lasso_cv.coef_
lasso_prediction = coef_path_lasso_cv.predict(X)
lasso_score = coef_path_lasso_cv.score(X,y)
print "Lasso_score:%.3g" % lasso_score
#print "Lasso precision:%.3g" %  precision_score(y, lasso_predict) 
#print "Lasso_confusion matrix:"
#print confusion_matrix(y, lasso_prediction)
lasso_cv_score = cross_validation.cross_val_score(coef_path_lasso_cv, X, y, n_jobs=2, cv=5)
print lasso_cv_score
print "#######ELASTIC#####"
coef_path_elastic_cv.fit(X,y)
print coef_path_elastic_cv.get_params
print "alphas:" 
print  coef_path_elastic_cv.alphas_
print "coef_:"
print coef_path_elastic_cv.coef_
print "length of elastic terms:%d" % len(coef_path_elastic_cv.coef_)
elastic_predict = coef_path_elastic_cv.predict(X)
elastic_score = coef_path_elastic_cv.score(X,y)
print "elastic_score:%.3g" % elastic_score
elastic_cv_score = cross_validation.cross_val_score(coef_path_elastic_cv, X, y, n_jobs=2, cv=5)
print elastic_cv_score
#print "elastic precision:%.3g" %  precision_score(y, elastic_predict, average='macro') 
print "#######Logistic#####"
coef_path_logistic_cv.fit(X,binary_y)
print coef_path_logistic_cv.get_params
print "coef_:"
print coef_path_logistic_cv.coef_
print "length of coefficient terms %d" % len(coef_path_logistic_cv.coef_[0])
logistic_prediction = coef_path_logistic_cv.predict(X)
print logistic_prediction
logistic_score = coef_path_logistic_cv.score(X,binary_y)
print "logistic_score:%.3g" % logistic_score
print "logistic precision:%.3g" %  precision_score(binary_y, logistic_prediction, average="binary") 
print "logistic confusion matrix:"
print confusion_matrix(binary_y, logistic_prediction)
print "logistic classification report:"
print  classification_report(binary_y, logistic_prediction)
logistic_cv_score = cross_validation.cross_val_score(coef_path_logistic_cv, X, binary_y, n_jobs=2, cv=3)
print "Accuracy: %0.4g (+/- %.4g)" % (logistic_cv_score.mean(), logistic_cv_score.std() * 2)
print logistic_cv_score
plt.figure()
plt.scatter(binary_y, logistic_prediction)
plt.ylabel("Predicted")
plt.xlabel("Truth")
plt.show()

print "#######Binary X Logistic#####"
coef_path_binary_x_logistic_cv.fit(binary_X,binary_y)
print coef_path_binary_x_logistic_cv.get_params
print "coef_:"
print coef_path_binary_x_logistic_cv.coef_
print "length of coefficient terms %d" % len(coef_path_binary_x_logistic_cv.coef_[0])
binary_x_logistic_prediction = coef_path_binary_x_logistic_cv.predict(binary_X)
print binary_x_logistic_prediction
binary_x_logistic_score = coef_path_binary_x_logistic_cv.score(binary_X,binary_y)
print "binary_x_logistic_score:%.3g" % binary_x_logistic_score
print "binary_x_logistic precision:%.3g" %  precision_score(binary_y, binary_x_logistic_prediction, average="binary") 
print "b-logistic confusion matrix:"
print confusion_matrix(binary_y, binary_x_logistic_prediction)
print "b-logistic classification report:"
print  classification_report(binary_y, binary_x_logistic_prediction)
print "#############"
print binary_y
print "#####"
print binary_x_logistic_prediction

print "width:%d" % len(X[0])
print "length:%d" % len(X)
print "feature length:%d" % len(vectorizer.get_feature_names())
binary_x_logistic_cv_score = cross_validation.cross_val_score(coef_path_binary_x_logistic_cv, binary_X, binary_y, n_jobs=2, cv=3)
print binary_x_logistic_cv_score
print "Accuracy: %0.4g (+/- %.4g)" % (binary_x_logistic_cv_score.mean(), binary_x_logistic_cv_score.std() * 2)

word_priority = []
for i in range(len(vectorizer.get_feature_names())):
	word_priority_list = [vectorizer.get_feature_names()[i], coef_path_lasso_cv.coef_[i], coef_path_elastic_cv.coef_[i], coef_path_logistic_cv.coef_[0][i], coef_path_binary_x_logistic_cv.coef_[0][i]]
	word_priority.append(word_priority_list)
#	print "b_lasso:%.2g, b_elastic_net:%.2g, b_logistic:%.2g, word:%s" % (coef_path_lasso_cv.coef_[i], coef_path_elastic_cv.coef_[i], coef_path_logistic_cv.coef_[0][i], vectorizer.get_feature_names()[i])

word_priority_lasso = sorted (word_priority, key= lambda x: float(x[1]))
word_priority_elastic = sorted (word_priority, key= lambda x: float(x[2]))
word_priority_logistic = sorted (word_priority, key= lambda x: float(x[3]))
word_priority_binary_logistic = sorted (word_priority, key= lambda x: float(x[4]))

print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

ranked_key_words = []

for i in range(len(word_priority_lasso)):
	ranked_key_words_list = [word_priority_lasso[i][1], word_priority_lasso[i][0],word_priority_elastic[i][2], word_priority_elastic[i][0],word_priority_logistic[i][3], word_priority_logistic[i][0],word_priority_binary_logistic[i][4], word_priority_binary_logistic[i][0]]
	ranked_key_words.append(ranked_key_words_list)
#	print ranked_key_words[i]

ranked_words_header = [["lasso rank"],["lasso word"],["elastic rank"],["elastic word"],["logistic rank"],["logistic word"],["b-logistic rank"],["b-logistic word"]]
with open('ranked_words_v01.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(ranked_words_header)
    for i in range(len(ranked_key_words)):
       	spamwriter.writerow(ranked_key_words[i])

csvfile.close()	
#	print "lasso_rank:%.3g, word:%s | elastic_rank:%.3g, word:%s | logistic_rank:%.3g, word:%s" % (word_priority_lasso[i][0],word_priority_lasso[i][1],word_priority_elastic[i][2],word_priority_elastic[i][0],word_priority_logistic[i][3],word_priority_logistic[i][3])
#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#for i in range(len(word_priority_elastic)):
#	print word_priority_elastic[i]
#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#for i in range(len(word_priority_lasso)):
#	print word_priority_logistic[i]

#print "Prediction:"
#print coef_path_lasso_cv.predict(X)
#print len(coef_path_lars)
#print coef_path_lars
#print "coeff_path is %d x %d:" (len(coef_path_lars), len(coef_path_lars[1]))

#lasso_zeros=[0]*len(coeff_path_lars_alpha)


#transpose_coef_path_lars = coef_path_lars.T

#for i in range(len(transpose_coef_path_lars)):
#	for j in range(len(transpose_coef_path_lars[i])):
#		if abs(float(transpose_coef_path_lars[i][j]))<0.0001: 
#			lasso_zeros[i]=lasso_zeros[i]+1
#	lasso_zeros[i]=1.-lasso_zeros[i]/float(len(coef_path_lars))

#plt.figure(1)
#plt.scatter(-1*np.log(coeff_path_lars_alpha),lasso_zeros)
#plt.ylabel("Percentage of non-zero Coefficients")
#plt.xlabel(r"$\displaystyle\alpha")
#plt.show()
