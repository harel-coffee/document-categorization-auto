#!/usr/bin/python
# -*-coding:utf-8-*

import nltk

from nltk.corpus import reuters
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Perceptron
from sklearn import metrics

import matplotlib.pyplot as plt


################################################################################
# Usage
################################################################################
# python categ.py

################################################################################



################################################################################
# Functions
################################################################################

def features(words):
	features = {}
	for word in words:
		features[word] = True
	return features


def buildSet(t_set,docs):
	for doc in docs:
		categories = reuters.categories(doc)
		for category in categories:
			t_set.append((features(reuters.words(doc)), category))


def classify(features_selection,classification_method):
	pipeline = []

	if features_selection == "tf-idf":
		pipeline.append((features_selection, TfidfTransformer()))
	elif features_selection == "chi2":
		pipeline.append((features_selection, SelectKBest(chi2, k=1000)))

	if classification_method == "M-NB":
		pipeline.append((classification_method, MultinomialNB()))
	if classification_method == "B-NB":
		pipeline.append((classification_method, BernoulliNB()))
	elif classification_method == "SVM":
		pipeline.append((classification_method, LinearSVC()))
	#elif classification_method == "Decision Tree":
	#	pipeline.append((classification_method, DecisionTreeClassifier()))
	#	classifier = SklearnClassifier(Pipeline(pipeline),sparse=False)
	elif classification_method == "KNN":
		pipeline.append((classification_method, KNeighborsClassifier()))
	elif classification_method == "Rocchio":
		pipeline.append((classification_method, NearestCentroid()))
	elif classification_method == "Perceptron":
		pipeline.append((classification_method, Perceptron()))

	return SklearnClassifier(Pipeline(pipeline))


def getResults(classifier,test_set,precisions,recalls):
	test = []
	truth = []
	for (feat,cat) in test_set:
		test.append(feat)
		truth.append(cat)
	categories = classifier.batch_classify(test)

	precision = metrics.precision_score(truth,categories)
	recall = metrics.recall_score(truth,categories)
	fmeasure = 2.0 * (precision*recall) / (precision+recall)

	precisions.append(precision)
	recalls.append(recall)

	print '\tPrecision =', precision
	print '\tRecall =', recall
	print '\tF-measure =', fmeasure, '\n'

################################################################################



################################################################################
# Main
################################################################################

print 'Loading docs...'
train_docs = []
test_docs = []
for fileid in reuters.fileids():
	if fileid.startswith('training'):
		train_docs.append(fileid)
	if fileid.startswith('test'):
		test_docs.append(fileid)

print 'Building training set...'
training_set = []
buildSet(training_set,train_docs)

print 'Building test set...'
test_set = []
buildSet(test_set,test_docs)

precisions_tfidf=[]
recalls_tfidf=[]
precisions_chi2=[]
recalls_chi2=[]


print '\n################################################################################'
print '# TF-IDF'
print '################################################################################\n'

print 'Training the Multinomial Naive Bayes classifier...'
classifier = classify("tf-idf","M-NB")
classifier.train(training_set)
getResults(classifier,test_set,precisions_tfidf,recalls_tfidf)

print 'Training the Bernoulli Naive Bayes classifier...'
classifier = classify("tf-idf","B-NB")
classifier.train(training_set)
getResults(classifier,test_set,precisions_tfidf,recalls_tfidf)

print 'Training the SVM classifier...'
classifier = classify("tf-idf","SVM")
classifier.train(training_set)
getResults(classifier,test_set,precisions_tfidf,recalls_tfidf)

print 'Training the K-nearest Neighbors classifier...'
classifier = classify("tf-idf","KNN")
classifier.train(training_set)
getResults(classifier,test_set,precisions_tfidf,recalls_tfidf)

print 'Training the Rocchio\'s classifier...'
classifier = classify("tf-idf","Rocchio")
classifier.train(training_set)
getResults(classifier,test_set,precisions_tfidf,recalls_tfidf)

print 'Training the Perceptron classifier...'
classifier = classify("tf-idf","Perceptron")
classifier.train(training_set)
getResults(classifier,test_set,precisions_tfidf,recalls_tfidf)

# print 'Training the Decision Tree classifier...'
# classifier = classify("tf-idf","Decision Tree")
# classifier.train(training_set)
# getResults(classifier,test_set)


print '\n################################################################################'
print '# Chi squared'
print '################################################################################\n'

print 'Training the Multinomial Naive Bayes classifier...'
classifier = classify("chi2","M-NB")
classifier.train(training_set)
getResults(classifier,test_set,precisions_chi2,recalls_chi2)

print 'Training the Bernoulli Naive Bayes classifier...'
classifier = classify("chi2","B-NB")
classifier.train(training_set)
getResults(classifier,test_set,precisions_chi2,recalls_chi2)

print 'Training the SVM classifier...'
classifier = classify("chi2","SVM")
classifier.train(training_set)
getResults(classifier,test_set,precisions_chi2,recalls_chi2)

print 'Training the K-nearest Neighbors classifier...'
classifier = classify("chi2","KNN")
classifier.train(training_set)
getResults(classifier,test_set,precisions_chi2,recalls_chi2)

print 'Training the Rocchio\'s classifier...'
classifier = classify("chi2","Rocchio")
classifier.train(training_set)
getResults(classifier,test_set,precisions_chi2,recalls_chi2)

print 'Training the Perceptron classifier...'
classifier = classify("chi2","Perceptron")
classifier.train(training_set)
getResults(classifier,test_set,precisions_chi2,recalls_chi2)


print 'Displaying plot...'
plt.plot(recalls_tfidf,precisions_tfidf,'ro',label='Tf-idf')
plt.plot(recalls_chi2,precisions_chi2,'b^',label='Chi2')
plt.ylabel('Precision')
plt.xlabel('Recall')
methods=["M-NB","B-NB","SVM","KNN","Rocchio","Perceptron"]
for i in range(len(precisions_tfidf)):
	plt.text(recalls_tfidf[i],precisions_tfidf[i],methods[i])
for i in range(len(precisions_chi2)):
	plt.text(recalls_chi2[i],precisions_chi2[i],methods[i])
plt.legend(loc='upper left')
plt.show()

################################################################################