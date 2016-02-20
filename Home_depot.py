# This code will solve the problem for the home depot search relevance, from the Kaggle competition. The problem is to take an item ID, and its description and predict the search relevance. The data contains the item's ID, a product uid a title and the relevance. The way to go about this is very similar to the Bag_of_words competition, and therefore the code will be heavily inspired by that competition.

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error

def train_data_import(data = 'train_data.csv'):
	return pd.read_csv(data, delimiter=',', header=0, quoting=0)

def description_data_import(data = 'product_descriptions.csv'):
	return pd.read_csv(data, delimiter=',', header=0, quoting=0)

def training_target_format( raw_train, raw_description ):	#picks the training and target data out of raw data.
	format_train = []
	format_target = []
	dic = {raw_description['product_uid'][i] : raw_description['product_description'][i] for i in xrange(len(raw_description))}	#converts a list into a dictionary, can look up the product_uid.
	for i in xrange(len(raw_train)):
		format_train.append(','.join((raw_train['product_title'][i],raw_train['search_term'][i], dic[raw_train['product_uid'][i]])))
		format_target.append(int(round(raw_train['relevance'][i])))
	return format_train, format_target

# The following function takes in the raw review, and outputs the cleaned review
def clean_product( raw_product ):		# Removes the HTML, punctuation, numbers, stopwords...
	rm_html = BeautifulSoup(raw_product).get_text()	# removes html
	letters_only = re.sub("[^0-9a-zA-Z]",           	# The pattern to search for; ^ means NOT
                   		  " ",                   	# The pattern to replace it with
                          rm_html )              	# The text to search
	lower_case = letters_only.lower()	         	# Convert to lower case
	words = lower_case.split()          	     	# Split into words
	stops = stopwords.words("english")
	stops.append('ve')
	stops = set(stops)
#	english_words = words.words()[1:100]
	meaningful_words = [w for w in words if not w in stops]	# Remove stop words from "words"
	return ' '.join(meaningful_words)			# Joins the words back together separated by a space


# The following function iterates clean_review over all reviews in the set
def clean_all_products( format_train ):
	cleaned_products = []
	for i in xrange(len(format_train)):
		cleaned_products.append(clean_product(format_train[i]))
		if ( (i+1) % 1000 == 0 ):
			print(' -- Clean product # %d' % i)
	return cleaned_products

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
def bag_of_words(cleaned_products, n_features = 300):
	vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,    	# Allows to tokenize
                             preprocessor = None, 	# Allows to do some preprocessing
                             stop_words = None,   	# We could remove stopwords from here
                             max_features = n_features) 	# Chooses a given number of words, just a subset of the huge total number of words.
	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	data_features = vectorizer.fit_transform(cleaned_products)
	data_features = data_features.toarray()
	return data_features, vectorizer

def Trainer():
	raw_description, raw_train = description_data_import()[:100], train_data_import()[:100]
	print '- Description and training data imported'
	format_train, format_target = training_target_format( raw_train, raw_description )
	print '- training and target data formatted'
	clean_train = clean_all_products( format_train )
	print '- Training data cleaned'
#	print clean_train[0]
	train_data_features, vectorizer = bag_of_words(clean_train)
	print '- Bag-of-words with %d prodcuts created' % len(raw_train['product_uid'])
#	print train_data_features[0]
	start = time.time()
	print '- Trains a classifier'
	X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(train_data_features, format_target, train_size = 0.9, random_state=10)
	clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=3, max_features='auto', bootstrap=True, oob_score=False, n_jobs=4, random_state=None, verbose=0, min_density=None, compute_importances=None)
	forest = clf.fit( X_train, Y_train )
	score = clf.score(X_valid, Y_valid)
	RMSE = mean_squared_error(clf.predict(X_valid), Y_valid)**0.5 #This is the scoring we ought to use.
	end = time.time()
	T = end - start
	print '- Finished training after %f seconds, with score = %f' % (T,score)
	print '- RMSE score = %f' % RMSE
	return forest, vectorizer, score, RMSE

# Write a log file of results obtained thus far
def log_file(clf, classifier, score, RMSE):
	with open('Home_depot_log_file', 'a') as f:
		f.write('\n\n')
		f.write('- Classifier: %s \n' % classifier)
		f.write(' -- Score = %f , \n' % score)
		f.write(' -- RMSE = %f , \n' % RMSE)
		f.write(' -- Parameters: %s, \n' % clf.get_params(True))

def main_function():
	clf, vectorizer, score, RMSE = Trainer()
#	log_file(clf, 'RandomForest', score, RMSE)
#	Trainer()
	
main_function()

# Without description data, and n_estimators = 100: Typical score ~ 0.5, which is just as good as random guesses, my RMSE score is approximately 0.76, which is just above the benchmark code.

# With description data, and n_estimators = 200: Typical score ~ 0.5, which is just as good as random guesses, my RMSE score is approximately 0.76, which is just above the benchmark code. I think something is not correct here, maybe I do not do what I think it is supposed to do.




