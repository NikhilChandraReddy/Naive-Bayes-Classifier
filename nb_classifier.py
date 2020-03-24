import sys
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi


# Load the dataset into the List of Lists data structure for furhter processing
def load(filename):
	data = list()
	with open(filename, 'r') as f:
		f_reader = reader(f)
		for r in f_reader:
			if not r:
				continue
			data.append(r)
	return data

# Convert the String Data to Float in the data (List of Lists)
def floating_column(data, col_number):
	for row in data:
		row[col_number] = float(row[col_number].strip())

# To enumerate the dataset coloms containing string names and to integers 
def enumerate_column(data, col_number):
	val = [row[col_number] for row in data]
	enum = set(val)
	lookup = dict()
	for i, value in enumerate(enum):
		lookup[value] = i
	for row in data:
		row[col_number] = lookup[row[col_number]]
	return lookup

# Split a dataset into k folds as training and test sets
def Kfold_CVS(data, k_folds):
	data_split = list()
	data_copy = list(data)
	fold_size = int(len(data) / k_folds)
	for _ in range(k_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split

# Calculate accuracy percentage
def percentage(x, y):
	result = 0
	for i in range(len(x)):
		if x[i] == y[i]:
			result += 1
	return result / float(len(x)) * 100.0

# Evaluate an algorithm using a Kfold cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
	folds = Kfold_CVS(data, n_folds)
	kfolds_scores = list()
	for f in folds:
		train_set = list(folds)
		train_set.remove(f)
		train_set = sum(train_set, [])
		test_set = list()
		for row in f:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		y= algorithm(train_set, test_set, *args)
		x = [row[-1] for row in f]
		accuracy = percentage(x, y)
		kfolds_scores.append(accuracy)
	return kfolds_scores

# Split the dataset by class values, returns a dictionary
def separate_by_class(data):
	separated = dict()
	for i in range(len(data)):
		each_row = data[i]
		each_class = each_row[-1]
		if (each_class not in separated):
			separated[each_class] = list()
		separated[each_class].append(each_row)
	return separated

# Returns Mean of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# returns Standard Deviation of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_data(data):
	summaries = [(mean(col), stdev(col), len(col)) for col in zip(*data)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_data(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def gaussian_probability(x, mean, stdev):
	if(stdev==0):
		stdev=0.1
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= gaussian_probability(row[i], mean, stdev)
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)


def main():
	if len(sys.argv) < 3:
	    print ("\n\n\nNot in correct format\n\nEnter in this format\npython nb_classifier.py <dataset filename> <no of kflods>\nLike \n\tpython nb_classifier.py hayes-roth.data 10\n\tpython nb_classifier.py car.data 10\n\tpython nb_classifier.py breast-cancer.data 10\n")

	elif(sys.argv[1]=="hayes-roth.data" or sys.argv[1]=="car.data" or sys.argv[1]=="breast-cancer.data" ):
		filename = sys.argv[1]
		k_folds = int(sys.argv[2].strip())
		dataset = load(filename)
		if(filename =="hayes-roth.data"):
			for i in range(len(dataset[0])-1):
				floating_column(dataset, i)

		else:
			for i in range(len(dataset[0])-1):
				enumerate_column(dataset, i)

		enumerate_column(dataset, len(dataset[0])-1)
		scores = evaluate_algorithm(dataset, naive_bayes, k_folds)
		print('kfolds_Scores: %s' % scores)
		print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

	else:
		print ("\n\n\nWrong file name or Not in correct format\n\nEnter in this format\npython nb_classifier.py <dataset filename> <no of kflods>\nLike \n\tpython nb_classifier.py hayes-roth.data 10\n\tpython nb_classifier.py car.data 10\n\tpython nb_classifier.py breast-cancer.data 10\n")
		
if __name__== "__main__":
  main()

