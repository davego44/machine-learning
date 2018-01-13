# David Lobaccaro
# 01 Prove

import sys
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

class HardCodedModel:
	def __init__(self):
		pass

	def predict(self, data_test):
		targets = []

		for item in data_test:
			targets.append(self.predict_single(item))

		return targets

	def predict_single(self, item):
		return 0

class HardCodedClassifier:
	def __init__(self):
		pass

	def fit(self, data_train, targets_train):
		return HardCodedModel()

def prepare_data(data, targets, test_size): 
	return train_test_split(data, targets, test_size=test_size)

def get_dataset():
	print("0 - Iris")
	print("1 - Digits")
	print("2 - Wine")
	print("3 - Breast Cancer")
	number = int(input("Select dataset by number: "))
	if number == 1:
		return datasets.load_digits()
	elif number == 2:
		return datasets.load_wine()
	elif number == 3:
		return datasets.load_breast_cancer()
	else:
		return datasets.load_iris()

def get_test_size():
	return float(input("Enter float for test size: "))

def get_classifier():
	print("0 - GaussianNB")
	print("1 - HardCoded")
	number = int(input("Select which algorithm to use by number: "))
	if number == 1:
		return HardCodedClassifier()
	else:
		return GaussianNB()

def calc_print_results(targets_predicted, targets_test):
	passed = 0
	for predicted, target in zip(targets_predicted, targets_test):
		if predicted == target:
			passed += 1
	percent_correct = passed / len(targets_predicted) * 100.0
	print("Percent correct: " + str(percent_correct))

def main():
	dataset = get_dataset()
	test_size = get_test_size()
	data_train, data_test, targets_train, targets_test = prepare_data(dataset.data, dataset.target, test_size)
	print("Data prepared.")
	classifier = get_classifier()
	model = classifier.fit(data_train, targets_train)
	print("Trained.")
	targets_predicted = model.predict(data_test)
	print("Predicted results in.")
	print("Predicted:")
	print(targets_predicted)
	print("Actual:")
	print(targets_test)
	calc_print_results(targets_predicted, targets_test)

if __name__ == "__main__":
	main()