# David Lobaccaro
# 03 Prove

import heapq as h
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

class kNNModel:
	def __init__(self, k, data_train, targets_train, reg):
		self.k = k
		self.data_train = data_train
		self.targets_train = targets_train
		self.reg = reg

	def predict(self, data_test):
		targets = []

		dist_item_list = [] # list for the minHeap
		for input_item in data_test:
			for stored_item, stored_item_class in zip(self.data_train, self.targets_train):
				distance = self.calc_distance_single(input_item, stored_item)
				h.heappush(dist_item_list, (distance, stored_item.tolist(), stored_item_class))
			neighbors_classes = []
			for i in range(0, self.k):
				neighbors_classes.append(h.heappop(dist_item_list)[2])
			if self.reg:
				average = 0
				for neighbor in neighbors_classes:
					average += neighbor
				average /= len(neighbors_classes)
				targets.append(average)
			else:
				targets.append(max(set(neighbors_classes), key=neighbors_classes.count))

		return targets

	def calc_distance_single(self, input_item, stored_item):
		distance = 0
		for input_attr, stored_attr in zip(input_item, stored_item):
			distance += (input_attr - stored_attr) ** 2
		return distance

class kNNClassifier:
	def __init__(self, k, reg):
		self.k = k
		self.reg = reg

	def fit(self, data_train, targets_train):
		return kNNModel(self.k, data_train, targets_train, self.reg)

def calc_results(targets_predicted, targets_test, reg):
	passed = 0
	for predicted, target in zip(targets_predicted, targets_test):
		if reg:
			if target - 1 <= predicted <= target + 1:
				passed += 1
		else:
			if predicted == target:
				passed += 1
	percent_correct = passed / len(targets_predicted) * 100.0
	return percent_correct

def read_prepare_car_data(pre_option):
	# No missing data
	# Every attribute is directly relatable to distance (ex: vhigh is further from low than
		# high) and therefore I will just replace with direct number scale
	# Normalize data
	headers= ["buying","maint","doors","persons","lug_boot","safety","class"]
	dataset = pd.read_csv('data/car.data', header=None, names=headers)
	replacements = {
		"buying"   : {"low"   : 0, "med" : 1, "high" : 2, "vhigh" : 3},
		"maint"    : {"low"   : 0, "med" : 1, "high" : 2, "vhigh" : 3},
		"doors"	   : {"5more" : 5},
		"persons"  : {"more"  : 5},
		"lug_boot" : {"small" : 0, "med" : 1, "big"  : 2},
		"safety"   : {"low"   : 0, "med" : 1, "high" : 2},
		"class"    : {"unacc" : 0, "acc" : 1, "good" : 2, "vgood" : 3}
	}
	dataset.replace(replacements, inplace=True)
	dataset = dataset.astype(float)
	return get_pre_data(dataset.drop("class", axis=1).values, pre_option), np.array(dataset["class"].values), False

def read_prepare_pima_data(option, pre_option):
	# Assume "times" column does not contain missing data
	# Remove all rows with 0s (converted to NaN) in the "plasma", "diastolic", and "bmi" columns
		# due to the small number of NaN in those columns
	# Fill in remaining NaNs with the mean
	# Better percent without normalizing the data
	headers = ["times","plasma","diastolic","skin","serum","bmi","pedigree","age","class"]
	zero_data_columns = ["plasma","diastolic","skin","serum","bmi","pedigree","age"]
	dataset = pd.read_csv('data/pima-indians-diabetes.data', header=None, names=headers)
	dataset[list(zero_data_columns)] = dataset[list(zero_data_columns)].replace(0, np.NaN)
	if option == 0:
		dataset.dropna(inplace=True)
	elif option == 1:
		dataset.fillna(dataset.mean(), inplace=True)
	else:
		dataset = dataset[pd.notnull(dataset["plasma"])]
		dataset = dataset[pd.notnull(dataset["diastolic"])]
		dataset = dataset[pd.notnull(dataset["bmi"])]
		dataset.fillna(dataset.mean(), inplace=True)
	return get_pre_data(dataset.drop("class", axis=1).values, pre_option), np.array(dataset["class"].values), False

def read_prepare_mpg_data(option, pre_option):
	# Remove all rows with missing values becaue there are only 6
	# Remove name column because every value is unique and adds no value
	# For all multi-valued discrete columns, just add columns for values
	# Use MinMaxScaler (better results than StandardScaler)
	headers = ["mpg","cyl","displ","horse","weight","acc","year","origin","name"]
	dataset = pd.read_csv('data/auto-mpg.data', header=None, names=headers, delim_whitespace=True, na_values=["?"])
	dataset.dropna(inplace=True)
	del dataset["name"]
	if option == 1:
		dataset = pd.get_dummies(dataset, columns=["cyl","year","origin"])
	return get_pre_data(dataset.drop("mpg", axis=1).values, pre_option), np.array(dataset["mpg"].values), True

def get_pre_data(data_values, pre_option):
	if pre_option == 0:
		return pre.MinMaxScaler().fit_transform(np.array(data_values))
	elif pre_option == 1:
		return pre.StandardScaler().fit_transform(np.array(data_values))
	else:
		return np.array(data_values)s

def cross_validation(n_splits, X, y, k, reg=False):
	# Only give the option to use library kNN if not doing regression
	if not reg:
		print("kNN")
		print("0 - Use my implemented kNN")
		print("1 - Use library kNN")
		knn_option = int(input("Select by number: "))
	else:
		knn_option = 0
	average_correct = 0
	# Use KFold library to loop n_splits times
	kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
	for train_index, test_index in kf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		classifier = None
		if knn_option == 0:
			classifier = kNNClassifier(k, reg)
		else:
			classifier = KNeighborsClassifier(n_neighbors=k)
		model = classifier.fit(X_train, y_train)
		targets_predicted = model.predict(X_test)
		average_correct += calc_results(targets_predicted, y_test, reg)
	average_correct /= n_splits
	return average_correct

def get_pre_option():
	print("Preprocessing Options")
	print("0 - Use MinMaxScaler")
	print("1 - Use StandardScaler")
	print("2 - Use no preprocessing")
	pre_option = int(input("Select option by number: "))
	if pre_option != 0 or pre_option != 1:
		pre_option = 2
	return pre_option

def get_dataset():
	print("Dataset Options")
	print("0 - Car")
	print("1 - Prima")
	print("2 - Mpg")
	number = int(input("Select dataset by number: "))
	if number == 1:
		print("Basic Options")
		print("0 - Remove all missing data entirely")
		print("1 - Replace all missing data with mean")
		print("2 - Use both methods as best as possible")
		option = int(input("Select option by number: "))
		if option != 0 or option != 1:
			option = 2
		return read_prepare_pima_data(option, get_pre_option())
	elif number == 2:
		print("Basic Options")
		print("0 - Keep multi-valued discrete values the same")
		print("1 - Turn multi-valued discrete values into columns")
		option = int(input("Select option by number: "))
		if option != 0:
			option = 1
		return read_prepare_mpg_data(option, get_pre_option())
	else:
		print("No basic options available for car data...")
		return read_prepare_car_data(get_pre_option())

def get_splits():
	return int(input("Enter the number of splits for KFold: "))

def get_k():
	return int(input("Enter the k for neighbors: "))

def main():
	X, y, reg = get_dataset()
	average_correct = cross_validation(get_splits(), X, y, get_k(), reg)
	print("Average correct: " + str(average_correct))

if __name__ == "__main__":
	main()