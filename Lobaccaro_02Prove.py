# David Lobaccaro
# 02 Prove

import heapq as h
from sklearn import datasets
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class kNNModel:
	def __init__(self, k, data_train, targets_train):
		self.k = k
		self.data_train = data_train
		self.targets_train = targets_train

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
			targets.append(max(set(neighbors_classes), key=neighbors_classes.count))

		return targets

	def calc_distance_single(self, input_item, stored_item):
		distance = 0
		for input_attr, stored_attr in zip(input_item, stored_item):
			distance += (input_attr - stored_attr) ** 2
		return distance

class kNNClassifier:
	def __init__(self, k):
		self.k = k

	def fit(self, data_train, targets_train):
		return kNNModel(self.k, data_train, targets_train)

def prepare_data(data, targets, test_size): 
	return train_test_split(pre.StandardScaler().fit_transform(data), targets, test_size=test_size)

def calc_print_results(targets_predicted, targets_test):
	passed = 0
	for predicted, target in zip(targets_predicted, targets_test):
		if predicted == target:
			passed += 1
	percent_correct = passed / len(targets_predicted) * 100.0
	print("Percent correct: " + str(percent_correct))

def get_test_size():
	return float(input("Enter float for test size: "))

def get_k(phrase):
	return int(input(phrase))

def main():
	dataset = datasets.load_iris()
	test_size = get_test_size()

	# kNN My Own Implementation
	own_k = get_k("Enter int for k for own implementation: ")
	data_train, data_test, targets_train, targets_test = prepare_data(dataset.data, dataset.target, test_size)
	classifier = kNNClassifier(own_k)
	model = classifier.fit(data_train, targets_train)
	targets_predicted = model.predict(data_test)
	calc_print_results(targets_predicted, targets_test)

	# kNN Library Implementation
	lib_k = get_k("Enter int for k for library implementation: ")
	classifier2 = KNeighborsClassifier(n_neighbors=lib_k)
	model2 = classifier2.fit(data_train, targets_train)
	predictions = model2.predict(data_test)
	calc_print_results(predictions, targets_test)

if __name__ == "__main__":
	main()