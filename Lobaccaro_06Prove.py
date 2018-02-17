# David Lobaccaro
# 06 Prove

import random
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split

class Node:
	def __init__(self, numNodesPrevLayer):
		self.threshold = 0
		self.weights = []
		self.initWeights(numNodesPrevLayer)

	def initWeights(self, numNodesPrevLayer):
		for _ in range(numNodesPrevLayer + 1):
			weight = random.random() 
			neg = random.randint(0, 1) * -1
			if (neg != 0):
				weight *= neg
			self.weights.append(weight)


class Layer:
	def __init__(self, numNodesLayer, numNodesPrevLayer):
		self.nodes = []
		self.initNodes(numNodesLayer, numNodesPrevLayer)

	def initNodes(self, numNodesLayer, numNodesPrevLayer):
		for _ in range(numNodesLayer):
			self.nodes.append(Node(numNodesPrevLayer))

	def calcValues(self, prevValues):
		values = []
		for node in self.nodes:
			node_sum = 0
			for i, prevValue in enumerate(prevValues):
				node_sum += prevValue * node.weights[i]
			if (node_sum >= node.threshold):
				values.append(1)
			else:
				values.append(0)
		values.append(-1);
		return values


class NeuralNetwork:
	def __init__(self, numAttrNodes, numHiddenNodes, numOutputNodes):
		self.layers = []
		self.initLayers(numAttrNodes, numHiddenNodes, numOutputNodes)

	def initLayers(self, numAttrNodes, numHiddenNodes, numOutputNodes):
		self.layers.append(Layer(numHiddenNodes, numAttrNodes))
		self.layers.append(Layer(numOutputNodes, numHiddenNodes))

	def calcValues(self, attrValues):
		results = attrValues
		for layer in self.layers:
			results = layer.calcValues(results)
		del results[-1]
		return results

	def train(self, data_train, targets_train):
		pass


class NNModel:
	def __init__(self, network):
		self.network = network

	def predict(self, data_test):
		one_item = data_test[0]
		return self.network.calcValues(one_item)


class NNClassifier:
	def __init__(self, numHiddenNodes):
		self.numHiddenNodes = numHiddenNodes

	def fit(self, data_train, targets_train):
		numAttrNodes = len(data_train[0])
		numOutputNodes = len(np.unique(targets_train))
		network = NeuralNetwork(numAttrNodes, self.numHiddenNodes, numOutputNodes)
		network.train(data_train, targets_train)
		return NNModel(network)



def prepare_iris_data():
	headers = ["slen", "swid", "plen", "pwid", "class"]
	df = pd.read_csv('data/iris.data', header=None, names=headers)
	return np.array(df.drop("class", axis=1).values), np.array(df["class"].values)

def read_prepare_pima_data():
	headers = ["times","plasma","diastolic","skin","serum","bmi","pedigree","age","class"]
	zero_data_columns = ["plasma","diastolic","skin","serum","bmi","pedigree","age"]
	dataset = pd.read_csv('data/pima-indians-diabetes.data', header=None, names=headers)
	dataset[list(zero_data_columns)] = dataset[list(zero_data_columns)].replace(0, np.NaN)
	dataset = dataset[pd.notnull(dataset["plasma"])]
	dataset = dataset[pd.notnull(dataset["diastolic"])]
	dataset = dataset[pd.notnull(dataset["bmi"])]
	dataset.fillna(dataset.mean(), inplace=True)
	return np.array(dataset.drop("class", axis=1).values), np.array(dataset["class"].values)

def main():
	dataI, targetsI = prepare_iris_data()
	dataP, targetsP = read_prepare_pima_data()
	dataI = pre.StandardScaler().fit_transform(dataI)
	dataP = pre.StandardScaler().fit_transform(dataP)
	data_train, data_test, targets_train, targets_test = train_test_split(dataI, targetsI, test_size=0.3)
	classifer = NNClassifier(2)
	model = classifer.fit(data_train, targets_train)
	predicted = model.predict(data_test)
	print(predicted)
	data_train, data_test, targets_train, targets_test = train_test_split(dataP, targetsP, test_size=0.3)
	model = classifer.fit(data_train, targets_train)
	predicted = model.predict(data_test)
	print(predicted)

if __name__ == "__main__":
	main()