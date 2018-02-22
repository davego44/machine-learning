# David Lobaccaro
# 06 Prove

import random
import numpy as np
import pandas as pd
from math import exp
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split

class Node:
	def __init__(self, numNodesPrevLayer):
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
			activation = self.sigmoid(node_sum)
			values.append(activation)
		values.append(-1);
		return values

	def sigmoid(self, n):
		return 1 / (1 + exp(-1 * n))


class NeuralNetwork:
	def __init__(self, numAttrNodes, layerDef):
		self.layers = []
		self.initLayers(numAttrNodes, layerDef)

	def initLayers(self, numAttrNodes, layerDef):
		prevLayerNodes = numAttrNodes
		for i in range(layerDef[0]):
			self.layers.append(Layer(layerDef[i + 1], prevLayerNodes))
			prevLayerNodes = layerDef[i + 1]

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
		output = []
		for item in data_test:
			output.append(self.network.calcValues(item))
		return output


class NNClassifier:
	def __init__(self, layerDef):
		self.layerDef = layerDef

	def fit(self, data_train, targets_train):
		numAttrNodes = len(data_train[0])
		network = NeuralNetwork(numAttrNodes, self.layerDef)
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
	classifer = NNClassifier([2, 3, 3])
	model = classifer.fit(data_train, targets_train)
	predicted = model.predict(data_test)
	print(predicted)
	print("-------------")
	count = 0
	for pred, tar in zip(predicted, targets_test):
		value = ""
		if (pred[0] > 0.5):
			value = "Iris-versicolor"
		elif (pred[1] > 0.5):
			value = "Iris-setosa"
		elif (pred[2] > 0.5):
			value = "Iris-virginica"
		else:
			value = "Iris-virginica"
		if (value == tar):
			count += 1
	print("Accuracy: " + str(count / len(predicted) * 100))
	#data_train, data_test, targets_train, targets_test = train_test_split(dataP, targetsP, test_size=0.3)
	#model = classifer.fit(data_train, targets_train)
	#predicted = model.predict(data_test)
	#print("----------------------")
	#print(predicted)

if __name__ == "__main__":
	main()