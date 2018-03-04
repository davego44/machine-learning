# David Lobaccaro
# 08 Prove

import random
import numpy as np
import pandas as pd
from math import exp
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split

class Node:
	def __init__(self):
		self.activation = 0
		self.error = 0
		self.prevConnections = []
		self.postConnections = []


class Connection:
	def __init__(self):
		self.weight = self.genWeight()
		self.prevNode = None
		self.postNode = None
		self.prevDelta = 0

	def genWeight(self):
		weight = random.random() 
		neg = random.randint(0, 1) * -1
		if (neg != 0):
			weight *= neg
		return weight


class Layer:
	def __init__(self):
		self.nodes = []


class NeuralNetwork:
	def __init__(self, layerDef):
		self.layers = []
		self.initLayersAndNodes(layerDef)
		self.initConnections(layerDef)

	def initLayersAndNodes(self, layerDef):
		#loop through each layer
		for layerIndex in range(len(layerDef)):
			layer = Layer()
			numOfNodes = layerDef[layerIndex]
			#if the layer is not the last layer, add a bias node
			if layerIndex + 1 != len(layerDef):
				numOfNodes += 1
			#loop through the number of nodes and add nodes to the layer
			for nodeIndex in range(numOfNodes):
				node = Node()
				#if this is the last node (and not last layer), set the activation to -1 (bias node)
				if nodeIndex + 1 == numOfNodes and layerIndex + 1 != len(layerDef):
					node.activation = -1
				layer.nodes.append(node)
			self.layers.append(layer)

	def initConnections(self, layerDef):
		#loop through all layers except the last one
		for layerIndex in range(len(layerDef) - 1):
			currentLayer = self.layers[layerIndex]
			nextLayer = self.layers[layerIndex + 1]
			#loop through all the nodes in the current layer
			for currentNodeIndex in range(len(currentLayer.nodes)):
				currentNode = currentLayer.nodes[currentNodeIndex]
				#loop through all the nodes in the next layer
				for nextNodeIndex in range(len(nextLayer.nodes)):
					nextNode = nextLayer.nodes[nextNodeIndex]
					#setup the connection between currentNode and nextNode
					conn = Connection()
					conn.prevNode = currentNode
					conn.postNode = nextNode
					currentNode.postConnections.append(conn)
					nextNode.prevConnections.append(conn)

	def feedForward(self, attrValues):
		inputLayer = self.layers[0]
		#loop through all nodes in the input layer except the last (bias) and set the input nodes
		for i in range(len(inputLayer.nodes) - 1):
			node = inputLayer.nodes[i]
			value = attrValues[i]
			node.activation = value
		#input nodes set, now run the feed forward algorithm
		output = []
		#loop through each layer
		for layerIndex in range(len(self.layers)):
			#skip the first layer
			if layerIndex != 0:
				#loop through each node in the current layer
				currentLayer = self.layers[layerIndex]
				for currentNodeIndex in range(len(currentLayer.nodes)):
					#if this is not the last layer and last node, skip it (bias)
					if layerIndex + 1 != len(self.layers) and currentNodeIndex + 1 == len(currentLayer.nodes):
						continue
					currentNode = currentLayer.nodes[currentNodeIndex]
					currentNodeActivation = 0
					#loop through each previous connection of the current node
					for connection in currentNode.prevConnections:
						currentNodeActivation += connection.prevNode.activation * connection.weight
					currentNodeActivation = self.sigmoid(currentNodeActivation)
					currentNode.activation = currentNodeActivation
					#if this is the last layer, append activation to ouput
					if layerIndex + 1 == len(self.layers):
						output.append(currentNodeActivation)
		return output

	def sigmoid(self, n):
		return 1 / (1 + exp(-1 * n))

	def feedBackward(self, targets):
		#update last layer
		for i in range(len(self.layers[len(self.layers) - 1].nodes)):
			node = self.layers[len(self.layers) - 1].nodes[i]
			target = targets[i]
			error = node.activation * (1 - node.activation) * (node.activation - target)
			node.error = error
		#update other layers from post to prev (skip last layer)
		for i in range(len(self.layers) - 1):
			layerIndex = len(self.layers) - 2 - i
			#skip first layer as well
			if layerIndex != 0:
				for node in self.layers[layerIndex].nodes:
					weightSum = 0
					for conn in node.postConnections:
						weightSum += conn.postNode.error * conn.weight
					error = node.activation * (1 - node.activation) * weightSum
					node.error = error

	def update(self, learnRate, momentumConstant):
		#loop through each layer from last to first
		for i in range(len(self.layers)):
			layerIndex = len(self.layers) - 1 - i
			#skip first layer
			if layerIndex != 0:
				for node in self.layers[layerIndex].nodes:
					for conn in node.prevConnections:
						delta = learnRate * node.error * conn.prevNode.activation + (conn.prevDelta * momentumConstant) 
						conn.weight -= delta
						conn.prevDelta = delta

	def train(self, data_train, targets_train, targets_map, learnRate, momentumConstant):
		for item, target in zip(data_train, targets_train):
			self.feedForward(item)
			self.feedBackward(targets_map[target])
			self.update(learnRate, momentumConstant)

	def display(self):
		for layer in self.layers:
			nodeAct = []
			weights = []
			nodeErrors = []
			for node in layer.nodes:
				nodeAct.append(node.activation)
				nodeErrors.append(node.error)
				if node.postConnections is not None:
					for conn in node.postConnections:
						weights.append(conn.weight)
			print("Layer Nodes")
			print(nodeAct)
			print("Layer Node Errors")
			print(nodeErrors)
			print("Weights")
			print(weights)


class NNModel:
	def __init__(self, network):
		self.network = network

	def predict(self, data_test):
		output = []
		for item in data_test:
			result = self.network.feedForward(item)
			output.append(result)
		return output

	def train(self, data_train, targets_train, targets_map, learnRate, momentumConstant):
		self.network.train(data_train, targets_train, targets_map, learnRate, momentumConstant)

	def display(self):
		self.network.display()


class NNClassifier:
	def __init__(self, layerDef):
		self.layerDef = layerDef

	def fit(self, data_train, targets_train, targets_map, learnRate, momentumConstant):
		numAttrNodes = len(data_train[0])
		self.layerDef.insert(0, numAttrNodes)
		network = NeuralNetwork(self.layerDef)
		network.train(data_train, targets_train, targets_map, learnRate, momentumConstant)
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

def accuracy_iris(predicted, targets_test):
	count = 0
	for pred, tar in zip(predicted, targets_test):
		value = ""
		if (pred[0] > pred[1] and pred[0] > pred[2]):
			value = "Iris-virginica"
		elif (pred[1] > pred[0] and pred[1] > pred[2]):
			value = "Iris-versicolor"
		elif (pred[2] > pred[0] and pred[2] > pred[1]):
			value = "Iris-setosa"
		if value == tar:
			count += 1
	print("Iris Accuracy: " + str(count / len(predicted) * 100))

def accuracy_prima(predicted, targets_test):
	count = 0
	for pred, tar in zip(predicted, targets_test):
		value = 0
		if (pred[0] > pred[1]):
			value = 1
		else:
			value = 0
		if value == tar:
			count += 1
	print("Prima Accuracy: " + str(count / len(predicted) * 100))

def main():
	dataI, targetsI = prepare_iris_data()
	dataP, targetsP = read_prepare_pima_data()
	dataI = pre.StandardScaler().fit_transform(dataI)
	dataP = pre.StandardScaler().fit_transform(dataP)
	data_train, data_test, targets_train, targets_test = train_test_split(dataI, targetsI, test_size=0.3)
	classifer = NNClassifier([3, 3])
	targets_map = {"Iris-virginica" : [1, 0, 0], "Iris-versicolor" : [0, 1, 0], "Iris-setosa" : [0, 0, 1]}
	learnRate = 0.3
	momentumConstant = 0.9
	model = classifer.fit(data_train, targets_train, targets_map, learnRate, momentumConstant)
	predicted = model.predict(data_test)
	accuracy_iris(predicted, targets_test)
	for _ in range(80):
		shuffler = np.arange(data_train.shape[0])
		data_train = data_train[shuffler]
		targets_train = targets_train[shuffler]
		model.train(data_train, targets_train, targets_map, learnRate, momentumConstant)
		predicted = model.predict(data_test)
		accuracy_iris(predicted, targets_test)

	print("------------")
	print("Prima")
	print("------------")
	
	data_train, data_test, targets_train, targets_test = train_test_split(dataP, targetsP, test_size=0.3)
	classifer = NNClassifier([2, 2])
	targets_map = {0 : [0, 1], 1 : [1, 0]}
	learnRate = 0.3
	model = classifer.fit(data_train, targets_train, targets_map, learnRate, momentumConstant)
	predicted = model.predict(data_test)
	accuracy_prima(predicted, targets_test)
	for _ in range(50):
		shuffler = np.arange(data_train.shape[0])
		data_train = data_train[shuffler]
		targets_train = targets_train[shuffler]
		model.train(data_train, targets_train, targets_map, learnRate, momentumConstant)
		predicted = model.predict(data_test)
		accuracy_prima(predicted, targets_test)

if __name__ == "__main__":
	main()