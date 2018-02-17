# David Lobaccaro
# 04 Prove

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



class Node:
	def __init__(self, data, value):
		self.data = data
		self.value = value
		self.children = []

	def add_child(self, node):
		self.children.append(node)

	def get_next_child(self, value):
		if self.is_leaf():
			return self
		for child in self.children:
			if child.value == value:
				return child

	def is_leaf(self):
		return not self.children

	def display(self, formating):
		print(formating + self.data + " by " + str(self.value))
		for child in self.children:
			child.display(formating + "\t")



class TreeModel:
	def __init__(self, tree):
		self.tree = tree

	def predict(self, df):
		predicted = []
		for index, row in df.iterrows():
			current_node = self.tree
			while(not current_node.is_leaf()):				
				current_node = current_node.get_next_child(row[current_node.data])
			predicted.append(current_node.data)
		return predicted

	def display(self):
		self.tree.display("")



class TreeClassifier:
	def __init__(self):
		pass

	def fit(self, df, target_col_name):
		return TreeModel(self.create_tree_prep(df, target_col_name))

	def create_tree_prep(self, df, target_col_name):
		col_to_values = {}
		for col in df.drop(target_col_name, axis=1):
			col_to_values[col] = df[col].unique().tolist()
		return self.create_tree(df, target_col_name, col_to_values)

	def create_tree(self, df, target_col_name, col_to_values, value_branched=None):
		if len(df[target_col_name].unique()) == 1:
			return Node(df[target_col_name].value_counts().index[0], value_branched)
		elif len(df.columns) == 1:
			return Node(df[target_col_name].value_counts().index[0], value_branched)
		else:
			lowest_entropy = ["", 100]
			for col in df.drop(target_col_name, axis=1):
				weighted_entropy = self.get_weighted_entropy(df, col, target_col_name)
				if weighted_entropy < lowest_entropy[1]:
					lowest_entropy[0] = col
					lowest_entropy[1] = weighted_entropy
			parent = Node(lowest_entropy[0], value_branched)
			for value in df[lowest_entropy[0]].unique():
				df_ = df[df[lowest_entropy[0]] == value]
				parent.add_child(self.create_tree(df_.drop(lowest_entropy[0], axis=1), target_col_name, col_to_values, value))
			for value in [x for x in col_to_values[lowest_entropy[0]] if x not in df[lowest_entropy[0]].unique().tolist()]:
				parent.add_child(Node(df[target_col_name].value_counts().index[0], value))
			return parent

	def get_weighted_entropy(self, df, col, target_col_name):
		total = len(df[col])
		sum_of_weighted_sum = 0
		for value in df[col].unique():
			df_ = df[df[col] == value]
			counts = df_[target_col_name].value_counts()
			sum = self.sum_entropy(counts)
			weighted_sum = sum * (len(df_[col]) / total)
			sum_of_weighted_sum += weighted_sum
		return sum_of_weighted_sum

	def sum_entropy(self, counts):
		total = counts.sum()
		sum = 0;
		for c in counts:
			sum += self.calc_entropy(c / total)
		return sum

	def calc_entropy(self, n):
		if n != 0:
			return -n * np.log2(n)
		return 0



def prepare_iris_data():
	headers = ["slen", "swid", "plen", "pwid", "class"]
	df = pd.read_csv('data/iris.data', header=None, names=headers)
	df["slen"] = pd.qcut(df["slen"], 4, labels=["small","med","big","vbig"])
	df["swid"] = pd.qcut(df["swid"], 4, labels=["small","med","big","vbig"])
	df["plen"] = pd.qcut(df["plen"], 4, labels=["small","med","big","vbig"])
	df["pwid"] = pd.qcut(df["pwid"], 4, labels=["small","med","big","vbig"])
	return df

def prepare_lense_data():
	headers = ["id", "age", "spec", "astig", "tear", "class"]
	df = pd.read_csv('data/lenses.data', header=None, names=headers, delim_whitespace=True)
	df = df.drop("id", axis=1)
	df["age"] = df["age"].replace([1, 2, 3], ["young", "pre-presb", "presb"])
	df["spec"] = df["spec"].replace([1, 2], ["myope", "hyper"])
	df["astig"] = df["astig"].replace([1, 2], ["no", "yes"])
	df["tear"] = df["tear"].replace([1, 2], ["reduced", "norm"])
	df["class"] = df["class"].replace([1, 2, 3], ["hard", "soft", "none"])
	df = df.astype(object)
	return df

def prepare_voting_data():
	headers = ["class", "handi", "water", "adopt", "phy", "el", "reli", "anti", "aid", "mx", "immi", "syn", "edu", "sup", "crime", "duty", "exp"]
	df = pd.read_csv('data/voting.data', header=None, names=headers, na_values=["?"])
	df.fillna(method="ffill", inplace=True)
	df.fillna(method="bfill", inplace=True) #need both to remove all NaN
	return df

def prepare_crx_data():
	headers = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", "a15", "class"]
	df = pd.read_csv('data/crx.data', header=None, names=headers, na_values=["?"])
	df.fillna(method="ffill", inplace=True)
	df.fillna(method="bfill", inplace=True) #need both to remove all NaN
	df["a2"] = pd.qcut(df["a2"], 4, labels=["small","med","big","vbig"])
	df["a3"] = pd.qcut(df["a3"], 4, labels=["small","med","big","vbig"])
	df["a8"] = pd.qcut(df["a8"], 4, labels=["small","med","big","vbig"])
	df["a14"] = pd.qcut(df["a14"], 4, labels=["small","med","big","vbig"])
	df = df.drop(["a11", "a15"], axis=1) #drop these columns because they are hard to deal with, especially because this data is unknown
	return df

def prepare_king_data(): #115.4s
	headers = ["wkf", "wkr", "wrf", "wrr", "bkf", "bkr", "class"]
	df = pd.read_csv('data/king.data', header=None, names=headers)
	df = df.astype(object)
	return df

def get_data():
	print("0 - iris")
	print("1 - lense")
	print("2 - voting")
	print("3 - crx")
	print("4 - king (~115 seconds)")
	option = int(input("Select option by number: "))
	if option == 1:
		return prepare_lense_data()
	elif option == 2:
		return prepare_voting_data()
	elif option == 3:
		return prepare_crx_data()
	elif option == 4:
		return prepare_king_data()
	else:
		return prepare_iris_data()

def main():
	print("0 - mine")
	print("1 - sklearn")
	option = int(input("Select option by number: "))
	if option == 1:
		print("Using iris dataset")
		iris = load_iris()
		X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
		classifier = tree.DecisionTreeClassifier()
		model = classifier.fit(X_train, y_train)
		predicted = model.predict(X_test)
		average_correct = 0
		correct = 0
		for test, pred in zip(y_test, predicted):
			if test == pred:
				correct += 1
		average_correct = correct / len(predicted) * 100
		print("Accuracy: " + str(average_correct))
	else:
		df = get_data()
		df_train, df_test = train_test_split(df, test_size=0.3)
		classifier = TreeClassifier()
		model = classifier.fit(df_train, "class")
		predicted = model.predict(df_test)
		average_correct = 0
		correct = 0
		for test, pred in zip(df_test["class"], predicted):
			if test == pred:
				correct += 1
		average_correct = correct / len(df_test["class"]) * 100
		print("Accuracy: " + str(average_correct))
		model.display()

if __name__ == "__main__":
	main()