import numpy as np
import pandas as pd


class Node:

    def __init__(self):
        self.left = None
        self.right = None
        self.term = False
        self.label = None
        self.feature = None
        self.value = None

    def set_split(self, feature, value):
        self.feature = feature
        self.value = value

    def set_term(self, label):
        self.term = True
        self.label = label


class DecisionTree:

    def __init__(self, min_el=1, criterion="gini"):
        self.root = Node()
        self.min_el = min_el
        self.criterion = criterion

    def _gini_impurity(self, classes):
        prop = np.unique(classes, return_counts=True)[1] / len(classes)
        return 1 - np.sum(prop * prop)

    def _w_gini_impurity(self, parts):
        res = 0
        size = sum(len(part) for part in parts)
        for part in parts:
            res += self._gini_impurity(part) * len(part) / size
        return res

    def _split_fun(self, data, res):
        features = data.columns
        split_feature, split_val, left, right = None, None, [], []
        min_gini = 1
        for feature in features:
            cur = data[feature]
            values = cur.unique()
            for value in values:
                cur_left = cur.where(cur == value).dropna().index
                cur_right = cur.where(cur != value).dropna().index
                cur_gini = self._w_gini_impurity([res[cur_left], res[cur_right]])
                if cur_gini < min_gini:
                    min_gini = cur_gini
                    split_feature = feature
                    split_val = value
                    left = cur_left
                    right = cur_right
        return min_gini, split_feature, split_val, left, right

    def _split(self, node, data, res):
        same = True
        gini = self._gini_impurity(res)
        for col in data.columns:
            if len(np.unique(data[col])) != 1:
                same = False
                break
        if len(data) <= self.min_el or gini == 0.0 or same:
            index = np.argmax(np.unique(res, return_counts=True)[1])
            node.set_term(np.unique(res)[index])
        else:
            w_gini, feature, val, left, right = self._split_fun(data, res)
            node.set_split(feature, val)
            node.left = Node()
            left_data = data.iloc[left, :].copy()
            left_data.index = range(len(left_data))
            self._split(node.left, left_data, res[left])
            node.right = Node()
            right_data = data.iloc[right, :].copy()
            right_data.index = range(len(right_data))
            self._split(node.right, right_data, res[right])

    def _pred(self, row, node):
        if node.term is True:
            print(f"   Predicted label: {node.label}")  # commented because of the 6th stage
            return node.label
        print(f"   Considering decision rule on feature {node.feature} with value {node.value}")
        if row[node.feature] == node.value:
            return self._pred(row, node.left)
        else:
            return self._pred(row, node.right)

    def fit(self, data, res):
        self._split(self.root, data, res)

    def predict(self, data):
        res = []
        for i in range(len(data)):
            cur = data.loc[i,:]
            print(f"Prediction for sample # {cur.name}")
            res.append(self._pred(cur, self.root))
        return np.array(res)

file_names = input().split()
train_data = pd.read_csv(file_names[0])
test_data = pd.read_csv(file_names[1])
features_train_stage5 = train_data[["Pclass", "Sex", "SibSp", "Parch"]]
target_train_stage5 = train_data["Survived"].values
features_test_stage5 = test_data[["Pclass", "Sex", "SibSp", "Parch"]]
tree = DecisionTree()
tree.fit(features_train_stage5, target_train_stage5)
y_pred = tree.predict(features_test_stage5)
