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

    def __init__(self, min_el=1, criterion="gini", numeric_features=None):
        self.root = Node()
        self.min_el = min_el
        self.criterion = criterion
        self.numeric_features = numeric_features

    def _gini_impurity(self, classes):
        prop = np.unique(classes, return_counts=True)[1] / len(classes)
        return 1 - np.sum(prop * prop)

    def _w_gini_impurity(self, parts):
        res = 0
        size = sum(len(part) for part in parts)
        for part in parts:
            res += self._gini_impurity(part) * len(part) / size
        return res

    def _split_fun_with_num(self, data, res):
        features = data.columns
        split_feature, split_val, left, right = None, None, [], []
        min_gini = 1
        for feature in features:
            cur = data[feature]
            values = cur.unique()
            for value in values:
                if feature in self.numeric_features:
                    cur_left = cur.where(cur <= value).dropna().index
                    cur_right = cur.where(cur > value).dropna().index
                else:
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

    def _split_with_num(self, node, data, res):
        gini, feature, val, left, right = self._split_fun_with_num(data, res)
        same = True
        for col in data.columns:
            if (len(np.unique(data[col])) != 1):
                same = False
                break
        if len(data) <= self.min_el or gini == 0.0 or same:
            index = np.argmax(np.unique(res, return_counts=True)[1])
            node.set_term(np.unique(res)[index])
        else:
            node.set_split(feature, val)
            print(f"Made split: {feature} is {val}") # commented because of the 5th stage
            node.left = Node()
            left_data = data.iloc[left, :].copy()
            left_data.index = range(len(left_data))
            self._split_with_num(node.left, left_data, res[left])
            node.right = Node()
            right_data = data.iloc[right, :].copy()
            right_data.index = range(len(right_data))
            self._split_with_num(node.right, right_data, res[right])

    def _pred_with_num(self, row, node):
        if (node.term is True):
            print(f"   Predicted label: {node.label}")
            return node.label
        print(f"   Considering decision rule on feature {node.feature} with value {node.value}")
        if node.feature in self.numeric_features:
            if (row[node.feature] <= node.value):
                return self._pred_with_num(row, node.left)
            else:
                return self._pred_with_num(row, node.right)
        else:
            if (row[node.feature] == node.value):
                return self._pred_with_num(row, node.left)
            else:
                return self._pred_with_num(row, node.right)

    def fit(self, data, res):
        self._split_with_num(self.root, data, res)

    def predict(self, data):
        res = []
        for i in range(len(data)):
            cur = data.loc[i,:]
            print(f"Prediction for sample # {cur.name}")
            res.append(self._pred_with_num(cur, self.root))
        return np.array(res)


file_names = input().split()
train_data = pd.read_csv(file_names[0])
test_data = pd.read_csv(file_names[1])
features_train_stage8 = train_data[["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"]]
target_train_stage8 = train_data["Survived"].values
features_test_stage8 = test_data[["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"]]
tree = DecisionTree(numeric_features=["Age", "Fare"])
tree.fit(features_train_stage8, target_train_stage8)
y_pred = tree.predict(features_test_stage8)
