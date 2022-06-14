import numpy as np
import pandas as pd


def gini_impurity(classes):
    prop = np.unique(classes, return_counts=True)[1] / len(classes)
    return 1 - np.sum(prop * prop)


def w_gini_impurity(parts):
    res = 0
    size = sum(len(part) for part in parts)
    for part in parts:
        res += gini_impurity(part) * len(part) / size
    return res


def split_fun(data, res):
    features = data.columns
    split_feature, split_val, left, right = None, None, [], []
    min_gini = 1
    for feature in features:
        cur = data[feature]
        values = cur.unique()
        for value in values:
            cur_left = cur.where(cur == value).dropna().index.tolist()
            cur_right = cur.where(cur != value).dropna().index.tolist()
            cur_gini = w_gini_impurity([res[cur_left], res[cur_right]])
            if cur_gini < min_gini:
                min_gini = cur_gini
                split_feature = feature
                split_val = value
                left = cur_left
                right = cur_right
    return min_gini, split_feature, split_val, left, right


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


MIN_EL = 1


def split(node, data, res):
    gini, feature, val, left, right = split_fun(data, res)
    same = True
    for col in data.columns:
        if len(np.unique(data[col])) != 1:
            same = False
            break
    if len(data) <= MIN_EL or gini == 0.0 or same:
        index = np.argmax(np.unique(res, return_counts=True)[1])
        node.set_term(np.unique(res)[index])
    else:
        node.set_split(feature, val)
        print(f"Made split: {feature} is {val}")
        node.left = Node()
        left_data = data.iloc[left, :].copy()
        left_data.index = range(len(left_data))
        split(node.left, left_data, res[left])
        node.right = Node()
        right_data = data.iloc[right, :].copy()
        right_data.index = range(len(right_data))
        split(node.right, right_data, res[right])


file_name = input()
data = pd.read_csv(file_name)
features_stage3 = data[["Pclass", "Sex"]]
target_stage3 = data["Survived"].values
root = Node()
split(root, features_stage3, target_stage3)
