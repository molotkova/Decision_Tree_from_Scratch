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
    return round(res, 5)


NUMERIC_FEATURES = ["Age", "Fare"]


def split_fun_with_num(data, res):
    features = data.columns
    split_feature, split_val, left, right = None, None, [], []
    min_gini = 1
    for feature in features:
        cur = data[feature]
        values = cur.unique()
        for value in values:
            if feature in NUMERIC_FEATURES:
                cur_left = cur.where(cur <= value).dropna().index
                cur_right = cur.where(cur > value).dropna().index
            else:
                cur_left = cur.where(cur == value).dropna().index
                cur_right = cur.where(cur != value).dropna().index
            cur_gini = w_gini_impurity([res[cur_left], res[cur_right]])
            if cur_gini < min_gini:
                min_gini = cur_gini
                split_feature = feature
                split_val = value
                left = cur_left.tolist()
                right = cur_right.tolist()
    return min_gini, split_feature, split_val, left, right


file_name = input()
data = pd.read_csv(file_name)
features_stage7 = data[["Pclass", "Sex", "Age", "Fare"]]
target_stage7 = data["Survived"].values
print(*split_fun_with_num(features_stage7, target_stage7))
