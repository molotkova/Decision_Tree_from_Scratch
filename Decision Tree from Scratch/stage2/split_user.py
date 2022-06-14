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


file_name = input()
data = pd.read_csv(file_name)
features_stage2 = data[["Pclass", "Sex"]]
target_stage2 = data["Survived"].values
print(*split_fun(features_stage2, target_stage2))

# correct output
# print("0.0 Sex 0 [0, 4, 5, 6, 7] [1, 2, 3, 8, 9]")