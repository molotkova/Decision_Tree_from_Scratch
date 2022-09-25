import numpy as np


def gini_impurity(classes):
    prop = np.unique(classes, return_counts=True)[1] / len(classes)
    return 1 - np.sum(prop * prop)


def w_gini_impurity(parts):
    res = 0
    size = sum(len(part) for part in parts)
    for part in parts:
        res += gini_impurity(part) * len(part) / size
    return round(res, 5)


node_inp = input().split()
first_inp = input().split()
second_inp = input().split()
print(round(gini_impurity(node_inp), 2), round(w_gini_impurity([first_inp, second_inp]), 2))
