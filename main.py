import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

main_df = np.loadtxt("datasets/iris.csv", delimiter=",", dtype=str)

def knn(df, X, Y, x, y, k):
    distances = np.sqrt(((X.astype(float) - x.astype(float)) ** 2) + ((Y.astype(float) - y.astype(float)) ** 2))
    dis_indexes = distances.argsort()[1:]
    return df[dis_indexes[:k]]

def predict(nbrs_list, x1, y1):
    varieties = nbrs_list[:, -1]
    unique, counts = np.unique(varieties, return_counts=True)
    count_vars = dict(zip(unique, counts))
    if len(unique) == 1:
        return unique[0]
    else:
        count_dict = Counter(count_vars.values())
        result = [key for key, value in count_vars.items()
                  if count_dict[value] > 1]
        if result:
            dict1 = {}
            for value in unique:
                vals = nbrs_list[nbrs_list[:, -1] == value]
                counter = 1
                distances = 0
                for val in vals:
                    distance = np.sqrt(((val[0].astype(float) - x1.astype(float)) ** 2) + (
                                (val[1].astype(float) - y1.astype(float)) ** 2))
                    distances += distance
                    counter += 1
                average_distance = distances / counter
                dict1[value] = average_distance
            return min(dict1, key=dict1.get)
        else:
            inverse = [(value, key) for key, value in count_vars.items()]
            return max(inverse)[1]


ks = {"k2": 0, "k3": 0, "k4": 0, "k5": 0, "k6": 0, "k7": 0, "k8": 0, "k9": 0, "k10": 0}
for k in ks:
    pk = []
    for i in range(1, main_df.shape[0]):
        neighbours = knn(main_df[1:], main_df[1:, 0], main_df[1:, 1], main_df[i, 0], main_df[i, 1], int(k[1:]))
        predicted_variety = predict(neighbours, main_df[i, 0], main_df[i, 1])
        pk.append(predicted_variety)
    ks[k] = sum(main_df[1:, -1] == pk)


plt.plot(["k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10"], [ks[k] for k in ks])
plt.show()