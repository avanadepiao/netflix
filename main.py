import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
# TODO: Your code here
K = [1, 2, 3, 4]
seed = [0, 1, 2, 3, 4, 5]
for i in K:
    result = []
    for j in seed:
        GaussianMixture, post = common.init(X=X, K=i, seed=j)
        mixture, post, cost = kmeans.run(X, GaussianMixture, post)
        result.append(cost)

        # cost
        print("K", i, "seed", j, "cost", cost)
    
    print("min cost", np.array(result).min())

