import numpy as np
import em
import common

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")

# K = 4
# n, d = X.shape
# seed = 0

# mytest
# X = np.loadtxt("netflix_incomplete.txt")

# K = [1, 12]
# n, d = X.shape
# seed = [0, 1, 2, 3, 4]


# last_log_likelihood = []
# for i in K:
#     result = []
#     for j in seed:
#         GaussianMixture, post = common.init(X=X, K=i, seed=j)
#         mixture, post, log_likelihood = em.run(X, GaussianMixture, post)
#         result.append(log_likelihood)

#         # log likelihood
#         print("K", i, "seed", j, "log_likelihood", log_likelihood)
#     # last_log_likelihood.append(result.max())
#     print("K", i, "max log_likelihood", np.array(result).max())
    # print("min log_likelihood", np.array(result).min())
    # common.plot(X, mixture, post, title="Gausian Mixture")

# max
X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

GaussianMixture, post = common.init(X=X, K=12, seed=1)
mixture, post, log_likelihood = em.run(X, GaussianMixture, post)

X_pred = em.fill_matrix(X, mixture)
rmse = common.rmse(X_gold, X_pred)
print("RMSE", rmse)