import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
# TODO: Your code here
K = [1, 2, 3, 4]
seed = [0, 1, 2, 3, 4]
# for i in K:
#     result = []
#     for j in seed:
#         GaussianMixture, post = common.init(X=X, K=i, seed=j)
#         mixture, post, cost = kmeans.run(X, GaussianMixture, post)
#         result.append(cost)

#         # cost
#         print("K", i, "seed", j, "cost", cost)
    
#     print("min cost", np.array(result).min())
new_log_likelihood = []
bic_score = []
for i in K:
    result = []
    for j in seed:
        GaussianMixture, post = common.init(X=X, K=i, seed=j)
        mixture, post, log_likelihood = naive_em.run(X, GaussianMixture, post)
        result.append(log_likelihood)
        new_log_likelihood.append(log_likelihood)
        # log likelihood
        # print("K", i, "seed", j, "log_likelihood", log_likelihood)
        bic_score.append(common.bic(X=X, mixture=mixture, log_likelihood=log_likelihood))
        # print("bic: ", bic_score)
    # print("min log_likelihood", np.array(result).min())
# common.plot(X, mixture, post, title="Gausian Mixture")

print(np.array(new_log_likelihood))
print(np.array(bic_score))
print(np.array(new_log_likelihood).max(), np.array(bic_score).max())