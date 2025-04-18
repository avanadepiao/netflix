import numpy as np
from typing import NamedTuple

class GaussianMixture(NamedTuple):
    mu: np.ndarray     # (K, d)
    var: np.ndarray    # (K,)
    p: np.ndarray      # (K,)

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    n, d = X.shape
    _, K = post.shape

    # 各クラスタの重み（soft count の合計） → shape (K,)
    Nk = np.sum(post, axis=0)  # クラスタごとの合計責任値（soft counts）

    # 平均の更新: mu_k = sum_i(post_ik * x_i) / Nk[k]
    mu = (post.T @ X) / Nk[:, np.newaxis]  # shape: (K, d)

    # 分散の更新: σ²_k = weighted average of squared distances
    var = np.zeros(K)
    for k in range(K):
        diff = X - mu[k]                      # shape: (n, d)
        squared_dist = np.sum(diff**2, axis=1)  # shape: (n,)
        weighted_squared_dist = post[:, k] * squared_dist  # shape: (n,)
        var[k] = np.sum(weighted_squared_dist) / (d * Nk[k])  # スカラー

    # 混合比（重み）: p_k = Nk / n
    p = Nk / n  # shape: (K,)

    return GaussianMixture(mu=mu, var=var, p=p)
