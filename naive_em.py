"""Mixture model using EM"""
from math import log
from typing import Tuple
import numpy as np
from sklearn.covariance import log_likelihood
from sklearn.metrics import euclidean_distances
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    log_likelihood = 0
    # p( i | theta ) = p_1* (1 / sqrt(2*np.pi)*sigma * np.exp(-(x-mu)**2/2*sigma**2))

    for k in range(K):

        # (n, d) - (d, ) -> (n, d)
        diff = X - mixture.mu[k]
        # exp部分: -(x - mu)**2 / (2 * sigma**2)
        # sum: (n, d) -> (n,), mixture.var[k]はスカラー (1, )
        # (n,) / (1,) -> (n,)
        exponent = -0.5 * np.sum((diff**2), axis=1) / mixture.var[k]

        # ガウス密度 (係数含む) -> (n, )
        coef = 1 / (2 * np.pi * mixture.var[k])**(d / 2)
        gaussian = coef * np.exp(exponent)

        # 重みを掛けて加算 (n, )
        post[:, k] = mixture.p[k] * gaussian

    # 各データ点の属するクラスの合計 (n, k) -> (n, 1)
    # 行ごとに合計を取る -> p1*gaussian + p2*gaussian + p3*gaussian + ...
    post_sum = np.sum(post, axis=1, keepdims=True)

    # posterior soft assignment (n, k)
    post /= post_sum

    # log-likelihood
    log_likelihood = np.sum(np.log(post_sum))

    return post, log_likelihood
    

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    # sigma_1^n p(j | i) 行の合計
    # 各クラスタの重み soft countの合計 (n, K) -> (K, )
    post_sum = post.sum(axis=0)
    # (n, d) -> (1, d)
    X_sum = X.sum(axis=0, keepdims=True)
    # mu (K, d) (k, 1) @ (1, d) -> (k, d)
    mu_hat = post.T @ X / post_sum[:, np.newaxis]  # (K, d)

    # 混合比（重み）p (k, )
    p_hat = post_sum / n

    # 分散の更新 (a_k)^2 =weighted average of squared distances
    var_hat = np.zeros(K) # (K, )
    for k in range(K):
        diff = X - mu_hat[k]  # (n, d) - (1, d) -> (n, d)
        squared_dist = np.sum(diff**2, axis=1) # (n, )
        weighted_squared_dist = post[:, k] * squared_dist # (n, ) * (n, ) -> (n, )
        var_hat[k] = np.sum(weighted_squared_dist) / (d * post_sum[k]) # スカラー

    return GaussianMixture(mu=mu_hat, var=var_hat, p=p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    pre_log_likelihood = np.zeros
    log_likelihood = np.zeros
    while (pre_log_likelihood is None or pre_log_likelihood - log_likelihood > 1e-6):
        pre_log_likelihood = log_likelihood
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
