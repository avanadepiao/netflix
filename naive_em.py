"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from sklearn.covariance import log_likelihood
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
    raise NotImplementedError


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
    raise NotImplementedError
