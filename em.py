"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import log_likelihood
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d     = X.shape
    K        = mixture.mu.shape[0]
    mu, var, pi = mixture.mu, mixture.var, mixture.p

    obs_mask   = X != 0
    obs_counts = obs_mask.sum(axis=1)

    log_pi = np.log(pi + 1e-16)
    log2pi = np.log(2*np.pi)

    log_prob = np.empty((n, K))
    for k in range(K):
        diff      = (X - mu[k]) * obs_mask
        sq_error  = (diff**2).sum(axis=1)
        log_prob[:, k] = (
            log_pi[k]
          - 0.5 * obs_counts * (log2pi + np.log(var[k]))
          - 0.5 * sq_error / var[k]
        )

    log_denom = logsumexp(log_prob, axis=1, keepdims=True)  # shape (n,1)
    post      = np.exp(log_prob - log_denom)
    ll        = log_denom.sum()                             # scalar

    return post, ll



    


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

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


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
