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
    n, d   = X.shape
    K      = mixture.mu.shape[0]

    # ---------- 便利な集計 ----------
    Nk          = post.sum(axis=0)           # shape (K,)
    obs_mask    = X != 0                     # (n,d)  True where observed
    obs_counts  = obs_mask.sum(axis=1)       # (n,)

    # ---------- 更新: 平均 µ ----------
    mu_new = np.zeros((K, d))

    # 分子: Σ_i γ_{ik} x_il       ;   分母: Σ_i γ_{ik}   (観測されているときだけ)
    for k in range(K):
        # 重み付きの観測値合計
        weighted_sum = (post[:, [k]] * X * obs_mask).sum(axis=0)   # (d,)
        # 重み付きの観測マスク合計
        mask_sum     = (post[:, [k]] * obs_mask).sum(axis=0)       # (d,)

        # 要素ごとに割り算。観測がない次元では旧 µ をそのまま残す
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_k = np.divide(weighted_sum, mask_sum,
                             out=mixture.mu[k].copy(),  # default = old µ
                             where=mask_sum > 0)
        mu_new[k] = mu_k

    # ---------- 更新: 分散 σ² ----------
    var_new = np.zeros(K)
    for k in range(K):
        # (n,d) 行列: (x_il − µ_kl)²  * mask
        diff2 = ((X - mu_new[k]) ** 2) * obs_mask
        # 分子: Σ_i γ_{ik} Σ_{l∈O_i} (x_il − µ_kl)²
        num = (post[:, [k]] * diff2).sum()
        # 分母: Σ_i γ_{ik} |O_i|
        den = (post[:, k] * obs_counts).sum()
        # 変動がない (den==0) → 旧値を保持
        sigma2 = num / den if den > 0 else mixture.var[k]
        var_new[k] = max(sigma2, min_variance)

    # ---------- 更新: 混合係数 π ----------
    p_new = Nk / n
    p_new = np.maximum(p_new, 1e-16)         # avoid zeros
    p_new /= p_new.sum()                     # renormalize

    return GaussianMixture(mu=mu_new, var=var_new, p=p_new)


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
    pre_log_likelihood = None
    log_likelihood = None

    while (pre_log_likelihood is None \
            or log_likelihood is None \
            or log_likelihood - pre_log_likelihood > 1e-6 * np.abs(log_likelihood)):
        pre_log_likelihood = log_likelihood
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K    = mixture.mu.shape[0]
    mu, var, pi = mixture.mu, mixture.var, mixture.p

    # ---------- 事後確率 γ_{ik} を計算 (E‑step と同じ) ----------
    obs_mask   = X != 0                     # (n,d)  True where observed
    obs_count  = obs_mask.sum(axis=1)       # |O_i|

    log_pi = np.log(pi + 1e-16)
    log2pi = np.log(2.0 * np.pi)

    log_prob = np.empty((n, K))
    for k in range(K):
        diff       = (X - mu[k]) * obs_mask
        sq_error   = (diff ** 2).sum(axis=1)          # ||x_O - μ_O||²
        log_prob[:, k] = (
              log_pi[k]
            - 0.5 * obs_count * (log2pi + np.log(var[k]))
            - 0.5 * sq_error / var[k]
        )

    log_denom = logsumexp(log_prob, axis=1, keepdims=True)   # (n,1)
    post      = np.exp(log_prob - log_denom)                 # (n,K)

    # ---------- 期待値で欠損を埋める ----------
    # 期待値 E[x_i | θ] = γ_i · μ  (shape = (n,d))
    exp_matrix = post @ mu                                   # (n,d)

    X_pred = X.astype(float).copy()
    missing_mask = ~obs_mask
    X_pred[missing_mask] = exp_matrix[missing_mask]

    return X_pred