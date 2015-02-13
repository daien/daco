""" Various functions for experimental purposes
"""


import numpy as np
from scipy.linalg import solve


from .utils import center_gram
from .distances_rkhs import distance_mahalanobis


def hsic(K1, K2):
    """ Compute the Hilbert-Schmidt independence criterion

    Parameters
    ----------
    K1: centered Gram matrix on time series 1
    K2: centered Gram matrix on time series 2

    Returns
    -------
    dhsic2: double,
        squared Hilbert-Schmidt norm of the empirical cross-covariance operator
        \hat{M}_{YX}^{(T)} = || \hat{\sigma}_{YX}^{(T)} ||_{HS}^2
                           = 1/T^2 Trace(\\tilde{K}_x, \\tilde{K}_y))

    Notes
    -----
    It's a "similarity measure": if the two series are sampled from
    independent distributions, then HSIC = 0

    """
    T = K1.shape[0]
    #return 1.0/(T*T) * np.dot(K1, K2).trace() # not efficient
    # 4x more efficient solution using trace of product as sum of Hadamard product
    return np.mean(K1 * K2)


def hsnorm_autocovariance(x_gram, tau=1, mode="truncated"):
    """ Compute the Hilbert-Schmidt norm of the auto-covariance operator in a RKHS

    || \\scov^{(x)}_{\\tau} ||_{HS}^2 =
        \\frac{1}{T^2} Tr(\\tilde{K}^{(x)} \\tilde{K}^{(x^\\tau)})

    Parameters
    ----------
    x_gram: T x T symmetric array containing the kernel evaluations between
        all vectors x_t of a time series x of duration T
    tau: lag, ie time shift used in the auto-covariance computation
    mode: {"truncated","cyclic"},
        either zero-padding (truncated) or cyclic estimation of HSIC

    Returns
    -------
    hsnorm: || \\scov^{(x)}_{\\tau} ||_{HS}^2
    """
    # define the sub-matrices
    T = x_gram.shape[0]
    if mode == "truncated":
        K = x_gram[:T - tau, :T - tau].copy()
        center_gram(K)
        Ktau = x_gram[tau:, tau:].copy()
        center_gram(Ktau)
    elif mode == "cyclic":
        K = x_gram.copy()
        center_gram(K)
        idxs = np.arange(tau, T + tau) % T
        perm_slice = np.ix_(idxs, idxs)
        Ktau = K[perm_slice]
    else:
        raise ValueError("Unknown mode (%s)" % mode)
    return hsic(K, Ktau)


def hsnorm_autocorrelation(x_gram, bf=0, ef=0, tau=1, regul=1e-3):
    """ Compute the Hilbert-Schmidt norm of the auto-correlation operator in a RKHS

    Parameters
    ----------
    x_gram: T x T symmetric array,
        containing the kernel evaluations between all vectors x_t of a time
        series x of duration T
    bf: int,
        begining frame (default: 0, beginning of the series)
    ef: int,
        end frame (default: 0, end of the series minus tau)
    tau: int,
        lag, ie time shift used in the auto-covariance computation
    regul: float,
        regularization parameter for the inverse computation

    Returns
    -------
    hsnorm: the Hilbert-Schmidt norm of the auto-correlation operator
    """
    T = x_gram.shape[0]
    if ef <= 0:
        ef = T - tau
    assert tau <= T / 2., "Too big tau"
    T = ef - bf
    K = x_gram[bf:ef, bf:ef].copy()
    center_gram(K)
    Ktau = x_gram[bf + tau:ef + tau, bf + tau:ef + tau].copy()
    center_gram(Ktau)
    # compute the different terms
    N = 1.0 / (regul * T) * np.eye(T)
    N -= 1.0 / (regul ** 2 * T) * solve(T * np.eye(T) + 1.0 / regul * K,
                                        K, sym_pos=True)
    hsnorm = np.trace(np.dot(np.dot(N.T, K), np.dot(N, Ktau)))
    return hsnorm


def kfda_separability(K, npos, nneg, regul=1e-3):
    """ Compute the separability between classes using kernel Fisher Discriminant Analysis

    According to kFDA, the separability is measured by the Mahalanobis distance,
    between the two classes, in the feature space induced by K

    Parameters
    ----------
    K: full kernel matrix between positive and negative samples
           [ [ K(xposs,xposs) , K(xposs, xnegs) ],
               [ K(xnegs, xposs), K(xnegs, xnegs) ] ]
    npos: int, number of positive samples (come first in K)
    nneg: int, number of negative samples (come last in K)
    regul: float, regularization parameter for Mahalanobis distance (default: 1e-3)

    Returns
    -------
    sep: float,
         separability between the two classes
    """
    # we multiply by regul to get a consistent scale across regul parameters
    return regul * distance_mahalanobis(K, npos, nneg, regul)
