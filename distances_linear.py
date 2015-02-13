"""
Pairwise distance functions between time series in the input space
==================================================================

They all have the following prototype:

        function(bcsc1, bcsc2, **kwargs)

"""


import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import solve, eigh

from .utils import compute_autocov, compute_autocorr
from .global_align import tga_dissimilarity


def linear_diff_means(bcsc1, bcsc2):
    """ Return the squared Euclidian-distance between time-series' means

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    Returns
    -------
    ddm: double,
         squared Euclidean distance between the means of the time series
    """
    m1 = np.asarray(bcsc1.mean(axis=1)).squeeze()
    m2 = np.asarray(bcsc2.mean(axis=1)).squeeze()
    ddm = ((m2 - m1) ** 2).sum()
    return ddm


def linear_mean_diffs(bcsc1, bcsc2):
    """ Return the mean of Euclidian-distances between time-series

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    Returns
    -------
    dmd: double,
         mean of the squared Euclidean distances between the time series
    """
    T = bcsc1.shape[1]
    assert T == bcsc2.shape[1], "the series should be of same duration"
    dmd = 1.0 / T * ((bcsc2 - bcsc1).data ** 2).sum()
    return dmd


def linear_allpairs(bcsc1, bcsc2):
    """ Return the mean of all pairwise dot products (*similarity*) between two-time series

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    Returns
    -------
    sim_ap: double,
            mean of all pairwise frame dot products

    Notes
    -----
    It's a *similarity*, not a distance!
    """
    sim_ap = (bcsc1.T * bcsc2).mean()  # * sparse matrices == matrix product!!!
    return sim_ap


def linear_hsac(bcsc1, bcsc2, tau=1, mntype=0):
    """ Return the distance between the auto-covariance matrices of two time-series

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    tau: int, optional, default: 1
         lag parameter

    mntype: int (default: 0), determines matrix norm used:
            0: Frobenius (HS) norm
            1: largest eigen-value

    Returns
    -------
    dhsac: double,
            distance between the auto-covariance matrices.
    """
    d = bcsc1.shape
    # autocovariances
    acv21 = compute_autocov(bcsc2, tau=tau)
    acv21 -= compute_autocov(bcsc1, tau=tau)
    # compute the distance
    if mntype == 0:
        # get the squared Frobenius norm of the difference between auto-covariances
        dhsac = np.core.add.reduce((acv21 * acv21).ravel())  # from numpy.linalg.norm
    elif mntype == 1:
        # get the largest eigenvalue of the difference between auto-covariances
        dhsac = eigh(acv21, eigvals_only=True, eigvals=(d - 1, d - 1))[0]
    else:
        raise ValueError("Invalid matrix norm type ({})".format(mntype))
    return dhsac


def linear_nhsac(bcsc1, bcsc2, tau=1, regul=1e-3, check_regul=False, mntype=0):
    """ Return the difference between auto-covariances of the time-series,
    normalized by the overall variance

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    tau: int, optional, default: 1
         lag parameter

    regul: float, optional, default: 1e-2
           regularization parameter for the inverse computation
           if < 0, then set regul to be 50% of the largest singular value

    check_regul: boolean, optional, default: False,
                 if True, then check that the regul parameter is smaller
                 than the largest eigen-values of the covariance matrices

    mntype: int (default: 0), determines matrix norm used:
            0: Frobenius (HS) norm
            1: largest eigen-value

    Returns
    -------
    dnhsac: double,
            variance-normalized distance between the auto-covariance matrices.
    """
    d = bcsc1.shape[0]  # d: number of vars
    T1 = bcsc1.shape[1]
    T2 = bcsc2.shape[1]
    # autocovariances
    acv21 = compute_autocov(bcsc2, tau=tau)
    acv21 -= compute_autocov(bcsc1, tau=tau)
    # normalize by overall frame covariance matrix
    C = np.cov(np.hstack([bcsc1.toarray(), bcsc2.toarray()]))
    # add regularization term
    if check_regul or regul < 0:
        mev = eigh(C, eigvals_only=True, eigvals=(d - 1, d - 1))[0]
        if regul < 0:
            used_regul = mev * 0.5
        else:
            assert regul < mev, "Too high regularization parameter"
            used_regul = regul
    else:
        used_regul = regul
    C += used_regul * np.eye(d)
    # compute the distance
    if mntype == 0:
        # get the squared Frobenius norm of the normalized difference between auto-covariances
        nacv21 = solve(C, acv21, sym_pos=True, overwrite_a=True, overwrite_b=True)
        dnhsac = np.core.add.reduce((nacv21 * nacv21).ravel())  # from numpy.linalg.norm
    elif mntype == 1:
        # get largest eigenvalue of the normalized difference between auto-covariances
        dnhsac = eigh(acv21, C, eigvals_only=True, eigvals=(d - 1, d - 1))[0]
    else:
        raise ValueError("Invalid matrix norm type ({})".format(mntype))
    return dnhsac


def linear_diff_autocor(bcsc1, bcsc2, tau=1, regul=1e-3, mntype=0):
    """ Distance between the repsective auto-correlation matrices of two time-series

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    tau: int, optional, default: 1
         lag parameter

    regul: float, optional, default: 1e-2
           regularization parameter for the inverse computation
           if < 0, then set regul to be 50% of the largest singular value

    mntype: int (default: 0), determines matrix norm used:
            0: Frobenius (HS) norm
            1: largest eigen-value

    Returns
    -------
    daco: double,
          distance between the auto-correlation matrices.

    Notes
    -----
    With Frobenius, this is equivalent to the DACO distance with a linear kernel.
    """
    d, T = bcsc1.shape  # d: number of vars, T: number of observations
    # autocorrelations
    acr21 = compute_autocorr(bcsc2, tau=tau, regul=regul)
    acr21 -= compute_autocorr(bcsc1, tau=tau, regul=regul)
    # compute the distance
    if mntype == 0:
        # get the squared Frobenius norm of the difference between auto-covariances
        daco = np.core.add.reduce((acr21 * acr21).ravel())  # from numpy.linalg.norm
    elif mntype == 1:
        # get the largest eigenvalue of the difference between auto-correlations
        daco = eigh(acr21, eigvals_only=True, eigvals=(d - 1, d - 1))[0]
    else:
        raise ValueError("Invalid matrix norm type ({})".format(mntype))
    return daco


def linear_crosscor(bcsc1, bcsc2, regul=1e-3, check_regul=False, mntype=0):
    """ Return the cross-correlation between time-series obtained by (linear) CCA

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    tau: int, optional, default: 1
         lag parameter

    regul: float, optional, default: 1e-2
           regularization parameter for the inverse computation
           if < 0, then set regul to be 50% of the largest singular value

    check_regul: boolean, optional, default: False,
                 if True, then check that the regul parameter is smaller
                 than the largest eigen-values of the covariance matrices

    mntype: int (default: 0), determines matrix norm used:
            0: Frobenius (HS) norm
            1: largest eigen-value

    Returns
    -------
    ccsim: double,
           norm of the cross-correlation matrix

    Notes
    -----
    Not a distance but a similarity!
    """
    d, T = bcsc1.shape  # d: number of vars, T: number of observations
    assert bcsc2.shape[1] == T, "Series must be of same duration"
    # full covariance matrix
    C = np.cov(bcsc1.toarray(), bcsc2.toarray())
    # add regularization terms
    if check_regul or regul < 0:
        mev1 = eigh(C[:d, :d], eigvals_only=True, eigvals=(d - 1, d - 1))[0]
        mev2 = eigh(C[d:, d:], eigvals_only=True, eigvals=(d - 1, d - 1))[0]
        #print "         mev1=%f, mev2=%f" % (mev1, mev2) # DEBUG
        if regul < 0:
            used_regul = min(mev1, mev2) * 0.5
        else:
            assert regul < mev1 and regul < mev2, "Too high regularization parameter"
            used_regul = regul
    else:
        used_regul = regul
    C[:d, :d] += used_regul * np.eye(d)
    C[d:, d:] += used_regul * np.eye(d)
    # build generalized eigenvalue problem A v = w B v
    A = C.copy()
    A[:d, :d] = 0.0
    A[d:, d:] = 0.0
    B = C  # .copy()
    B[:d, d:] = 0.0
    B[d:, :d] = 0.0
    # compute the similarity
    if mntype == 0:
        # get the squared Frobenius norm (trace)
        BinvA = solve(B, A)
        ccsim = np.core.add.reduce((BinvA * BinvA).ravel())  # from numpy.linalg.norm
    elif mntype == 1:
        # get largest eigenvalue of generalized eigenvalue problem (assumes B is p.d.)
        ccsim = eigh(A, B, eigvals_only=True, eigvals=(2 * d - 1, 2 * d - 1))[0]
    elif mntype == 2:
        # sum of the largest eigenvalues of generalized eigenvalue problem (assumes B is p.d.)
        ccsim = np.sum(eigh(A, B, eigvals_only=True))
    else:
        raise ValueError("Invalid matrix norm type ({})".format(mntype))
    return ccsim


def minus_logGAK(bcsc1, bcsc2, regul=1e0, tau=0):
    """ Return minus the normalized log Global Alignment kernel between series

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
           contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
           contains the sparse column wise representation of time series 2

    tau: int, optional, default: 0,
         'triangular' parameter of logGAK

    regul: float, optional, default: 1e0,
           'sigma' parameter of logGAK

    Returns
    -------
    mlga: double,
          minus the normalized log Global Alignment score.

    Note
    ----
    This is actually a non-linear kernel, but this function has the same
    signature as linear distances.
    """
    mlga = tga_dissimilarity(bcsc1.T.toarray(), bcsc2.T.toarray(), regul, tau)
    return mlga


def linear_autocov_likelihood_ratio(bcsc1, bcsc2, tau=1):
    """ P-value of statistical test where H0 is auto-covariance equality

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    tau: int, optional, default: 1
         lag parameter

    Returns
    -------
    daclr: double,
           p-value of the statistical test where H0: auto-covs are equal

    Notes
    -----
    Not a distance but a similarity: if it is low, then we can reject the null
    hypothesis, i.e. the auto-covariances are different.

    This is not exactly correct: we base this similarity on the statistical
    test for the equality of covariances (not auto-covariances) matrices under
    normality assumptions (i.e. the column vectors are drawn from a Gaussian
    distribution).
    """
    d = bcsc1.shape[0]  # d: number of vars
    T1 = bcsc1.shape[1]
    T2 = bcsc2.shape[1]
    T = T1 + T2
    # compute the log det of the autocovariances
    A1 = compute_autocov(bcsc1, tau=tau)
    det1 = slogdet(A1)[1]
    A2 = compute_autocov(bcsc2, tau=tau)
    det2 = slogdet(A2)[1]
    A = 1. / T * (T1 * A1 + T2 * A2)  # mean of the auto-covariances
    daclr = max(0.0, T * slogdet(A)[1] - T1 * det1 - T2 * det2)
    # Note: threshold is just for numerical issues (very low and small value)
    return daclr


# XXX not designed for auto-co{r,v} and sensitive to departure from normality
def linear_autocor_likelihood_ratio(bcsc1, bcsc2, tau=1, regul=1e-3):
    """ P-value of statistical test where H0 is auto-correlation equality

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
            contains the sparse column wise representation of time series 2

    tau: int, optional, default: 1
         lag parameter

    regul: float, optional, default: 1e-2
           regularization parameter for the inverse computation
           if < 0, then set regul to be 50% of the largest singular value

    Returns
    -------
    darlr: double,
           p-value of the statistical test where H0: auto-cors are equal

    Notes
    -----
    Not a distance but a similarity: if it is low, then we can reject the null
    hypothesis, i.e. the auto-correlations are different.

    This is not exactly correct: we base this similarity on the statistical
    test for the equality of covariances (not auto-corrleations) matrices under
    normality assumptions (i.e. the column vectors are drawn from a Gaussian
    distribution).
    """
    d = bcsc1.shape[0]  # d: number of vars
    T1 = bcsc1.shape[1]
    T2 = bcsc2.shape[1]
    T = T1 + T2
    # compute the log det of the autocovariances
    A1 = compute_autocorr(bcsc1, tau=tau, regul=regul)
    det1 = slogdet(A1)[1]
    A2 = compute_autocorr(bcsc2, tau=tau, regul=regul)
    det2 = slogdet(A2)[1]
    A = 1. / T * (T1 * A1 + T2 * A2)  # mean of the auto-covariances
    darlr = max(0.0, T * slogdet(A)[1] - T1 * det1 - T2 * det2)
    # Note: threshold is just for numerical issues (very low and small value)
    return darlr
