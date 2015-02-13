"""
Pairwise distance functions between time series in a RKHS
=========================================================

They all have the following prototype:

        function(K, T1, T2, **kwargs)
"""


import numpy as np
from scipy.linalg import solve, eigvals, inv
from scipy.signal import correlate2d


# mean-element-based ----------------------------------------------------------


def distance_mean_elements(K, T1, T2):
    """ Compute the squared distance between mean elements of two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    Returns
    -------
    dme2: double,
          squared distance between the mean-elements in RKHS
    """
    dme2 = K[:T1, :T1].mean()
    dme2 += K[T1:, T1:].mean()
    dme2 += -2.0 * K[:T1, T1:].mean()
    # # normalization vector
    # m = np.zeros((T1+T2, 1), dtype=np.double)
    # m[:T1,:] = -1./T1
    # m[T1:,:] = 1./T2
    # # return the distance
    # dme2 = np.dot(m.T, np.dot(K, m))[0,0]
    return dme2


def distance_me_squared(K, T1, T2):
    """ Compute the squared distance between the squared mean elements of two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    Returns
    -------
    dme2: double,
          squared HS distance between the mean-elements squared
    """
    dme2 = (K[:T1, :T1].mean()) ** 2
    dme2 += (K[T1:, T1:].mean()) ** 2
    dme2 += -2.0 * (K[:T1, T1:].mean()) ** 2
    return dme2


def distance_mahalanobis(K, T1, T2, regul=1e-3):
    """ Compute the squared distance between mean elements of two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    regul: double, optional, default: 1e-3,
           regularization parameter

    Returns
    -------
    dmpc2: double,
           squared Mahalanobis distance between time-series in RKHS
    """
    # normalization vector
    n = T1 + T2
    m = np.zeros((n, 1), dtype=np.double)
    m[:T1, :] = -1.0 / T1
    m[T1:, :] = 1.0 / T2
    # centering matrix
    PiT1 = np.eye(T1, dtype=np.double) - 1.0 / T1
    PiT2 = np.eye(T2, dtype=np.double) - 1.0 / T2
    N = np.vstack([np.hstack([PiT1, np.zeros((T1, T2), dtype=np.double)]),
                   np.hstack([np.zeros((T2, T1), dtype=np.double), PiT2])])
    # compute the distance
    mTK = np.dot(m.T, K)
    me = np.dot(mTK, m)  # difference between mean elements
    mTKN = np.dot(mTK, N)
    NTK = np.dot(N.T, K)
    A = regul * np.eye(n) + 1.0 / n * np.dot(NTK, N)
    AinvNTK = solve(A, NTK, overwrite_a=True)  # A^{-1} N.T K
    AinvNTKm = np.dot(AinvNTK, m)
    dmpc2 = 1.0 / regul * (me - 1.0 / n * np.dot(mTKN, AinvNTKm))
    return dmpc2[0, 0]


# alignment-based -------------------------------------------------------------


def distance_aligned_frames_truncated(K, T1, T2, tau=0):
    """ Compute the squared distance between aligned frames

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 0,
         temporal shift (in frames) to apply to time series 2 before computing
         alignment, using "cyclic" padding

    Returns
    -------
    dme2: double,
          squared distance between aligned frames in the RKHS

    Notes
    -----
    Truncated verion (equivalent to zero padding)

        dme2 = K[0,0] - 1/(T2-tau) * sum_{t=0}^{T2-tau} K[x1_t, x2_{t+tau}]

    """
    assert T1 == T2, "the series should be of same duration"
    T = T1
    # constant base kernel value k(x,x)
    c = K[0, 0]
    # matrix of k(x,y)
    Kxy = K[:T, T:]
    # return the distance
    return c - np.mean(np.diag(Kxy, k=tau))


def distance_aligned_frames_cyclic(K, T1, T2, tau=0):
    """ Compute the squared distance between aligned frames

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: positive int, optional, default: 0,
         temporal shift (in frames) to apply to time series 2 before computing
         alignment, using "cyclic" padding

    Returns
    -------
    dme2: double,
          squared distance between aligned frames in the RKHS

    Notes
    -----
    Cyclic verion

        dme2 = K[0,0] - 1/T2 * sum_{t=0}^{T2} K[x1_t, x2_{(t+tau) % T2}]

    """
    assert T1 == T2, "the series should be of same duration"
    T = T1
    # constant base kernel value k(x,x)
    c = K[0, 0]
    # matrix of k(x,y)
    Kxy = K[:T, T:]
    # return the distance
    if tau:
        tr = Kxy.trace(offset=tau) + Kxy.trace(offset=tau - T)
    else:
        tr = Kxy.trace()
    return c - tr / float(T)


# auto-covariance-based -------------------------------------------------------


def distance_hsac_truncated(K, T1, T2, tau=1):
    """ Compute the squared HS distance between the autocovariance operators of
    two time series

    || \\scov^{(y)}_{\\tau} - \\scov^{(x)}_{\\tau} ||_{HS}^2 =
    1/T**2 ( Tr(K_1 x K_1^\\tau) + Tr(K_2 x K_2^\\tau) - 2 Tr(K_{1,2} x K_{2,1}^\\tau ) )

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: -1
         lag, ie time shift used in the auto-covariance computation

    Returns
    -------
    dhsac: double,
           squared Hilbert-Schmidt norm of the difference between the
           auto-covariance operators, in the RKHS induced by 'frame_kern', of
           the two time series

    Notes
    -----
    Truncated version between X[:-tau] and X[tau:] (equivalent to zero padding).
    """
    assert tau <= min(T1 / 2.0, T2 / 2.0), "Too big tau"
    # define the truncated matrices of the non-shifted series
    K1 = K[:T1 - tau, :T1 - tau]
    K2 = K[T1:T1 + T2 - tau, T1:T1 + T2 - tau]
    K12 = K[:T1 - tau, T1:T1 + T2 - tau]
    # define the truncated matrices of the shifted series
    K1tau = K[tau:T1, tau:T1]
    K2tau = K[T1 + tau:, T1 + tau:]
    K12tau = K[tau:T1, T1 + tau:]
    # compute the different traces using Hadamard products (and sym of K)
    tr1 = np.mean(K1 * K1tau)
    tr2 = np.mean(K2 * K2tau)
    tr12 = np.mean(K12 * K12tau)  # no transpose (K21tau.T == K12tau)
    # return dhsac
    return tr1 + tr2 - 2 * tr12


def distance_hsac_cyclic(K, T1, T2, tau=1):
    """ Compute the squared HS distance between the autocovariance operators of
    two time series

    || \\scov^{(y)}_{\\tau} - \\scov^{(x)}_{\\tau} ||_{HS}^2 =
    1/T**2 ( Tr(K_1 x K_1^\\tau) + Tr(K_2 x K_2^\\tau) - 2 Tr(K_{1,2} x K_{2,1}^\\tau ) )

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: -1
         lag, ie time shift used in the auto-covariance computation

    Returns
    -------
    dhsac: double,
           squared Hilbert-Schmidt norm of the difference between the
           auto-covariance operators, in the RKHS induced by 'frame_kern', of
           the two time series

    Notes
    -----
    Cyclic version between X and [ X[tau:], X[:tau] ].
    Artefacts may arise if the two series were not synchronized and comprised
    of the same number of periods.
    """
    assert tau <= min(T1 / 2.0, T2 / 2.0), "Too big tau"
    # define the (non-truncated) matrices of the non-shifted series
    K1 = K[:T1, :T1]
    K2 = K[T1:, T1:]
    K12 = K[:T1, T1:]
    # circular permutation of tau frames
    idxs1 = np.arange(tau, T1 + tau) % T1
    idxs2 = np.arange(tau, T2 + tau) % T2
    # Note: no need for copy as we re-use the previous centering (indep. of frame order)
    K1tau = K1[np.ix_(idxs1, idxs1)]
    K2tau = K2[np.ix_(idxs2, idxs2)]
    K12tau = K12[np.ix_(idxs1, idxs2)]
    # compute the different traces using Hadamard products (and sym of K)
    tr1 = np.mean(K1 * K1tau)
    tr2 = np.mean(K2 * K2tau)
    tr12 = np.mean(K12 * K12tau)  # no transpose (K21tau.T == K12tau)
    # return dhsac
    return tr1 + tr2 - 2 * tr12


# TODO use incomplete Cholesky decomposition (ST & C chap. 6, p. 175)
def hsnorm_cross_correlation(K, T1, T2, regul=1e-3):
    """ Compute the squared Hilbert-Schmidt norm of the cross-correlation

    This *similarity* measures the strength of the cross-correlation between
    two series, i.e. the degree to which you can linearly (in feature space!)
    predict one knowing the other (0 => not linked).

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
        between frames kernel matrix

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    regul: double, optional, default: 1e-3,
           regularization parameter

    Returns
    -------
    hscorr: double,
            squared Hilbert-Schmidt norm of the cross-correlation operator
            between time series 1 and 2, in the RKHS induced by a base kernel

    Notes
    -----
    This is computed as a trace by solving a generalized eigenvalue problem
    equivalent to the one appearing in kernel CCA.
    """
    assert T1 == T2, "the series should be of same duration"
    T = T1
    # define the gram matrices of the series
    K1 = K[:T, :T]
    K2 = K[T:, T:]
    # build right-hand-side symetric matrix of the gen. eigenvalue problem
    A = np.zeros(K.shape)
    K1_K2 = np.dot(K1, K2)
    A[:T, T:] = K1_K2  # upper triangular part
    A[T:, :T] = K1_K2.T  # lower triangular part (symetric)
    # build left-hand-side symetric matrix of the gen. eigenvalue problem
    B = np.zeros(K.shape)
    B[:T, :T] = (1.0 - regul) * np.dot(K1, K1) + regul * K1
    B[T:, T:] = (1.0 - regul) * np.dot(K2, K2) + regul * K2
    # get the eigen-values (w) of Av = wBv (generalized eigenvalue problem)
    tr = float(np.mean(eigvals(A, B, overwrite_a=True)))
    return tr


def distance_autocor_truncated(K, T1, T2, tau=1, regul=1e-3):
    """ Compute the squared HS distance between the autocorrelation operators of
    two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
       between frames kernel matrix (assumed to be centered!)

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 1
         lag, ie time shift used in the auto-covariance computation

    regul: float, optional, default: 1e-3
           regularization parameter for the inverse computation

    Returns
    -------
    dacor: double,
           squared Hilbert-Schmidt norm of the difference between the
           auto-correlation operators, in the RKHS induced by 'frame_kern', of
           the two time series

    Notes
    -----
    Truncated version.
    """
    assert tau <= min(T1 / 2.0, T2 / 2.0), "Too big tau"
    # define the truncated matrices of the non-shifted series
    K1 = K[:T1 - tau, :T1 - tau]
    K2 = K[T1:T1 + T2 - tau, T1:T1 + T2 - tau]
    K12 = K[:T1 - tau, T1:T1 + T2 - tau]
    # define the truncated matrices of the shifted series
    K1tau = K[tau:T1, tau:T1]
    K2tau = K[T1 + tau:, T1 + tau:]
    K12tau = K[tau:T1, T1 + tau:]
    # compute the different terms
    N1 = regul * np.eye(T1 - tau) - solve(
        (T1 - tau) * np.eye(T1 - tau) + 1.0 / regul * K1, K1, sym_pos=True)
    N2 = regul * np.eye(T2 - tau) - solve(
        (T2 - tau) * np.eye(T2 - tau) + 1.0 / regul * K2, K2, sym_pos=True)
    KK1 = np.dot(np.dot(N1.T, K1), np.dot(N1, K1tau))
    KK2 = np.dot(np.dot(N2.T, K2), np.dot(N2, K2tau))
    KK12 = np.dot(np.dot(N1.T, K12), np.dot(N2, K12tau.T))
    # compute the different traces
    tr1 = 1.0 / ((regul ** 4) * (T1 - tau) ** 2) * KK1.trace()
    tr2 = 1.0 / ((regul ** 4) * (T2 - tau) ** 2) * KK2.trace()
    tr12 = 1.0 / ((regul ** 4) * (T1 - tau) * (T2 - tau)) * KK12.trace()
    return tr1 + tr2 - 2.0 * tr12


def distance_autocor_cyclic(K, T1, T2, tau=1, regul=1e-3):
    """ Compute the squared HS distance between the autocorrelation operators of
    two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
       between frames kernel matrix (assumed to be centered!)

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 1
         lag, ie time shift used in the auto-covariance computation

    regul: float, optional, default: 1e-3
           regularization parameter for the inverse computation

    Returns
    -------
    dacor: double,
           squared Hilbert-Schmidt norm of the difference between the
           auto-correlation operators, in the RKHS induced by 'frame_kern', of
           the two time series

    Notes
    -----
    Cyclic version.
    """
    # define per-series tau
    if tau < 0.5:
        # tau as a fraction of series length
        tau1 = max(1, int(T1 * tau + 0.5))
        tau2 = max(1, int(T2 * tau + 0.5))
    elif 1 <= tau < min(T1 / 2.0, T2 / 2.0):
        # constant tau: same for each series
        tau1 = tau2 = int(tau)
    else:
        raise ValueError("Too big tau")
    # define the (non-truncated) matrices of the non-shifted series
    K1 = K[:T1, :T1]
    K2 = K[T1:, T1:]
    K12 = K[:T1, T1:]
    # circular permutation of tau frames
    idxs1 = np.arange(tau1, T1 + tau1) % T1
    idxs2 = np.arange(tau2, T2 + tau2) % T2
    # Note: no need for copy as we re-use the previous centering (indep. of frame order)
    K1tau = K1[np.ix_(idxs1, idxs1)]
    K2tau = K2[np.ix_(idxs2, idxs2)]
    K12tau = K12[np.ix_(idxs1, idxs2)]
    # compute the different terms
    N1 = regul * np.eye(T1) - solve(
        T1 * np.eye(T1) + 1.0 / regul * K1, K1, sym_pos=True)
    N2 = regul * np.eye(T2) - solve(
        T2 * np.eye(T2) + 1.0 / regul * K2, K2, sym_pos=True)
    KK1 = np.dot(np.dot(N1.T, K1), np.dot(N1, K1tau))
    KK2 = np.dot(np.dot(N2.T, K2), np.dot(N2, K2tau))
    KK12 = np.dot(np.dot(N1.T, K12), np.dot(N2, K12tau.T))
    # compute the different traces
    tr1 = 1.0 / ((regul ** 4) * T1 ** 2) * KK1.trace()
    tr2 = 1.0 / ((regul ** 4) * T2 ** 2) * KK2.trace()
    tr12 = 1.0 / ((regul ** 4) * T1 * T2) * KK12.trace()
    # TODO: check if more efficient to use Hadamard products?
    return tr1 + tr2 - 2.0 * tr12


def hsdotprod_autocor_truncated(K, T1, T2, tau=1, regul=1e-3):
    """ Compute the Hilbert-Schmidt inner-product between the autocorrelation
    operators of two time series (**similarity**, not a distance)

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
       between frames kernel matrix (assumed to be centered!)

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 1
         lag, ie time shift used in the auto-covariance computation

    regul: float, optional, default: 1e-3
           regularization parameter for the inverse computation

    Returns
    -------
    hsdp: double,
          Hilbert-Schmidt inner product between the auto-correlation operators,
          in the RKHS induced by 'frame_kern', of the two time series

    Notes
    -----
    Truncated version.
    """
    assert tau <= min(T1 / 2.0, T2 / 2.0), "Too big tau"
    # define the truncated matrices of the non-shifted series
    K1 = K[:T1 - tau, :T1 - tau]
    K2 = K[T1:T1 + T2 - tau, T1:T1 + T2 - tau]
    K12 = K[:T1 - tau, T1:T1 + T2 - tau]
    # define the truncated matrices of the shifted series
    K12tau = K[tau:T1, T1 + tau:]
    # compute the different terms
    N1 = regul * np.eye(T1 - tau) - solve(
        (T1 - tau) * np.eye(T1 - tau) + 1.0 / regul * K1, K1, sym_pos=True)
    N2 = regul * np.eye(T2 - tau) - solve(
        (T2 - tau) * np.eye(T2 - tau) + 1.0 / regul * K2, K2, sym_pos=True)
    KK12 = np.dot(np.dot(N1.T, K12), np.dot(N2, K12tau.T))
    # compute the trace
    hsdp = 1.0 / ((regul ** 4) * (T1 - tau) * (T2 - tau)) * KK12.trace()
    return hsdp


def hsdotprod_autocor_cyclic(K, T1, T2, tau=1, regul=1e-3):
    """ Compute the Hilbert-Schmidt inner-product between the autocorrelation
    operators of two time series (**similarity**, not a distance)

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
       between frames kernel matrix (assumed to be centered!)

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 1
         lag, ie time shift used in the auto-covariance computation

    regul: float, optional, default: 1e-3
           regularization parameter for the inverse computation

    Returns
    -------
    hsdp: double,
          Hilbert-Schmidt inner product between the auto-correlation operators,
          in the RKHS induced by 'frame_kern', of the two time series

    Notes
    -----
    Cyclic version.
    """
    # define per-series tau
    if tau < 0.5:
        # tau as a fraction of series legth
        tau1 = max(1, int(T1 * tau + 0.5))
        tau2 = max(1, int(T2 * tau + 0.5))
    elif 1 <= tau < min(T1 / 2.0, T2 / 2.0):
        # constant tau: same for each series
        tau1 = tau2 = int(tau)
    else:
        raise ValueError("Too big tau")
    # define the (non-truncated) matrices of the non-shifted series
    K1 = K[:T1, :T1]
    K2 = K[T1:, T1:]
    K12 = K[:T1, T1:]
    # circular permutation of tau frames
    idxs1 = np.arange(tau1, T1 + tau1) % T1
    idxs2 = np.arange(tau2, T2 + tau2) % T2
    # Note: no need for copy as we re-use the previous centering (indep. of frame order)
    K12tau = K12[np.ix_(idxs1, idxs2)]
    # compute the different terms
    N1 = regul * np.eye(T1) - solve(
        T1 * np.eye(T1) + 1.0 / regul * K1, K1, sym_pos=True)
    N2 = regul * np.eye(T2) - solve(
        T2 * np.eye(T2) + 1.0 / regul * K2, K2, sym_pos=True)
    KK12 = np.dot(np.dot(N1.T, K12), np.dot(N2, K12tau.T))
    # compute the trace
    hsdp = 1.0 / ((regul ** 4) * T1 * T2) * KK12.trace()
    return hsdp


# auto-regressive-model-based -------------------------------------------------

def distance_predictive_codings(K, T1, T2, tau=1, regul=1e-3):
    """ Compute the squared HS distance between the parameters of AR(p) models
    (in feature space) of two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
       between frames kernel matrix (assumed to be centered!)

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 1
         order of the AR models (use tau past frames)

    regul: float, optional, default: 1e-3
           regularization parameter for the inverse computation

    Returns
    -------
    dpc: double,
         squared Hilbert-Schmidt norm of the difference between the AR(p) models
         learned by kernel ridge regression in the RKHS induced by 'frame_kern'
    """
    p = int(tau)
    assert 1 <= p < min(T1 / 2.0, T2 / 2.0), \
        "Too big p (p=%d >= %d or %d)" % (p, T1 / 2.0, T2 / 2.0)
    K1 = K[:T1, :T1]
    K2 = K[T1:, T1:]
    K12 = K[:T1, T1:]
    # compute the convolutions
    Ip = np.eye(p)
    S1 = correlate2d(K1[:-1, :-1], Ip, mode='valid')
    S2 = correlate2d(K2[:-1, :-1], Ip, mode='valid')
    S21 = correlate2d(K12.T[:-1, :-1], Ip, mode='valid')
    # compute the inverses
    # TODO: rewrite formula better (to replace inv with solve and convolutions by products?)
    Q1 = inv(regul * np.eye(T1 - p) + S1)
    Q2 = inv(regul * np.eye(T2 - p) + S2)
    # compute the product terms
    P1 = np.dot(np.dot(Q1, K1[p:, p:]), np.dot(Q1, S1))
    P2 = np.dot(np.dot(Q2, K2[p:, p:]), np.dot(Q2, S2))
    P12 = np.dot(np.dot(Q1, K12[p:, p:]), np.dot(Q2, S21))
    # compute the different traces
    return 1.0 / T1 * P1.trace() + 1.0 / T2 * P2.trace() - 2.0 / T1 * P12.trace()


def distance_dual_predictive_codings(K, T1, T2, tau=1, regul=1e-3):
    """ Compute the squared HS distance between the dual parameters of AR(p)
    models (in feature space) of two time series

    Parameters
    ----------
    K: (T1+T2) x (T1+T2) array,
       between frames kernel matrix (assumed to be centered!)

    T1: int,
        duration of time series 1

    T2: int,
        duration of time series 2

    tau: int, optional, default: 1
         order of the AR models (use tau past frames)

    regul: float, optional, default: 1e-3
           regularization parameter for the inverse computation

    Returns
    -------
    ddpc: double,
          squared Hilbert-Schmidt norm of the difference between the dual
          parameters of AR(p) models learned by kernel ridge regression in the
          RKHS induced by 'frame_kern'
    """
    p = int(tau)
    assert 1 <= p < min(T1 / 2.0, T2 / 2.0), \
        "Too big p (p=%d >= %d or %d)" % (p, T1 / 2.0, T2 / 2.0)
    K1 = K[:T1, :T1]
    K2 = K[T1:, T1:]
    K12 = K[:T1, T1:]
    # compute the convolutions
    Ip = np.eye(p)
    S1 = correlate2d(K1[:-1, :-1], Ip, mode='valid')
    S2 = correlate2d(K2[:-1, :-1], Ip, mode='valid')
    # compute the inverses
    # XXX incomplete Cholesky would be better but is 3x slower...
    Q1 = inv(regul * np.eye(T1 - p) + S1)
    Q2 = inv(regul * np.eye(T2 - p) + S2)
    # compute the product terms
    P1 = np.dot(np.dot(Q1, K1[p:, p:]), Q1)
    P2 = np.dot(np.dot(Q2, K2[p:, p:]), Q2)
    P12 = np.dot(np.dot(Q1, K12[p:, p:]), Q2)
    # compute the different traces
    return 1.0 / T1 * P1.trace() + 1.0 / T2 * P2.trace() - 2.0 / T1 * P12.trace()


# FOR DEBUG PURPOSES
def distance_hsac_decomp(K, T1, T2, tau=1, mode="truncated"):
    """ Return the components 1/T**2 * (tr1, tr2, tr12) of HSAC

    mode {"truncated"/"cyclic"} defines way to compute HSAC
    """
    assert mode in ["truncated", "cyclic"], "Unknown HSAC mode (%s)" % mode
    assert T1 == T2, "the series should be of same duration"
    assert tau <= T1 / 2.0, "Too big tau"
    T = T1
    if mode == "truncated":
        # define the truncated matrices of the non-shifted series
        K1 = K[:T - tau, :T - tau]
        K2 = K[T:T + T - tau, T:T + T - tau]
        K12 = K[:T - tau, T:T + T - tau]
        # define the truncated matrices of the shifted series
        K1tau = K[tau:T, tau:T]
        K2tau = K[T + tau:, T + tau:]
        K12tau = K[tau:T, T + tau:]
        # normalization factor
        nzf = 1.0 / ((T - tau) * (T - tau))
    elif mode == "cyclic":
        # define the (non-truncated) matrices of the non-shifted series
        K1 = K[:T, :T]
        K2 = K[T:, T:]
        K12 = K[:T, T:]
        # circular permutation of tau frames
        idxs = np.arange(tau, T + tau) % T
        # indexes used to make the permuted views of the kernel matrix
        perm_slice = np.ix_(idxs, idxs)
        # Note: no need for copy as we re-use the previous centering (indep. of frame order)
        K1tau = K1[perm_slice]
        K2tau = K2[perm_slice]
        K12tau = K12[perm_slice]
        # normalization factor
        nzf = 1.0 / (T * T)
    # compute the different traces using Hadamard products
    tr1 = nzf * (K1 * K1tau.T).sum()
    tr2 = nzf * (K2 * K2tau.T).sum()
    tr12 = nzf * (K12 * K12tau.T).sum()  # do not forget the transpose!
    return (tr1, tr2, tr12)


def _get_centered_gram(kern_mat, is_sym=True):
    """ Center (NOT in place) the Gram (kernel) matrix in the feature space

    Mathematical operation: K <- PKP where P = eye(n) - 1/n ones((n,n))

    Parameters
    ----------
    kern_mat: (n,n) symmetric positve semi-definite kernel matrix
    is_sym: boolean (default: True), assume the matrix is symmetric

    Returns
    -------
    cmat: the centered gram matrix
    """
    # number of rows and cols
    nr, nc = kern_mat.shape
    assert not is_sym or nr == nc, "Matrix cannot be symmetric if not square!"
    # mean of the columns of the original matrix (as (nc,) row vector)
    cms = np.mean(kern_mat, 0).reshape((1, nc))
    # mean of the rows (as (nr,1) column vector)
    if is_sym:
        rms = cms.reshape((nr, 1))
    else:
        rms = np.mean(kern_mat, 1).reshape((nr, 1))
    # mean of the means over columns
    mcm = np.mean(cms)  # precomputed once for efficiency
    # return the centered matrix (using array broadcasting)
    return kern_mat + mcm - cms - rms
