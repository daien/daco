""" Utility functions for distances between time series
"""


import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import solve, eigh
from scipy.signal import convolve, resample


def center_gram(kern_mat, is_sym=True):
    """ Center (in place) the Gram (kernel) matrix in the feature space

    Mathematical operation: K <- PKP where P = eye(n) - 1/n ones((n,n))

    Parameters
    ----------
    kern_mat: (nr, nc) numpy array,
              positve semi-definite kernel matrix

    is_sym: boolean, optional, default: True,
            assume the matrix is symmetric

    Returns
    -------
    cms: (1, nc) numpy array,
         column means of the original kernel matrix

    mcm: double,
         mean of the original column means, which, like cms, are parameters
         needed to center in the same way the future kernel evaluations

    """
    # number of rows and cols
    nr, nc = kern_mat.shape
    assert not is_sym or nr == nc, "Matrix cannot be symmetric if not square!"
    # mean of the columns of the original matrix (as (1, nc) row vector)
    cms = np.mean(kern_mat, 0)[np.newaxis, :]
    # mean of the rows (as (nr, 1) column vector)
    if is_sym:
        rms = cms.T
    else:
        rms = np.mean(kern_mat, 1)[:, np.newaxis]
    # mean of the means over columns (mean of the full matrix)
    mcm = np.mean(cms)
    # center the matrix (using array broadcasting)
    kern_mat += mcm
    kern_mat -= cms
    kern_mat -= rms
    return cms, mcm


def center_rows(kern_rows, cms, mcm):
    """ Center (in place) a kernel row in the feature space

    WARNING: assumes kernel row NOT IN LIBSVM FORMAT!

    Parameters
    ----------
    kern_rows: (m, n) numpy array,
               rows of kernel evaluations k(x,x_i) of m test samples x
               with a (training) set {x_i}, i=1...n

    cms: (1, nc) numpy array,
         column means of the original kernel matrix

    mcm: double,
         mean of the original column means
    """
    if kern_rows.ndim == 2:
        # multiple rows at once
        rows_mean = np.mean(kern_rows, axis=-1)[:, np.newaxis]
    else:
        # only one row: 1D vector
        rows_mean = np.mean(kern_rows)
        cms = cms.squeeze()  # to broadcast correctly
    kern_rows += mcm
    kern_rows -= cms
    kern_rows -= rows_mean


def compute_autocov(bcsc, tau=1):
    """ Return the auto-covariance matrix at lag tau

    Cyclic version
    """
    d, T = bcsc.shape  # d: number of vars, T: number of observations
    assert tau < T / 2, "Too big tau"
    b = bcsc.toarray()
    # center (row-wise: center per-variable observations)
    b -= b.mean(axis=1)[:, np.newaxis]
    # circular permutation of tau frames
    # changing only the order of observations: no need to recenter
    btau = np.roll(b, int(tau), axis=-1)
    # auto-correlation: cross-correlation between b and btau
    return np.dot(b, btau.T) / (T - 1.0)


def compute_autocorr(bcsc, tau=1, regul=1e-3):
    """ Return the auto-correlation matrix at lag tau

    auto-correlation: auto-covariance normalized by variance

    Cyclic version
    """
    acv = compute_autocov(bcsc, tau=tau)
    var = np.cov(bcsc.toarray())
    if regul < 0:
        d = bcsc.shape[0]  # number of vars
        used_regul = 0.5 * eigh(var, eigvals_only=True,
                                eigvals=(d - 1, d - 1))[0]
    else:
        used_regul = regul
    var += used_regul * np.eye(var.shape[0])
    return solve(var, acv)


def get_subsampled_series(bcsc1, bcsc2):
    """ Return two subsampled views of same duration

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
           contains the sparse column wise representation of time series 1

    bcsc2: sparse.csc_matrix object,
           contains the sparse column wise representation of time series 2

    Returns
    -------
    sbcsc1: sparse.csc_matrix object,
            subsampled version of bcsc1

    sbcsc2: sparse.csc_matrix object,
            subsampled version of bcsc2

    Notes
    -----
    The longest time series is down-sampled in a 1-out-of-N way, to match the
    duration of the shortest time series.
    """
    T1 = bcsc1.shape[1]
    T2 = bcsc2.shape[1]
    # compute the subsampled time series
    if T1 > T2:
        # subsample T1
        ratio = float(T1) / T2
        sidx1 = [int(round(i * ratio)) for i in range(T2)]
        sbcsc1 = bcsc1[:, sidx1]
        sbcsc2 = bcsc2
    elif T1 < T2:
        # subsample T2
        ratio = float(T2) / T1
        sbcsc1 = bcsc1
        sidx2 = [int(round(i * ratio)) for i in range(T1)]
        sbcsc2 = bcsc2[:, sidx2]
    else:
        # both clips are already of the same duration
        sbcsc1 = bcsc1
        sbcsc2 = bcsc2
    return sbcsc1, sbcsc2


def resample_ts(bcsc, dest_T):
    """ Resample a time series to a duration 'dest_T'

    Resampling (down or up) is done using Fourier method (fft and ifft)

    Parameters
    ----------
    bcsc: (d, T) sparse.csc_matrix object,
          contains the sparse column wise representation of a time series
    dest_T: int,
            output duration in frames of the time series obtained by resampling

    Returns
    -------
    rbcsc: (d, dest_T) resampled time series

    See also
    --------
    scipy.signal.resample
    """
    T = bcsc.shape[1]
    if T == dest_T:
        rbcsc = bcsc
    else:
        rbcsc = csc_matrix(resample(bcsc.toarray(), dest_T, axis=1))
    return rbcsc


def get_resampled_series(bcsc1, bcsc2, dur="up"):
    """ Resample two time series so that they are of the same duration

    Parameters
    ----------
    bcsc1: sparse.csc_matrix object,
           contains the sparse column wise representation of time series 1
    bcsc2: sparse.csc_matrix object,
           contains the sparse column wise representation of time series 2
    dur: string or int,
         if dur == "up": up-sample shortest series to duration of the longest
         if dur == "down": down-sample longest to shortest
         if dur is an integer, then resample both series to this duration

    Returns
    -------
    (rbcsc1, rbcsc2): two sparse.csc_matrix objects,
                      the resampled series

    """
    T1 = bcsc1.shape[1]
    T2 = bcsc2.shape[1]
    if isinstance(dur, int) and dur > 0:
        used_dur = dur
    elif dur == "up":
        used_dur = max(T1, T2)
    elif dur == "down":
        used_dur = min(T1, T2)
    else:
        raise ValueError("Unknown resampling scheme (%s)" % dur)
    # compute the re-sampled time series
    rbcsc1 = resample_ts(bcsc1, used_dur)
    rbcsc2 = resample_ts(bcsc2, used_dur)
    return rbcsc1, rbcsc2


def gauss_kern(size):
    """ Returns a normalized 2D Gaussian kernel array for convolutions
    """
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)))
    return g / g.sum()


def gauss_blur_image(im, n=4):
    """ Blurs an image by convolving with a gaussian kernel of typical size n.

    Used for Moving Average

    Note: the border (of width 'n') is cropped out, so if im.shape == h, w,
        the result is an image of shape = h-2n, w-2n
    """
    g = gauss_kern(n)
    improc = convolve(im, g, mode='valid')  # 'valid' => crop out border
    return improc


def deco_average_tau_dfunc(dfunc):
    """ Decorator returning a function averaging the distances over taus
    """
    def new_dfunc(*args, **kwargs):
        # list of taus
        taus = kwargs['tau']
        # get the per-tau key-word args
        kws = [0] * len(taus)
        for i, tau in enumerate(taus):
            kws[i] = kwargs.copy()
            kws[i]['tau'] = int(tau)  # only one tau at a time
        # return the average distance over taus
        return 1.0 / len(taus) * sum(dfunc(*args, **kw) for kw in kws)
    # return the new distance function
    return new_dfunc
