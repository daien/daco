"""
Kernel PCA on time series
=========================
"""


import numpy as np
from bisect import bisect_left
from scipy.sparse import csc_matrix, hstack

from ekovof.utils import get_kernel_object, kpca, kpca_proj


def kpca_proj_features(bss, pca_kn, ndim=-1, gamma=-1, ntr=-1, max_frs=1e4):
    """ Performs kernel PCA on the features in bss

    Parameters
    ----------
    bss: list of per sample (voc_dim, n_frs) CSC matrices

    pca_kn: string, kernel name

    ndim: int, optional, default: -1,
          output projection dimension, use all dimensions if ndim <= 0

    gamma: float, optional, default: -1,
           gamma parameter of kernel if needed (default: automatic tuning)

    ntr: int, optional, default: -1,
         number of training samples, ie use only bss[:Ntr] to compute kPCA
         use all if ntr <= 0

    max_frs: int, optional, default: 1e4,
             If there are more than max_frs frames in bss, then first perform
             kPCA on a subsample of the series, then project the others using
             the found principal components.

    Returns
    -------
    proj_error: float,
                projection error

    nevects: array of normalized principal eigen-vectors (used to project new data)

    pbss: list of projected features (CSC columns of dimension ndim)

    Notes
    -----

    """
    if ntr <= 0:
        ntr = len(bss)
    # define the kernel between frames
    pca_kern = get_kernel_object(
        pca_kn, gamma=gamma, center=True, num_threads=0)
    # if too large number of frames: subsample to compute PCA, then project the rest
    tot_nfrs = sum(b.shape[1] for b in bss[:ntr])
    if tot_nfrs <= max_frs:
        # use all training series
        sel_idxs = range(ntr)
    else:
        np.random.seed(1)  # for DEBUG
        # randomly select frames from the training series
        rand_sidxs = np.random.permutation(
            ntr)  # randomly ordered training series indexes
        cfrs = np.cumsum([bss[i].shape[1] for i in rand_sidxs])
        li = bisect_left(cfrs, max_frs)
        sel_idxs = rand_sidxs[:li]
    # stack the selected frames
    tr_csc_m = hstack(
        [bss[i] for i in sel_idxs], format="csc", dtype=np.double)
    # compute the Gram matrix
    kgram = pca_kern.gram(tr_csc_m)
    if ndim <= 0 or ndim >= kgram.shape[0]:
        ndim = kgram.shape[0] - 1
    # compute the principal components in the feature space
    evals, nevects, projs = kpca(kgram, k=ndim)
    # compute the projection error on the "training" part
    proj_error = 1.0 / len(bss) * (kgram.trace() - evals.sum())
    # compute the projections for all series with sparse matrix format (required later)
    pbss = [0.0] * len(bss)
    # include already projected series
    prev_et = 0
    for i in sel_idxs:
        pbss[i] = csc_matrix(projs[:, prev_et:prev_et + bss[i].shape[1]],
                             dtype=np.double)
        prev_et = prev_et + bss[i].shape[1]
    # project the rest
    if ntr < len(bss) or tot_nfrs > max_frs:
        nsel_idxs = sorted(
            set(range(ntr)).difference(sel_idxs)) + range(ntr, len(bss))
        for i in nsel_idxs:
            K_rows = pca_kern.m2m(bss[i], tr_csc_m)
            proj_cols = kpca_proj(K_rows, nevects)
            pbss[i] = csc_matrix(proj_cols, dtype=np.double)
    return proj_error, nevects, pbss
