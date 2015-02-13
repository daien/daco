"""
TSDist: main callable class to compare time series
==================================================

"""


import numpy as np
from scipy.sparse import hstack, isspmatrix_csr, isspmatrix_csc


from .utils import get_resampled_series, get_subsampled_series, \
    deco_average_tau_dfunc, gauss_blur_image, center_gram


LINEAR_DISTS = ['linear_diff_means']
LINEAR_DISTS += ['linear_mean_diffs']
LINEAR_DISTS += ['linear_allpairs']  # (not a distance but a similarity!)
LINEAR_DISTS += ['linear_hsac']
LINEAR_DISTS += ['linear_nhsac']
LINEAR_DISTS += ['linear_diff_autocor']
LINEAR_DISTS += ['linear_crosscor']  # (not a distance but a similarity!)
LINEAR_DISTS += ['minus_logGAK']  # not linear but input = series not kernel
LINEAR_DISTS += ['linear_autocov_likelihood_ratio']  # (not a distance but a similarity!)
LINEAR_DISTS += ['linear_autocor_likelihood_ratio']  # (not a distance but a similarity!)


RKHS_DISTS = ['mean_elements']
RKHS_DISTS += ['me_squared']
RKHS_DISTS += ['mahalanobis']
RKHS_DISTS += ['aligned_frames_truncated']
RKHS_DISTS += ['aligned_frames_cyclic']
RKHS_DISTS += ['hsac_truncated']
RKHS_DISTS += ['hsac_cyclic']
RKHS_DISTS += ['autocor_truncated']
RKHS_DISTS += ['autocor_cyclic']
RKHS_DISTS += ['predictive_codings']
RKHS_DISTS += ['dual_predictive_codings']
RKHS_DISTS += ['cross_correlation']  # (not a distance but a similarity!)
RKHS_DISTS += ['hsdotprod_autocor_truncated']  # (not a distance but a similarity!)
RKHS_DISTS += ['hsdotprod_autocor_cyclic']  # (not a distance but a similarity!)


def is_sim_f(ts_kname):
    """ Returns True if the TSDist is actually a similarity and not a distance
    """
    return ts_kname in ('linear_allpairs',
                        'linear_crosscor',
                        'cross_correlation',
                        'hsdotprod_autocor_truncated',
                        'hsdotprod_autocor_cyclic')


class TSDist(object):
    """ Callable class used to compute distances between time series

    Parameters
    ----------
    dist_name: string,
               name of the distance type, eventually appended with
               "-<centering mode>", where <centering mode> is nc (no
               centering), gc (global centering), bc (block centering)

    kwargs: dictionary,
            Key-word args used by the distance (eg. 'regul', 'tau') and
        - frame_kern: Kernel object,
                      kernel between time series elements (per-frame BoFS)
                      if not given, then check that the time-series
                      distance specified in "name" operates in the input space
                      (WARNING: frame_kern.gamma is assumed to be
                      previously defined if a RBF kernel is used)
        - resamp: "sub", "down", "up", integer or None
                  resampling parameter used for TS kernels requiring same
                  duration series
        - ma_n: integer,
                moving average radius parameter (in frames)
                NOTE: moving average is performed last (eg after resampling)

    Example usage
    -------------
    >>> dist = TSDist('linear_diff_autocor', tau=1)
    >>> d = dist(bcsc1, bcsc2)

    """
    def __init__(self, dist_name, **kwargs):
        dn_cm = dist_name.split('-')  # distance name [and centering mode]
        if len(dn_cm) == 2:
            self.name = name = dn_cm[0]
            self.centering = dn_cm[1]
            if self.centering not in ('', 'nc', 'gc', 'bc'):
                raise ValueError(
                    "Invalid centering mode in {}".format(dist_name))
        elif len(dn_cm) == 1:
            self.name = name = dist_name
            # use distance-specific centering (may be none) specified below
            self.centering = ''
        else:
            raise ValueError(
                "Invalid distance name format ({})".format(name))
        # base kernel between frames ------------------------------------------
        frame_kern = kwargs.get('frame_kern', None)
        if name in LINEAR_DISTS:
            # don't use a kernel
            self.is_linear = True
        elif name in RKHS_DISTS:
            if not frame_kern:
                raise ValueError(
                    "Must specifiy a base kernel for RKHS distances")
            if frame_kern.use_rbf and (not frame_kern.gamma or frame_kern.gamma <= 0):
                raise ValueError("Undefined base gamma")
            self.is_linear = False
            self.frame_kern = frame_kern
            # do not center (maybe problem with parallel param rewrite?)
            self.frame_kern.center = False
            self.frame_kern.libsvm_fmt = False
        else:
            raise ValueError("Unknown distance name (%s)" % name)
        # define internal distance function (_dfunc) --------------------------
        self.is_sim = False  # True if it is a similarity measure instead of a distance
        self.dparams = {}  # optional parameters for _dfunc
        if name == 'mean_elements':
            from .distances_rkhs import distance_mean_elements
            self._dfunc = distance_mean_elements
        elif name == 'me_squared':
            from .distances_rkhs import distance_me_squared
            self._dfunc = distance_me_squared
        elif name == 'mahalanobis':
            from .distances_rkhs import distance_mahalanobis
            self._dfunc = distance_mahalanobis
            # optional params for dfunc (if absent, use defaults)
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
        elif name == 'aligned_frames_truncated':
            from .distances_rkhs import distance_aligned_frames_truncated
            self._dfunc = distance_aligned_frames_truncated
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
        elif name == 'aligned_frames_cyclic':
            from .distances_rkhs import distance_aligned_frames_cyclic
            self._dfunc = distance_aligned_frames_cyclic
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
        elif name == 'hsac_truncated':
            from .distances_rkhs import distance_hsac_truncated
            self._dfunc = distance_hsac_truncated
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
        elif name == 'hsac_cyclic':
            from .distances_rkhs import distance_hsac_cyclic
            self._dfunc = distance_hsac_cyclic
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
        elif name == 'hsac_decomp':
            # FOR DEBUG PURPOSES only
            from .distances_rkhs import distance_hsac_decomp
            self._dfunc = distance_hsac_decomp
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'mode' in kwargs:
                self.dparams['mode'] = kwargs['mode']
        elif name == 'autocor_truncated':
            from .distances_rkhs import distance_autocor_truncated
            self._dfunc = distance_autocor_truncated
            if not self.centering:
                # series assumed stationary so mean = mean of shifted version
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
        elif name == 'autocor_cyclic':
            from .distances_rkhs import distance_autocor_cyclic
            self._dfunc = distance_autocor_cyclic
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
        elif name == 'predictive_codings':
            from .distances_rkhs import distance_predictive_codings
            self._dfunc = distance_predictive_codings
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                # use tau as order of the Auto-Regressive model
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
            # disable annoying useless warnings cause by correlate2d
            import warnings
            warnings.simplefilter('ignore', category=np.ComplexWarning)
        elif name == 'dual_predictive_codings':
            from .distances_rkhs import distance_dual_predictive_codings
            self._dfunc = distance_dual_predictive_codings
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                # use tau as order of the Auto-Regressive model
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
            # disable annoying useless warnings cause by correlate2d
            import warnings
            warnings.simplefilter('ignore', category=np.ComplexWarning)
        elif name == 'cross_correlation':
            # a SIMILARITY, not a distance!
            from .distances_rkhs import hsnorm_cross_correlation
            self._dfunc = hsnorm_cross_correlation
            self.is_sim = True
            if not self.centering:
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
        elif name == 'hsdotprod_autocor_truncated':
            from .distances_rkhs import hsdotprod_autocor_truncated
            self._dfunc = hsdotprod_autocor_truncated
            self.is_sim = True
            if not self.centering:
                # series assumed stationary so mean = mean of shifted version
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
        elif name == 'hsdotprod_autocor_cyclic':
            from .distances_rkhs import hsdotprod_autocor_cyclic
            self._dfunc = hsdotprod_autocor_cyclic
            self.is_sim = True
            if not self.centering:
                # series assumed stationary so mean = mean of shifted version
                self.centering = 'bc'
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
        elif name == 'linear_diff_means':
            from .distances_linear import linear_diff_means
            self._dfunc = linear_diff_means
        elif name == 'linear_mean_diffs':
            from .distances_linear import linear_mean_diffs
            self._dfunc = linear_mean_diffs
        elif name == 'linear_allpairs':
            from .distances_linear import linear_allpairs
            # a SIMILARITY, not a distance!
            self.is_sim = True
            self._dfunc = linear_allpairs
        elif name == 'linear_hsac':
            from .distances_linear import linear_hsac
            self._dfunc = linear_hsac
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'mntype' in kwargs:
                self.dparams['mntype'] = kwargs['mntype']
        elif name == 'linear_nhsac':
            from .distances_linear import linear_nhsac
            self._dfunc = linear_nhsac
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
            if 'check_regul' in kwargs:
                self.dparams['check_regul'] = kwargs['check_regul']
            if 'mntype' in kwargs:
                self.dparams['mntype'] = kwargs['mntype']
        elif name == 'linear_diff_autocor':
            from .distances_linear import linear_diff_autocor
            self._dfunc = linear_diff_autocor
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
            if 'mntype' in kwargs:
                self.dparams['mntype'] = kwargs['mntype']
        elif name == 'linear_crosscor':
            from .distances_linear import linear_crosscor
            # a SIMILARITY, not a distance!
            self.is_sim = True
            self._dfunc = linear_crosscor
            # optional params for dfunc (if absent, use defaults)
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']
            if 'check_regul' in kwargs:
                self.dparams['check_regul'] = kwargs['check_regul']
            if 'mntype' in kwargs:
                self.dparams['mntype'] = kwargs['mntype']
        elif name == 'minus_logGAK':
            # return -logGAK kernel and don't flag as similarity to use
            # exp(-gamma*(-logGAK)) as in M. Cuturi's paper
            from .distances_linear import minus_logGAK
            self._dfunc = minus_logGAK
            # optional params for dfunc (if absent, use defaults)
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']  # actually 'sigma'
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']  # actually 'triangular'
        elif name == 'linear_autocov_likelihood_ratio':
            from .distances_linear import linear_autocov_likelihood_ratio
            # a SIMILARITY, not a distance!
            self.is_sim = True
            self._dfunc = linear_autocov_likelihood_ratio
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
        elif name == 'linear_autocor_likelihood_ratio':
            from .distances_linear import linear_autocor_likelihood_ratio
            # a SIMILARITY, not a distance!
            self.is_sim = True
            self._dfunc = linear_autocor_likelihood_ratio
            # optional params for dfunc (if absent, use defaults)
            if 'tau' in kwargs:
                self.dparams['tau'] = kwargs['tau']
            if 'regul' in kwargs:
                self.dparams['regul'] = kwargs['regul']  # actually 'sigma'
        else:
            raise ValueError("Unknown distance name (%s)" % name)
        # check args
        if not self.is_linear and self.dparams.get('regul', 1) <= 0:
            raise ValueError(
                "Invalid regularization ({})".format(kwargs['regul']))
        # define pre-processing function --------------------------------------
        self.resamp = kwargs.get('resamp', None)
        if self.resamp:
            # pre-process time series to have same duration
            if self.resamp == 'sub':
                self._preproc = get_subsampled_series
                self.pparams = {}
            else:
                self._preproc = get_resampled_series
                self.pparams = {'dur': self.resamp}
        else:
            # no preprocessing of time series (if not necessary or already done)
            self._preproc = None
        # define base kernel function -----------------------------------------
        if not self.is_linear:
            if kwargs.get('ma_n', 0) > 0:
                # use moving average version
                # NOTE: moving average is performed last (after resampling)
                self.ma_n = kwargs['ma_n']
                self._get_K = self.__get_frames_K_MA__
            else:
                if self.centering == 'bc':
                    self._get_K = self.__get_frames_block_centered_K__
                elif self.centering == 'gc':
                    self._get_K = self.__get_frames_global_centered_K__
                elif self.centering in ('', 'nc'):
                    self._get_K = self.__get_frames_K__
                else:
                    raise ValueError(
                        "Bug: invalid centering mode ({})".format(
                            self.centering))
        # average over tau's ? ------------------------------------------------
        if hasattr(self.dparams.get('tau', None), '__iter__'):
            # list of taus: sum the distances for all taus
            self._dfunc = deco_average_tau_dfunc(self._dfunc)  # decorator
        elif 'tau' in self.dparams:
            # make sure it is an int
            self.dparams['tau'] = int(self.dparams['tau'])

    def __call__(self, bcsc1, bcsc2):
        """ Compute a RKHS distance between two time series of BoFs

        Parameters
        ----------
        bcsc1: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 1

        bcsc2: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 2

        Returns
        -------
        dval: double,
              the distance between the two time-series
        """
        # check CSC format
        if isspmatrix_csr(bcsc1):
            bcsc1 = bcsc1.T
        if isspmatrix_csr(bcsc2):
            bcsc2 = bcsc2.T
        assert isspmatrix_csc(bcsc1), "bcsc1 is not a CSC matrix"
        assert isspmatrix_csc(bcsc2), "bcsc2 is not a CSC matrix"
        # eventually, pre-process the bofs
        if self._preproc:
            bcsc1, bcsc2 = self._preproc(bcsc1, bcsc2, **self.pparams)
        # compute the distance between time series
        if self.is_linear:
            dval = self._dfunc(bcsc1, bcsc2, **self.dparams)
        else:
            # distance in RKHS induced by frame_kern
            K, T1, T2 = self._get_K(bcsc1, bcsc2)
            dval = self._dfunc(K, T1, T2, **self.dparams)
        return dval

    def __get_frames_K__(self, bcsc1, bcsc2):
        """ Compute the kernel matrix between the frames of two time series

        Parameters
        ----------
        bcsc1: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 1
        bcsc2: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 2

        Returns
        -------
        K, T1, T2: kernel matrix and the dimension of the two time series
        """
        K = self.frame_kern.gram(
            hstack([bcsc1, bcsc2], format="csc", dtype=np.double))
        T1 = bcsc1.shape[1]
        T2 = bcsc2.shape[1]
        return K, T1, T2

    def __get_frames_global_centered_K__(self, bcsc1, bcsc2):
        """ Compute the kernel matrix between the frames of two time series
        and center it as a whole

        Parameters
        ----------
        bcsc1: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 1

        bcsc2: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 2

        Returns
        -------
        K, T1, T2: kernel matrix and the dimension of the two time series

        Note
        ----
        This is just for investigation purposes: use block centering in general.
        """
        K = self.frame_kern.gram(hstack([bcsc1, bcsc2], format="csc", dtype=np.double))
        center_gram(K)
        T1 = bcsc1.shape[1]
        T2 = bcsc2.shape[1]
        return K, T1, T2

    def __get_frames_block_centered_K__(self, bcsc1, bcsc2):
        """ Compute the kernel matrix between the frames of two time series
        and center each block (K11, K12, K21, K22) approprietly.

        Parameters
        ----------
        bcsc1: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 1

        bcsc2: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 2

        Returns
        -------
        K, T1, T2: kernel matrix and the dimension of the two time series

        Notes
        -----
        It is similar to computing the kernel between the centered (in feature
        space) time series.
        """
        K = self.frame_kern.gram(hstack([bcsc1, bcsc2], format="csc", dtype=np.double))
        T1 = bcsc1.shape[1]
        T2 = bcsc2.shape[1]
        center_gram(K[:T1, :T1])
        center_gram(K[T1:, T1:])
        center_gram(K[:T1, T1:], is_sym=False)
        center_gram(K[T1:, :T1], is_sym=False)
        return K, T1, T2

    # TODO: BUG => smoothing across series !?
    def __get_frames_K_MA__(self, bcsc1, bcsc2):
        """ Compute the kernel matrix between the Moving Average of the frames
        of two time series

        Parameters
        ----------
        bcsc1: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 1
        bcsc2: sparse.csc_matrix object,
               contains the sparse column wise representation of time series 2

        Returns
        -------
        K, T1, T2: kernel matrix and the dimension of the two time series
        """
        K, T1, T2 = self.__get_frames_K__(bcsc1, bcsc2)
        K = gauss_blur_image(K, n=self.ma_n)
        T1 = T1 - self.ma_n
        T2 = T2 - self.ma_n
        return K, T1, T2
