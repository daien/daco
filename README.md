# Distances between time series of sparse vectors

Various distance functions between time series of sparse vectors:
    * either directly in the input space (`distances_linear.py`): `dist(bcsc1, bcsc2, **kwargs)`
    * or in the RKHS induced by a kernel between vectors (`distances_rkhs.py`): `dist(K, T1, T2, **kwargs)`

Includes the Difference Between Auto-Correlation Operators (DACO) proposed in the paper:

    A time series kernel for action recognition
    Adrien Gaidon, Zaid Harchaoui, Cordelia Schmid,
    BMVC, 2013
    https://hal.inria.fr/inria-00613089v2


## Author

Adrien Gaidon

## License

MIT, except for the global alignment kernel (cf. README-logGAK.txt).
