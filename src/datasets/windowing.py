import numpy as np
from sklearn.preprocessing import RobustScaler
import warnings
import sys
from scipy.stats import spearmanr

def rolling_apply_1d(fun, X, window_size):
    r = np.empty(X.shape)
    r.fill(np.nan)
    for i in range(window_size-1, X.shape[0]):
        r[i] = fun(X[(i-window_size+1):i+1])
    
    return r


def rolling_apply_cov(fun, X, window_size, b_scaler=True):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            cov_matrix = list()
            r = np.empty((X.shape[-1], X.shape[-1]))
            r.fill(np.nan)
            for k in range(window_size - 1):
                if k < window_size:
                    cov_matrix.append(r)

            for i in range(window_size - 1, X.shape[0]):
                    # cov_matrix.append((fun(X[(i - window_size + 1):i + 1])).tolist())
                    cov_matrix.append(fun(X[(i - window_size + 1):i + 1], b_scaler))
                    sys.stdout.write('\r>> [%d/%d days] correlation matrix calculation....!!!' % (i, X.shape[0]))
                    sys.stdout.flush()

            cov_matrix = np.array(cov_matrix, dtype=np.float32)
        except Warning:
            assert False, 'check !!!'
    return cov_matrix

def rolling_apply_cross_cov(fun, X, Y, window_size):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            r = np.empty([window_size - 1, 2, 2])
            r.fill(np.nan)
            cov_matrix = r.tolist()

            for i in range(window_size - 1, X.shape[0]):
                cov_matrix.append(fun(X[(i - window_size + 1) : i + 1], Y[(i - window_size + 1) : i + 1]))
            cov_matrix = np.array(cov_matrix, dtype=np.float32)
        except Warning:
            assert False, 'check !!!'
    return cov_matrix


def rolling_apply(fun, X, window_size):
    r = np.empty(X.shape)
    r.fill(np.nan)

    if X.ndim == 1:
        r = rolling_apply_1d(fun, X, window_size)
        sys.stdout.write('\r>> [1/1] rolling_apply....!!!')
        sys.stdout.flush()
    else:
        _, cnt = X.shape
        for i in range(cnt):
            r[:, i] = rolling_apply_1d(fun, X[:, i], window_size)
            sys.stdout.write('\r>> [%d/%d] rolling_apply....!!!' % (i, cnt))
            sys.stdout.flush()
    
    return r

def fun_sum(X):
    return np.nansum(X)

def fun_cumsum(X):
    return np.nancumsum(X)[-1]

def fun_mean(X):
    return np.nanmean(np.array(X, dtype=np.float))

#  caution: correlation-matrix adopted
def fun_cov(X, b_scaler=True):
    if b_scaler:
        return np.corrcoef(RobustScaler().fit_transform(X), rowvar=False)
    else:
        return np.corrcoef(X, rowvar=False)

def fun_cross_cov(X, Y):
    return np.corrcoef(X, Y, rowvar=False)

def fun_cov_spearman(X):
    # need extra dimension manipulation
    cor, _ = spearmanr(RobustScaler().fit_transform(X), axis=0)
    return cor

def rolling_window(X, window):
    shape = X.shape[:-1] + (X.shape[-1] - window + 1, window)
    strides = X.strides + (X.strides[-1],)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)



"""Unit test

"""
def ut_rolling_apply_1d():
    X = np.array([1,2,3,4,5])
    print(rolling_apply_1d(fun_sum, X, 3))

def ut_rolling_apply():
    X = np.array([[2,3,4,5,6,7,7,7],
        [0,1,-2,3,4,2,3,2], 
        [1,2,3,4,5,0,0,2],
        [2,3,4,5,6,7,7,7],
        [1,2,3,4,5,0,0,2],
        [2,3,4,5,6,7,7,7],
        [0,1,-2,3,4,2,3,2], 
        [0,1,-2,3,4,2,3,2], 
        [1,2,3,4,5,0,0,2]])
    print(X)
    print(rolling_apply(fun_cumsum, X, 5))


def ut_rolling_window():
    X = np.arange(50).reshape((10,5))
    print(rolling_window(X, 3))


def ut_rolling_cov():
    X = np.random.random((6, 3))
    print(rolling_apply_cov(fun_cov, X, 2))

# ut_rolling_cov()

