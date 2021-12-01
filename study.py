#!/usr/bin/env python
#coding:utf-8

import os
import time
import bcolz
import numpy as np

import torch
import torch.nn as nn

here = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(here, 'ma250', '000001.XSHG')
datepath = os.path.join(rootpath, 'date')
ohlcpath = os.path.join(rootpath, 'ohlcs')

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def softplus(x):
    beta = 1/1000.
    return  np.log10(1 + np.exp(x*beta))/beta


@timeit
def read_data():
    start = 250
    date_ar = bcolz.open(datepath)[start:]
    ohlc_ar = bcolz.open(ohlcpath)[start:]
    assert len(date_ar) == len(ohlc_ar)
    n = len(date_ar)
    X = []
    Y = []

    names = ['close'] + ['ma%s' % i for i in range(5, 255, 5)]
    _max_price = np.max([np.max(ohlc_ar[_name]) for _name in names])
    _min_price = np.min([np.min(ohlc_ar[_name]) for _name in names])
    for _name in names:
        ohlc_ar[_name] = (ohlc_ar[_name] - _min_price)/(_max_price - _min_price)


    for i in range(249, n):
        if i + 5 >= n:
            break
        loc_index = [i] + list(range(i - 4, i - 4 - 250, -5))
        one_x = np.array(ohlc_ar[loc_index][names].tolist()).flatten()
        one_y = ohlc_ar[i + 5]['ma5']

      #  one_x = 1.0 / (1 + np.exp(-one_x))
      #  one_y = 1.0 / (1 + np.exp(-one_y))
      #  one_y = softplus(one_y)

        # print(one_x, one_y)
        X.append(one_x)
        Y.append(one_y)

    X = np.array(X)
    Y = np.array([Y]).T
    return X, Y

@timeit
def main():
    x, y = read_data()
    print(x.shape, y.shape)
    np.random.seed(1)
    weights = np.random.random((len(x[0]), 1))
    print(x)
    print(weights)
    print(weights.shape)
    for _ in range(10):
        z = np.dot(x, weights)
        print(z)
        output = 1.0/(1 + np.exp(-z))
       # output = z
       # print(z)
       # output = softplus(z)
        error = y - output
        print(max(error), min(error))
        print(output)
        slope = output * (1 - output)
        delta = error * slope
       # print(delta)
        weights = weights + np.dot(x.T, delta)

        # print(weights.shape)
    print(weights)







if __name__ == "__main__":
    main()
