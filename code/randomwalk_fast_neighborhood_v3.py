import numpy as np
from scipy import optimize as op
import random
import time

import os
import threading


class Order:
    def __init__(self, frequency, cardinality, itemset):
        self.frequency = frequency
        self.cardinality = cardinality
        self.itemset = itemset


def getset(n, k):
    ret = set()
    for i in range(k):
        while True:
            r = random.randint(1, n)
            if r not in ret:
                break
        ret.add(r)
    return ret


def cal_fulfillment_rate(s, orderlist):
    total = 0.
    fill = 0.

    for order in orderlist:
        total += order.frequency
        pd = True
        for item in order.itemset:
            if item not in s:
                pd = False
                break
        if pd:
            fill += order.frequency
    return 1. * fill / total


def sigmoid(x, mode='cp'):
    if mode == 'cp':
        import cupy as tp
    elif mode == 'np':
        import numpy as tp

    s = 1 / (1 + tp.exp(-x))
    return s


def derivative_sigmoid(x, mode='cp'):
    return sigmoid(x, mode) * (1 - sigmoid(x, mode))


def get_trans(curlist, param, neighborhood, mode='cp'):
    if mode == 'cp':
        import cupy as tp
    elif mode == 'np':
        import numpy as tp

    X = param['X'] # shape: (n + 1) * d
    x_t = param['x_t'] # shape: n
    n = param['n']

    tp_curlist = tp.array(curlist)
    g = X[tp_curlist * (tp_curlist <= n)].dot(X.T)

    mx = tp.max(g, axis=1, keepdims=True,
                where=neighborhood[tp_curlist * (tp_curlist <= n)].astype(bool), initial=-1e5)
    mx = np.maximum(mx, x_t[tp_curlist * (tp_curlist <= n)].reshape(len(curlist), 1))

    g = g * neighborhood[tp_curlist * (tp_curlist <= n)]
    g -= mx
    # print(tp.max(g))
    g = tp.exp(g)
    g = g * neighborhood[tp_curlist * (tp_curlist <= n)]

    # tmp = tp.zeros((len(curlist), 1))
    tmp = x_t[tp_curlist * (tp_curlist <= n)].reshape(len(curlist), 1)
    tmp -= mx
    tmp = tp.exp(tmp)
    g = tp.concatenate((g, tmp), axis=1)


    w = np.ones(g.shape)
    for i in range(len(curlist)):
        if curlist[i] == n + 1:
            w[i, :] = 0
            w[i, n + 1] = 1.
        elif curlist[i] == 0:
            w[i, n + 1] = 0.
    if mode == 'cp':
        w = tp.array(w)
    g = g * w

    g /= tp.sum(g, axis=1, keepdims=True)

    return g


def sample(param, neighborhood, cardinality_constraint):
    n = param['n']
    s = set()
    cur = 0
    g = get_trans([0], param, neighborhood, 'np')
    # print(g.shape, type(g))
    g = g.reshape((n + 2))
    cur = np.random.choice(a=n + 2, p=g)
    s.add(cur)

    while True:
        g = get_trans([cur], param, neighborhood, 'np')
        g = g.reshape((n + 2))
        cur = np.random.choice(a=n + 2, p=g)
        if cur == n + 1:
            break
        s.add(cur)
        # print(cur)
        if len(s) > cardinality_constraint:
            break
    return s


def cal_prob_grad(s, param, neighborhood):
    X = param['X']
    x_t = param['x_t']
    n = param['n']

    l = list(s)
    l = [0] + l + [n + 1]
    size = len(s)

    # cu_G = get_trans(l, param, True, 'cp')
    # np_G = cp.asnumpy(cu_G)
    np_G = get_trans(l, param, neighborhood, 'np')

    # t1 = time.time()
    A = np.zeros((size + 3, size + 3))
    b = np.zeros(size + 3)
    A[size + 1, size + 1] = 1.
    b[size + 1] = 1.
    A[size + 2, size + 2] = 1.
    b[size + 2] = 0

    np_coe = np.zeros((1 << size))
    np_Mask = np.zeros((1 << size, size + 3))
    np_Mask[:, size + 1] = 1.
    np_Mask[:, size + 2] = 1.
    np_Mask[:, 0] = 1.
    # print(size)
    for i in range(1 << size):
        t = i
        count = 0
        for j in range(size):
            np_Mask[i, j + 1] = t % 2
            count = count + t % 2
            t = t >> 1
        np_coe[i] = pow(-1, (size - count) % 2)

    # cp_coe = cp.array(np_coe)
    # cp_Mask = cp.array(np_Mask)

    np_subset_A = np.zeros((1 << size, size + 3, size + 3))
    np_subset_A[:, size + 1, size + 1] = 1.
    np_subset_A[:, size + 2, size + 2] = 1.

    for i in range(size + 1):
        # g = get_trans(l[i], param, False)
        # g = cp.asnumpy(g)
        g = np_G[i]

        tmp = 0.
        for j in range(size + 2):
            tmp += g[l[j]]
            A[i, j] -= g[l[j]]
        A[i, size + 2] -= 1. - tmp
        A[i, i] += 1

        np_subset_A[:, i, :] += np_Mask[:, i:i + 1].dot(A[i:i + 1, :])
        np_subset_A[:, i, i] += 1 - np_Mask[:, i]

    np_subset_inv_A = np.linalg.inv(np_subset_A)

    np_subset_h = np_subset_inv_A.dot(b)

    GradX = np.zeros(X.shape)
    Gradx_t = np.zeros(x_t.shape)

    for i in range(size + 1):
        # g = get_trans(l[i], param, False)
        g = np_G[i]
        g = g[:n + 1]

        GradX[l[1:size + 1]] += ((np_coe * np_subset_inv_A[:, 0, i] * np_Mask[:, i]).dot(
            np_subset_h[:, 1:size + 1] * g[l[1:size + 1]])).reshape((size, 1)) * \
                                X[l[i]].reshape((1, X.shape[1]))

        GradX[l[i]] += ((np_coe * np_subset_inv_A[:, 0, i] * np_Mask[:, i]).dot(
            ((g[l[1:size + 1]] * np_subset_h[:, 1:size + 1]).dot(X[l[1:size + 1]]))))

        t = g.dot(X)
        GradX[l[i]] -= np.sum(np_coe * np_subset_inv_A[:, 0, i] *
                              np_Mask[:, i] *
                              (np_subset_h[:, 1: size + 1].dot(g[l[1:size + 1]]))) * t

        GradX -= np.sum(np_coe * np_subset_inv_A[:, 0, i] *
                        np_Mask[:, i] *
                        (np_subset_h[:, 1:size + 1].dot(g[l[1:size + 1]]))) * \
                 (g.reshape((n + 1, 1)) * X[l[i]:l[i] + 1, :]).reshape(X.shape)

        if i > 0:
            GradX -= np.sum(np_coe * np_subset_inv_A[:, 0, i] *
                            np_Mask[:, i] * np_subset_h[:, size + 1] * np_G[i, n + 1]) * \
                     (g.reshape((n + 1, 1)) * X[l[i]:l[i] + 1, :]).reshape(X.shape)

            t = g.dot(X)
            GradX[l[i]] -= np.sum(np_coe * np_subset_inv_A[:, 0, i] *
                                  np_Mask[:, i] *
                                  np_subset_h[:, size + 1] * np_G[i, n + 1]) * t

            Gradx_t[l[i]] += np.sum(np_coe * np_subset_inv_A[:, 0, i] *
                   np_Mask[:, i] *
                   (np_G[i, n + 1] *
                    (- (np.sum(np_subset_h[:, 1:size + 1] * g[l[1:size + 1]], axis=1) +
                    np_subset_h[:, size + 1] * np_G[i, n + 1]) + np_subset_h[:, size + 1])))

    # b = np.zeros((1 << size, size + 3))
    # for i in range(1, size + 1):
    #     g = np_G[i]
    #     tmp = 0.
    #     for j in range(1, size + 3):
    #         if j < size + 1:
    #             b[:, i] -= np_Mask[:, i] * np_subset_h[:, j] * g[l[j]] * g[n + 1]
    #             tmp += g[l[j]]
    #         elif j == size + 1:
    #             b[:, i] += np_Mask[:, i] * np_subset_h[:, j] * (1 - g[n + 1]) * g[n + 1]
    #         else:
    #             b[:, i] -= np_Mask[:, i] * np_subset_h[:, j] * (1 - tmp - g[n + 1])
    #
    # np_subset_Gradx_t = np.sum(b * np_subset_inv_A[:, 0], axis=1)
    #
    # Gradx_t = np_coe.dot(np_subset_Gradx_t)

    hitting_probability = np_coe.dot(np_subset_h[:, 0])

    return {'GradX': GradX, 'Gradx_t': Gradx_t, 'hit': hitting_probability}
