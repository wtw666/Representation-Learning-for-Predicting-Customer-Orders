import numpy as np
from scipy import optimize as op
import random
import time

import os
import threading

from multiprocessing import Process, Queue

import heapq


class Order:
    def __init__(self, frequency, cardinality, itemset):
        self.frequency = frequency
        self.cardinality = cardinality
        self.itemset = itemset


from randomwalk_fast_neighborhood_v3 import *

initial_learning_rate = 0.01
factor = 0.999
steps = 3000
batch_size = 128
d = 20  # length of feature vector

lb_mix = 2
cardinality_constraint = 3
periods = 3
predict_days = 1
num_of_try = 3
times = 30


# prediction starts from 21th

def iter(batch_size, orderlist, param, neighborhood):
    GradX = np.zeros(param['X'].shape)
    Gradx_t = np.zeros(param['x_t'].shape)
    likelihood = np.array(0.)

    # cp_param = {'X': cp.array(param['X']), 'x_t': cp.array(param['x_t']), 'n': param['n']}

    count = 0
    for i in range(batch_size):
        id = random.randint(0, len(orderlist) - 1)
        r = cal_prob_grad(orderlist[id].itemset, param, neighborhood)
        if r['hit'] > 1e-40:
            GradX += r['GradX'] / r['hit']
            Gradx_t += r['Gradx_t'] / r['hit']
            likelihood += np.log(r['hit'])
        else:
            count += 1
    print('count: ', count)

    # mean = (np.sum(GradX) + np.sum(Gradx_t)) / (GradX.shape[0] * GradX.shape[1] + Gradx_t.shape[0])
    # std = np.sqrt((np.sum((GradX - mean) * (GradX - mean)) + np.sum((Gradx_t - mean) *
    #                (Gradx_t - mean))) / (GradX.shape[0] * GradX.shape[1] + Gradx_t.shape[0]))
    std = 1
    GradX = (GradX) / std
    print(np.max(GradX), np.min(GradX), Gradx_t, likelihood)
    Gradx_t /= std
    likelihood /= batch_size - count
    return {'GradX': GradX, 'Gradx_t': Gradx_t,
            'likelihood': likelihood}


def train(orderlist, param, neighborhood):
    X = param['X']
    x_t = param['x_t']

    learning_rate = initial_learning_rate
    l = []
    for step in range(steps):
        t1 = time.time()
        ret = iter(batch_size, orderlist, param, neighborhood)
        X += learning_rate * ret['GradX']
        # X = X / (np.std(X) * 100)
        x_t += learning_rate * ret['Gradx_t']
        learning_rate *= factor
        t2 = time.time()
        print(ret['likelihood'])
        l.append(ret['likelihood'])
        print('elapsed time: ', t2 - t1)

    return l


def accelerated_sample(param, trans_matrix):
    n = param['n']
    s = set()
    cur = 0
    g = np.array(trans_matrix[0])
    # g = get_trans([0], param, True, 'np')
    # print(g.shape, type(g))
    # g = g.reshape((n + 2))
    cur = np.random.choice(a=n + 2, p=g)
    s.add(cur)

    while True:
        g = np.array(trans_matrix[cur])
        # g = get_trans([cur], param, True, 'np')
        # g = g.reshape((n + 2))
        cur = np.random.choice(a=n + 2, p=g)
        if cur == n + 1:
            break
        s.add(cur)
        # print(cur)
        if len(s) > cardinality_constraint:
            break
    return s


def Rank(n, m, map_order):
    tmp = []
    que = []
    # tot = 0
    for x in map_order:
        map_order[x] /= m
        tmp.append(Order(1, len(x), set(x)))

    for i in range(len(tmp)):
        p = map_order[tuple(sorted(tmp[i].itemset))]
        que.append([-p, i])

    heapq.heapify(que)

    ret = []

    for _ in range(n):
        pair = heapq.heappop(que)
        ret.append(tmp[pair[1]])
        pair[0] += 1 / n
        heapq.heappush(que, pair)

    return ret


def work(month):
    # f = open("../data3/" + month + ".txt", "r")
    f = open("../data_ningbo_with_user/" + month + ".txt", "r")
    n, m = [int(x) for x in f.readline().strip().split()]
    print(n, m)
    # n m
    # date userID cardinality v_1 v_2 ... v_|s|
    # ...

    true_orderlist = []
    # distr = {1: 0, 2: 0, 3: 0}
    distr = {x: 0 for x in range(1, cardinality_constraint + 1)}

    training_data = []
    mix_data = []
    for i in range(m):
        l = [int(x) for x in f.readline().strip().split()]
        day = l[0] % 100
        if 21 <= day < 21 + predict_days:
            itemset = set(l[3:])
            cardinality = len(itemset)
            if cardinality > cardinality_constraint:
                continue
            true_orderlist.append(Order(1, cardinality, itemset))
        elif 21 - periods <= day < 21:
            itemset = set(l[3:])
            cardinality = len(itemset)
            if cardinality > cardinality_constraint:
                continue
            # if cardinality < lb_card:
            #     continue
            distr[cardinality] += 1
            if cardinality >= lb_mix:
                training_data.append(Order(1, cardinality, itemset))
            else:
                mix_data.append(Order(1, cardinality, itemset))

    random.shuffle(training_data)

    neighborhood = np.ones((n + 1, n + 1))
    for i in range(n + 1):
        neighborhood[i, i] = 0
    for i in range(n + 1):
        neighborhood[i, 0] = 0

    print('number of orders: ', len(training_data))
    # with cp.cuda.Device(1):
    # X = np.random.normal(size=(n + 1, d), scale=0.1)
    X = np.random.normal(size=(n + 1, d), scale=0.01)
    # print(X.shape)
    x_t = np.zeros((n + 1,))
    x_t[:] = 7
    # x_t = np.array(-1.5)
    # print(x_t.shape)
    param = {'X': X, 'x_t': x_t, 'n': n}
    ret = train(training_data, param, neighborhood)

    # m = len(true_orderlist)

    # m = 10

    # Store training result

    dir_path = "../result/mixture_" + str(lb_mix) + "_" + str(cardinality_constraint) + str(periods) + \
               str(predict_days) + "/" + str(d)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    output = open(dir_path + "/category_train_log_" + month + "_" +
                  time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".txt", "w")
    print('dimension: ', d, file=output)
    print('sample_size: ', m, file=output)
    print('cardinarity_constraint: ', cardinality_constraint, file=output)
    print('batch_size: ', batch_size, file=output)
    print('learning_rate: ', initial_learning_rate, file=output)
    print('discount_factor: ', factor, file=output)
    print('iterations: ', steps, file=output)
    for x in ret:
        print(x, file=output)

    output = open(dir_path + "/category_train_parameter_" + month + "_" +
                  time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".txt", "w")

    print('dimension: ', d, file=output)
    print('sample_size: ', m, file=output)
    print('cardinarity_constraint: ', cardinality_constraint, file=output)
    print('batch_size: ', batch_size, file=output)
    print('learning_rate: ', initial_learning_rate, file=output)
    print('discount_factor: ', factor, file=output)
    print('iterations: ', steps, file=output)

    print(n, file=output)
    print(d, file=output)
    for i in range(n + 1):
        for j in range(d):
            print(param['X'][i, j], file=output)
    # print(param['x_t'], file=output)
    for i in range(n + 1):
        print(param['x_t'][i], file=output)

    # Store generated orders
    trans_matrix = get_trans([x for x in range(n + 2)], param, neighborhood, 'np')
    for tt in range(num_of_try):
        m = int(len(true_orderlist) * (len(training_data) / (len(training_data) + len(mix_data))))
        m2 = m * times
        print(m)
        print(tt)

        map_order = {}
        tot = 0
        while tot < m2:
            s = accelerated_sample(param, trans_matrix)
            if len(s) > cardinality_constraint:
                continue
            if len(s) < lb_mix:
                continue
            if tuple(sorted(s)) not in map_order:
                map_order[tuple(sorted(s))] = 0
            map_order[tuple(sorted(s))] += 1
            tot += 1
        print('generated')

        # positive_orderlist = Rank(m, m2, map_order)
        random.shuffle(mix_data)
        m = len(true_orderlist)
        m2 = m * times
        while tot < m2:
            s = mix_data[tot % len(mix_data)].itemset
            if tuple(sorted(s)) not in map_order:
                map_order[tuple(sorted(s))] = 0
            map_order[tuple(sorted(s))] += 1
            tot += 1
            # positive_orderlist += mix_data[:(len(true_orderlist) - m)]
        positive_orderlist = Rank(m, m2, map_order)

        output = open(dir_path + "/order_rank_" + month + "_" + str(tt) + ".txt", "w")
        print(n, len(positive_orderlist), file=output)
        for x in positive_orderlist:
            print(1, x.cardinality, end='', file=output)
            for y in x.itemset:
                print(' ' + str(y), end='', file=output)
            print('', file=output)

        map_order = {}
        tot = 0
        positive_orderlist = []
        m = int(len(true_orderlist) * (len(training_data) / (len(training_data) + len(mix_data))))
        while tot < m:
            s = accelerated_sample(param, trans_matrix)
            if len(s) > cardinality_constraint:
                continue
            if len(s) < lb_mix:
                continue
            positive_orderlist.append(Order(1, len(s), sorted(s)))
            tot += 1
        print('generated')
        positive_orderlist += mix_data[:(len(true_orderlist) - m)]

        output = open(dir_path + "/order_rw_" + month + "_" + str(tt) + ".txt", "w")
        print(n, len(positive_orderlist), file=output)
        for x in positive_orderlist:
            print(1, x.cardinality, end='', file=output)
            for y in x.itemset:
                print(' ' + str(y), end='', file=output)
            print('', file=output)


if __name__ == '__main__':

    monthlist = ['201808', '201809', '201810', '201811', '201812', '201901',
                 '201902', '201903', '201904', '201905', '201906']

    queue = Queue()
    process_list = []
    for T in range(11):
        process_list.append(Process(target=work, args=(
            monthlist[T],)))
    for T in range(11):
        process_list[T].start()
    for T in range(11):
        process_list[T].join()
