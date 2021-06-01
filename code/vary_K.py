'''
 Test recency
'''
import numpy as np
from scipy import optimize as op
import random
import time

import os
# os.environ["OMP_NUM_THREADS"] = "1"
import threading
import copy

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
dimension = 20  # length of feature vector

cardinality_constraint = 4
periods = 7
predict_days = 3
num_of_try = 5
times = 100


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


def Rank(n, m, map_order2):
    map_order = copy.deepcopy(map_order2)
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
    f = open("../data3/" + month + ".txt", "r")
    n, m = [int(x) for x in f.readline().strip().split()]
    print(n, m)

    true_orderlist = []
    # distr = {1: 0, 2: 0, 3: 0}
    distr = {x: 0 for x in range(1, cardinality_constraint + 1)}

    training_data = []
    for i in range(m):
        l = [int(x) for x in f.readline().strip().split()]
        day = l[0] % 100
        if 8 <= day < 8 + predict_days:
            l = l[1:]
            frequency = l[0]
            itemset = set(l[2:])
            cardinality = len(itemset)
            if cardinality > cardinality_constraint:
                continue
            distr[cardinality] += frequency
            for j in range(frequency):
                true_orderlist.append(Order(1, cardinality, itemset))
        elif 7 - periods < day <= 7:
            l = l[1:]
            frequency = l[0]
            itemset = set(l[2:])
            cardinarity = len(itemset)
            if cardinarity > cardinality_constraint:
                continue
            for j in range(frequency):
                training_data.append(Order(1, cardinarity, itemset))

    random.shuffle(training_data)

    dir_path = "../result/adaptive_v3_" + str(cardinality_constraint) + str(periods) + \
               str(predict_days) + "/" + str(dimension)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    string = dir_path
    for root, dirs, files in os.walk(string):
        pass

    string = 'category_train_parameter_' + monthlist[T] + '_'
    filename = ''
    for x in files:
        if string in x:
            filename = x
    print(filename)
    f = open(os.path.join(root, filename), 'r')

    for i in range(7):
        x = f.readline()
        # print(x)

    n = int(f.readline())
    d = int(f.readline())

    X = np.zeros((n + 1, d))
    x_t = np.zeros((n + 1,))
    param = {'X': X, 'x_t': x_t, 'n': n}
    for i in range(n + 1):
        for j in range(d):
            x = float(f.readline())
            X[i, j] = x
    for i in range(n + 1):
        x_t[i] = float(f.readline())

    neighborhood = np.ones((n + 1, n + 1))
    for i in range(n + 1):
        neighborhood[i, i] = 0
    for i in range(n + 1):
        neighborhood[i, 0] = 0

    m = len(true_orderlist)

    trans_matrix = get_trans([x for x in range(n + 2)], param, neighborhood, 'np')
    for tt in range(num_of_try):
        m2 = m * times
        print(m)
        print(tt)

        map_order = {}
        tot = 0
        while tot < m2:
            s = accelerated_sample(param, trans_matrix)
            if len(s) > cardinality_constraint:
                continue
            if tuple(sorted(s)) not in map_order:
                map_order[tuple(sorted(s))] = 0
            map_order[tuple(sorted(s))] += 1
            tot += 1
        print('generated')

        positive_orderlist = Rank(m, m2, map_order)

        output = open(dir_path + "/order_varyk_" + month + "_" + str(tt * 3) + ".txt", "w")
        print(n, len(positive_orderlist), file=output)
        for x in positive_orderlist:
            print(1, x.cardinality, end='', file=output)
            for y in x.itemset:
                print(' ' + str(y), end='', file=output)
            print('', file=output)

        positive_orderlist = Rank(int(0.9*m), m2, map_order)

        output = open(dir_path + "/order_varyk_" + month + "_" + str(tt * 3 + 1) + ".txt", "w")
        print(n, len(positive_orderlist), file=output)
        for x in positive_orderlist:
            print(1, x.cardinality, end='', file=output)
            for y in x.itemset:
                print(' ' + str(y), end='', file=output)
            print('', file=output)

        positive_orderlist = Rank(int(1.1 * m), m2, map_order)

        output = open(dir_path + "/order_varyk_" + month + "_" + str(tt * 3 + 2) + ".txt", "w")
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