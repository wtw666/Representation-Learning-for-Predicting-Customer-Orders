import numpy as np
from scipy import optimize as op
import random
import time
import os
import heapq
import frequent_itemset_mining as frm

# from randomwalk_cpu import *

cardinarity_constraint = 4
periods = 7
predict_days = 3
number_of_try = 5
d = 20


class Order:
    def __init__(self, frequency, cardinality, itemset):
        self.frequency = frequency
        self.cardinality = cardinality
        self.itemset = itemset


def cal_acc_unique(positive_order_list, true_order_list, size=None):
    s1 = set()
    for order in true_order_list:
        if size == None or size == order.cardinality:
            s1.add(tuple(sorted(order.itemset)))

    s2 = set()
    for order in positive_order_list:
        if size == None or size == order.cardinality:
            s2.add(tuple(sorted(order.itemset)))

    if len(s2) == 0:
        return None

    return len(s1 & s2) / len(s2)


def cal_overlap(positive_order_list, true_order_list, size=None):
    d1 = {}
    tot = 0
    for order in positive_order_list:
        if size == None or size == order.cardinality:
            if tuple(sorted(order.itemset)) not in d1:
                d1[tuple(sorted(order.itemset))] = 0
            d1[tuple(sorted(order.itemset))] += 1
            tot += 1
    for x in d1:
        d1[x] /= tot

    d2 = {}
    tot = 0
    for order in true_order_list:
        if size == None or size == order.cardinality:
            if tuple(sorted(order.itemset)) not in d2:
                d2[tuple(sorted(order.itemset))] = 0
            d2[tuple(sorted(order.itemset))] += 1
            tot += 1
    for x in d2:
        d2[x] /= tot

    ret = 0
    for x in d2:
        if x in d1:
            ret += min(d1[x], d2[x])
    return ret


def count_size_multiple(order_list, size=None):
    ret = 0
    for order in order_list:
        if size == None or size == order.cardinality:
            ret += order.frequency
    return ret


def count_size_unique(order_list, size=None):
    s = set()
    for order in order_list:
        if size == None or size == order.cardinality:
            s.add(tuple(sorted(order.itemset)))
    return len(s)


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


if __name__ == '__main__':

    monthlist = ['201808', '201809', '201810', '201811', '201812', '201901',
                 '201902', '201903', '201904', '201905', '201906']

    # dir_path = "../result/fast_neighborhood" + str(cardinarity_constraint) + str(periods) + \
    #            str(predict_days) + "/" + str(d)

    dir_path = "../result/random_" + str(cardinarity_constraint) + str(periods) + \
               str(predict_days)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    output = open(dir_path + "/" + str(cardinarity_constraint) + str(periods)
                  + str(predict_days) + '.csv', 'w')

    print('date', end='', file=output)

    print(',method', end='', file=output)

    print(",#_true_orders_multiple", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",#_true_orders_" + str(i) + "_multiple", end='', file=output)

    print(",#_true_orders_unique", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",#_true_orders_" + str(i) + "_unique", end='', file=output)

    print(",#_positive_orders_multiple", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",#_positive_orders_" + str(i) + "_multiple", end='', file=output)

    print(",#_positive_orders_unique", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",#_positive_orders_" + str(i) + "_unique", end='', file=output)

    print(",recall_unique", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",recall_" + str(i) + "_unique", end='', file=output)

    print(",accuracy_unique", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",accuracy_" + str(i) + "_unique", end='', file=output)

    print(",overlap", end='', file=output)
    for i in range(1, cardinarity_constraint + 1):
        print(",overlap_" + str(i), end='', file=output)

    print('', file=output)

    for T in range(11):
        f = open("../data3/" + monthlist[T] + ".txt", "r")
        n, m = [int(x) for x in f.readline().strip().split()]
        print(n, m)

        true_order_list = []
        training_order_list = []

        for i in range(m):
            l = [int(x) for x in f.readline().strip().split()]

            day = l[0] % 100
            if 7 >= day > 7 - periods:
                tmp = training_order_list
            elif 7 < day <= 7 + predict_days:
                tmp = true_order_list
            else:
                continue

            l = l[1:]
            frequency = l[0]
            itemset = set(l[2:])
            cardinality = len(itemset)
            if cardinality > cardinarity_constraint:
                continue
            for j in range(frequency):
                tmp.append(Order(1, cardinality, itemset))

        print(len(true_order_list), len(training_order_list))
        m = len(true_order_list)

        # random.shuffle(training_order_list)
        # training_order_list = training_order_list[0:m]

        map_order = {}
        tot = 0
        for order in training_order_list:
            s = order.itemset
            if tuple(sorted(s)) not in map_order:
                map_order[tuple(sorted(s))] = 0
            map_order[tuple(sorted(s))] += 1
            tot += 1
        positive_order_list = Rank(m, tot, map_order)


        def save(positive_order_list, method):

            print(monthlist[T], end='', file=output)

            print(',' + method, end='', file=output)
            order_col = [true_order_list, positive_order_list]
            size_list = [i for i in range(1, cardinarity_constraint + 1)]
            size_list = [None] + size_list

            for i in range(2):
                for size in size_list:
                    value = count_size_multiple(order_col[i], size)
                    print(',' + str(value), end='', file=output)

                for size in size_list:
                    value = count_size_unique(order_col[i], size)
                    print(',' + str(value), end='', file=output)

            for size in size_list:
                value = cal_acc_unique(true_order_list, positive_order_list, size)
                print(',' + str(value), end='', file=output)

            for size in size_list:
                value = cal_acc_unique(positive_order_list, true_order_list, size)
                print(',' + str(value), end='', file=output)

            overlap = cal_overlap(positive_order_list, true_order_list)
            print(', ' + str(overlap), end='', file=output)
            for i in range(1, cardinarity_constraint + 1):
                overlap = cal_overlap(positive_order_list, true_order_list, i)
                print(',' + str(overlap), end='', file=output)

            print('', file=output)


        save(positive_order_list, 'ss+rank')

        for tt in range(number_of_try):
            freq = {x:0 for x in range(1, n + 1)}
            for order in training_order_list:
                for x in order.itemset:
                    freq[x] += 1
            for x in freq:
                freq[x] /= len(training_order_list)

            random_order_list = []
            while(len(random_order_list) < len(training_order_list)):
                s = set()
                for x in freq:
                    random_value = random.random()
                    if random_value < freq[x]:
                        s.add(x)
                if len(s) > cardinarity_constraint or len(s) == 0:
                    continue
                print(len(random_order_list))
                random_order_list.append(Order(1, len(s), s))
            save(random_order_list, 'random')

