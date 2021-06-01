import os
import heapq
import copy
import json
import random
import numpy as np
from randomwalk_fast_neighborhood_v3 import *

cardinality_constraint = 3
lb_card = 2
periods = 3
predict_days = 1

number_of_try = 3
# prediction starts from 21th

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


def cal_acc_multiple(positive_order_list, true_order_list, size=None):
    d1 = {}
    tot = 0
    for order in positive_order_list:
        if size == None or size == order.cardinality:
            if tuple(sorted(order.itemset)) not in d1:
                d1[tuple(sorted(order.itemset))] = 0
            d1[tuple(sorted(order.itemset))] += 1
            tot += 1

    overlap = 0
    for order in true_order_list:
        if size == None or size == order.cardinality:
            if tuple(sorted(order.itemset)) in d1 and d1[tuple(sorted(order.itemset))] > 0:
                d1[tuple(sorted(order.itemset))] -= 1
                overlap += 1
    return overlap / (tot + 1)


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


if __name__ == '__main__':

    monthlist = ['201808', '201809', '201810', '201811', '201812', '201901',
                 '201902', '201903', '201904', '201905', '201906']

    dir_path = "../result/mixture_" + str(lb_card) + "_" + str(cardinality_constraint) + str(periods) + \
               str(predict_days) + "/" + str(d) + "/"
    # dir_path = "../result/without_1/"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    output = open(dir_path + "/result.csv", 'w')

    print('month', end='', file=output)

    print(', method', end='', file=output)

    # print(', K_type', end='', file=output)

    print(", #_true_orders_multiple", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",#_true_orders_" + str(i) + "_multiple", end='', file=output)

    print(",#_true_orders_unique", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",#_true_orders_" + str(i) + "_unique", end='', file=output)

    print(",#_positive_orders_multiple", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",#_positive_orders_" + str(i) + "_multiple", end='', file=output)

    print(",#_positive_orders_unique", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",#_positive_orders_" + str(i) + "_unique", end='', file=output)

    print(",recall_unique", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",recall_" + str(i) + "_unique", end='', file=output)

    print(",accuracy_unique", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",accuracy_" + str(i) + "_unique", end='', file=output)

    print(",recall_multiple", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",recall_" + str(i) + "_multiple", end='', file=output)

    print(",accuracy_multiple", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",accuracy_" + str(i) + "_multiple", end='', file=output)

    print(",overlap", end='', file=output)
    for i in range(1, cardinality_constraint + 1):
        print(",overlap_" + str(i), end='', file=output)

    print('', file=output)

    for T in range(11):
        f = open("../data_ningbo_with_user/" + monthlist[T] + ".txt", "r")
        # f = open("../Taobao/taobao_rw_without_1.txt")
        n, m = [int(x) for x in f.readline().strip().split()]
        print(n, m)

        true_order_list = []
        training_order_list = []

        for i in range(m):
            l = [int(x) for x in f.readline().strip().split()]

            day = l[0] % 100
            if 21 - periods <= day < 21:
                tmp = training_order_list
            elif 21 <= day < 21 + predict_days:
                tmp = true_order_list
            else:
                continue

            itemset = set(l[3:])
            cardinality = len(itemset)
            if cardinality > cardinality_constraint:
                continue
            # if cardinality < lb_card:
            #     continue
            tmp.append(Order(1, cardinality, itemset))

        print(len(true_order_list), len(training_order_list))
        m = len(true_order_list)


        # random.shuffle(training_order_list)
        # ss_order_list = training_order_list[0:m]

        def save(positive_order_list, method):

            print(monthlist[T], end='', file=output)

            print(',', method, end='', file=output)

            # print(',', per, end='', file=output)
            order_col = [true_order_list, positive_order_list]
            size_list = [i for i in range(1, cardinality_constraint + 1)]
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

            for size in size_list:
                value = cal_acc_multiple(true_order_list, positive_order_list, size)
                print(',' + str(value), end='', file=output)

            for size in size_list:
                value = cal_acc_multiple(positive_order_list, true_order_list, size)
                print(',' + str(value), end='', file=output)

            overlap = cal_overlap(positive_order_list, true_order_list)
            print(', ' + str(overlap), end='', file=output)
            for i in range(1, cardinality_constraint + 1):
                overlap = cal_overlap(positive_order_list, true_order_list, i)
                print(',' + str(overlap), end='', file=output)

            print('', file=output)


        map_order = {}
        tot = 0
        for order in training_order_list:
            s = order.itemset
            if tuple(sorted(s)) not in map_order:
                map_order[tuple(sorted(s))] = 0
            map_order[tuple(sorted(s))] += 1
            tot += 1

        ss_rank_order_list = Rank(m, tot, map_order)
        # for per in [0.9, 1, 1.1]:
        #     ss_rank_order_list = Rank(int(m * per), tot, map_order)
        save(ss_rank_order_list, 'ss+rank')

        for _ in range(number_of_try):
            random.shuffle(training_order_list)
            positive_order_list = training_order_list[0:m]
            save(positive_order_list, 'ss')

        for _ in range(number_of_try):
            item_freq = np.zeros((1 + n))
            size_freq = np.zeros((1 + cardinality_constraint))
            for order in training_order_list:
                size_freq[order.cardinality] += 1
                for x in order.itemset:
                    item_freq[x] += 1
            size_freq /= np.sum(size_freq)
            item_freq /= np.sum(item_freq)
            positive_order_list = []
            while len(positive_order_list) < len(true_order_list):
                size = np.random.choice(a=cardinality_constraint + 1, p=size_freq)
                s = set()
                for i in range(size):
                    s.add(np.random.choice(a=n + 1, p=item_freq))
                positive_order_list.append(Order(1, len(s), s))
            save(positive_order_list, "Random")
        # save(ss_order_list, 'ss')

        # f = open(dir_path + "/order_varyk_" + monthlist[T] + "_" + str(tt) + ".txt")

        f = open("../DNNTSP/results/Tmall_" +
                 str(cardinality_constraint) + str(periods) + str(predict_days) + "_" + monthlist[
                     T] + "/output_set.json")

        # f = open("../DNNTSP/results/Taobao_rw_without_1/output_set.json")
        output_set = json.loads(f.readline())
        positive_order_list = []
        for key in output_set:
            positive_order_list.append(Order(1, len(output_set[key]), set(output_set[key])))

        lb_order_list = []
        for order in training_order_list:
            if order.cardinality < lb_card:
                lb_order_list.append(order)
        random.shuffle(lb_order_list)
        positive_order_list += lb_order_list[:len(true_order_list) - len(positive_order_list)]
        print('number: ', len(true_order_list), len(positive_order_list))
        save(positive_order_list, 'DNNTSP')

        # -------------------------------------------------------------------------

        for type in ["rank", "rw"]:
            for _ in range(number_of_try):
                f = open(dir_path + "order_" + type + "_" + monthlist[T] + "_" + str(_) + ".txt")
                n, m = [int(x) for x in f.readline().strip().split()]
                positive_order_list = []
                for i in range(m):
                    l = [int(x) for x in f.readline().strip().split()]
                    itemset = set(l[2:])
                    cardinality = len(itemset)
                    positive_order_list.append(Order(1, cardinality, itemset))

                print('number: ', len(true_order_list), len(positive_order_list))
                save(positive_order_list, type)

            # --------------------------------------------------------#

        f = open("../discrete-subset-choice-master/output/Tmall_" + str(cardinality_constraint) + str(periods) +
                 str(predict_days) + "_positive_" + monthlist[T] + ".txt")
        m = int(f.readline())
        positive_order_list = []
        for i in range(m):
            l = [int(x) for x in f.readline().strip().split()]
            positive_order_list.append(Order(1, len(l), set(l)))
        print('number: ', len(true_order_list), len(positive_order_list))
        save(positive_order_list, "DCM")
