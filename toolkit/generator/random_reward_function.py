#!/usr/bin/env python3
import argparse
import bisect
import random
import sys
from math import ceil, floor
from sys import stderr

import numpy as np
from expr import *


class RewardFunction:
    def __init__(self, pts=[], already_sorted=False) -> None:
        self.points = []
        self.set_points(pts, already_sorted=already_sorted)

    def set_points(self, pts, already_sorted=False):
        if already_sorted:
            self.points = pts[:]
        else:
            self.points = sorted(pts)

    def print(self, start_from_1=False, file=sys.stdout, for_dbg=False):
        if not for_dbg:
            print(f'{len(self.points)}', end='', file=file)
        for t, v in self.points:
            print(f' {t+(1 if start_from_1 else 0)} {v}',
                  end='' if not for_dbg else '\n', file=file)
        print(file=file)

    # NOTE:Not 128bit float
    def at(self, t: int) -> float:
        pos = bisect.bisect_right(self.points, (t,))-1
        if pos == -1:
            return self.points[0][1]
        elif pos == len(self.points)-1:
            return self.points[len(self.points)-1][1]

        return (self.points[pos+1][1]-self.points[pos][1])/(self.points[pos+1][0]-self.points[pos][0])*(self.points[pos+1][0]-t)+self.points[pos][1]

    def last_time(self):
        if len(self.points) == 0:
            raise ValueError('self.points is empty')

        return self.points[len(self.points)-1][0]

    def start_time(self):
        if len(self.points) == 0:
            raise ValueError('self.points is empty')

        return self.points[0][0]

    def time_nth(self, i):
        if len(self.points) == 0:
            raise ValueError('self.points is empty')

        return self.points[i][0]

    def value_nth(self, i):
        if len(self.points) == 0:
            raise ValueError('self.points is empty')

        return self.points[i][1]


def generate_random_reward_function(begin, end, c, n, dev, upper_limit=-1, lower_limit=-1) -> RewardFunction:
    n = int(n)
    n += 1
    end -= 1
    while True:
        np.random.seed(random.getrandbits(32))
        values = np.cumprod(np.random.lognormal(0.0, dev, n))

        unitsum = sum(v*v for v in values)
        base = c*math.sqrt(n/unitsum)

        r = [round(base*v) for v in values]
        if upper_limit >= 0 and any(x > upper_limit for x in r):
            continue
        if lower_limit >= 0 and any(x < lower_limit for x in r):
            continue
        ret = []
        ret.append(((begin-1), 0))
        for i in range(n):
            t = round(begin+(end-begin)*i/(n-1))
            ret.append((t, r[i]))
        ret.append(((end+1), 0))
        return RewardFunction(pts=ret, already_sorted=True)


def main():

    parser = argparse.ArgumentParser(
        description='報酬関数生成')
    parser.add_argument('-b', '--begin', required=True,
                        type=int, help='開始時刻(区間=[開始時刻,終了時刻))')
    parser.add_argument('-e', '--end', required=True,
                        type=int, help='終了時刻(区間=[開始時刻,終了時刻))')
    parser.add_argument('-v', '--value-expr',  default="1000000",
                        type=str, help='sqrt(制御点の値の2乗和の平均)の分布')
    parser.add_argument('-i', '--interval-n', required=True,
                        type=int, help='区間個数')
    parser.add_argument('-n', '--num', required=True,
                        type=int, help='生成個数')
    parser.add_argument('-d', '--stddev-expr', required=True,
                        type=str, help='制御点の値の標準偏差の分布')
    parser.add_argument('--one-based',
                        action='store_true', help='use 1-based time')
    parser.add_argument('-s', '--seed', default=0, type=int, help='シード値')
    parser.add_argument('--upper-limit', help='値上限', type=int, default=-1)
    parser.add_argument('--lower-limit', help='値下限', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    for i in range(args.num):
        generate_random_reward_function(args.begin, args.end, random_expr_float('reward_value_expr',
                                                                                args.value_expr), args.interval_n, random_expr_float('reward_stddev_expr', args.stddev_expr), upper_limit=args.upper_limit, lower_limit=args.lower_limit).print(args.one_based, file=sys.stdout, for_dbg=args.debug)


if __name__ == '__main__':
    main()
