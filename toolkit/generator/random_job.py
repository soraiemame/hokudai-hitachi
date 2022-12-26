#!/usr/bin/env python3

import global_value as gv
import random

import random
import argparse
from typing import List
import networkx as nx
from expr import *
import math

from random_reward_function import *


class Job:
    def __init__(self) -> None:
        self.id = -1
        self.type = -1
        self.n_task = -1
        self.pos = -1
        self.penalty = -1
        self.mandatory = False
        self.others_begin = -1
        self.dependency = []


def main(rawargs, file=sys.stdout):
    parser = argparse.ArgumentParser(
        description='獲得量関数のランダム生成')
    parser.add_argument('-n', '--job-num', required=True,
                        type=int, help='ジョブ個数')
    parser.add_argument('-t', '--type-num-expr', default="0",
                        type=str, help='タイプの分布(0-based)')
    parser.add_argument('--n-task-expr', default="1+1001*x",
                        type=str, help='タスク数の分布')
    parser.add_argument('-v', '--vertex-num', required=True,
                        type=int, help='頂点数')
    parser.add_argument('--penalty-expr', default="0.9+0.09*x",
                        type=str, help='ペナルティ係数の分布')
    parser.add_argument('-m', '--mandatory-num',
                        default=1, type=int, help='必須ジョブ数')
    parser.add_argument('-s', '--seed', default=0, type=int, help='シード値')
    parser.add_argument('--reward-interval-expr', default="10",
                        type=str, help='制御点間隔分布')
    parser.add_argument('--reward-value', default="1000000",
                        type=str, help='制御点定数')
    parser.add_argument('--reward-duration-expr', default="50",
                        type=str, help='報酬獲得可能時刻長さ分布')
    parser.add_argument('--reward-stddev-expr', default='0.2',
                        type=str, help='報酬標準偏差分布')
    parser.add_argument('--reward-upper-limit-expr', type=str,
                        default="-1", help='報酬関数上限値')
    parser.add_argument('--reward-lower-limit-expr', type=str,
                        default="-1", help='報酬関数下限値')
    parser.add_argument('--dep-nodes-expr', default="1+5*x",
                        type=str, help='ジョブ依存関係の連結成分のサイズ分布')
    parser.add_argument('--max-dep-expr', default="3",
                        type=str, help='依存するジョブの個数の最大数')
    parser.add_argument('--t-max', required=True,
                        help='最大時刻(1-based)', type=int)
    parser.add_argument('--weather-dependency-expr',
                        required=True, help='天候依存度分布', type=str)
    parser.add_argument('--processable-types', nargs='*', type=int,
                        help='処理可能なタイプの列(これに含まれないジョブタイプを持つジョブは生成されない)', default=[])

    args = parser.parse_args(rawargs)
    print(args.processable_types, file=sys.stderr)
    random.seed(args.seed)
    uniq_id = 0
    deps: List[nx.DiGraph] = []
    while True:
        n = random_expr_int('dep_nodes_expr', args.dep_nodes_expr)
        while True:
            G = nx.gnp_random_graph(
                n, 0.5, seed=random.getrandbits(64), directed=True)
            DAG = None
            if n == 1:
                DAG = G
            else:
                DAG = nx.DiGraph(
                    [(u, v, {'weight': random.randint(-n, n)}) for (u, v) in G.edges() if u < v])
            max_dep = random_expr_int('max_dep', args.max_dep_expr)

            if nx.is_directed_acyclic_graph(DAG) and all(DAG.out_degree(i) <= max_dep for i in DAG.nodes()):
                for id in list(DAG.nodes):
                    DAG.nodes[id]['uniq_id'] = uniq_id
                    uniq_id += 1
                deps.append(DAG)
                break
        if uniq_id >= args.job_num:
            break

    jobs: List[Job] = []

    for DAG in deps:
        for id in list(DAG.nodes):
            j = Job()
            j.id = DAG.nodes[id]['uniq_id']
            for pre in DAG.predecessors(id):
                j.dependency.append(DAG.nodes[pre]['uniq_id'])
            while True:
                j.type = random_expr_int('type_num_expr', args.type_num_expr)
                if len(args.processable_types) == 0 or j.type in args.processable_types:
                    break
            j.n_task = random_expr_int('n_task_expr', args.n_task_expr)
            j.pos = random.randint(0, args.vertex_num-1)
            j.penalty = random_expr_float('penalty_expr', args.penalty_expr)
            jobs.append(j)
    mands: List[bool] = [False]*len(jobs)

    for i in range(args.mandatory_num):
        mands[i] = True
    random.shuffle(mands)

    for i in range(len(jobs)):
        jobs[i].mandatory = mands[i]

    print(len(jobs))
    for j in jobs:
        dur = random_expr_int('reward_duration_expr',
                              args.reward_duration_expr)
        begin = random.randint(
            0, args.t_max-dur)
        end = begin+dur
        point_n = round(
            dur/random_expr_float('reward_interval_expr', args.reward_interval_expr))
        rewfn = generate_random_reward_function(
            begin, end, random_expr_float('reward_value_expr', args.reward_value), point_n, random_expr_float('reward_stddev_expr', args.reward_stddev_expr), upper_limit=random_expr_int('reward_upper_limit', args.reward_upper_limit_expr), lower_limit=random_expr_int('reward_lower_limit', args.reward_lower_limit_expr))

        print(f'{j.id+1} {j.type+1} {j.n_task} {j.pos+1} {j.penalty} {random_expr_float("weather_dependency_expr",args.weather_dependency_expr)} {1 if j.mandatory else 0}')
        rewfn.print(True)
        print(f'{len(j.dependency)}', end='')
        for i in j.dependency:
            print(f' {i+1}', end='')
        print('')


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
