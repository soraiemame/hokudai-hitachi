#!/usr/bin/env python3

import random
import argparse
from expr import *


class WorkerInfo:
    def __init__(self) -> None:
        self.processable_types = []
        self.pos = -1
        self.proc_task = -1


def random_worker(num, job_type_num_expr, job_type_expr, vertex_num, task_expr) -> List[WorkerInfo]:
    print(num)
    ret: List[WorkerInfo] = []
    for i in range(num):
        job_types = []
        processable_type_num = random_expr_int(
            'job_type_num_expr', job_type_num_expr)
        for jt in range(processable_type_num):
            while True:
                t = random_expr_int('job_type_expr', job_type_expr)
                if not t in job_types:
                    job_types.append(t)
                    break
        job_types.sort()
        w = WorkerInfo()
        w.pos = random.randint(1, vertex_num)
        w.proc_task = random_expr_int("task_expr", task_expr)
        w.processable_types = job_types.copy()
        print(
            f'{w.pos} {w.proc_task} {len(w.processable_types)} {" ".join([str(a+1) for a in w.processable_types])}')
        ret.append(w)
    return ret


def main():
    parser = argparse.ArgumentParser(
        description='ランダムWorker')
    parser.add_argument('-d', '--num', default=1,
                        type=int, help='生成する個数')
    parser.add_argument('--job-type-num-expr', default="1",
                        type=str, help='job typeの個数の分布')
    parser.add_argument('--job-type-expr', default="1",
                        type=str, help='job typeの出現確率の分布')
    parser.add_argument('-v', '--vertex-num', required=True,
                        type=int, help='頂点数')
    parser.add_argument('-s', '--seed', default=0, type=int, help='シード値')
    parser.add_argument('--task-expr', default="1+100*x",
                        type=str, help='単位時間あたりに処理可能なタスク数の分布')
    args = parser.parse_args()
    random.seed(args.seed)
    random_worker(args.num, args.job_type_num_expr, args.job_type_expr, args.vertex_num,
                  args.task_expr)


if __name__ == '__main__':
    main()
