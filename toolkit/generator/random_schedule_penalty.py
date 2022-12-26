#!/usr/bin/env python3

import random
import argparse
import sys
from expr import *


def random_schedule_penalty(coeff_expr, base_expr, scale_expr, of=sys.stdout):
    print(f'{random_expr_float("coeff_expr",coeff_expr)} {random_expr_float("base_expr",base_expr)} {random_expr_float("scale_expr",scale_expr)}', file=of)


def main():
    parser = argparse.ArgumentParser(
        description='random schedule penalty setting')
    parser.add_argument('--coeff-expr', default="x",
                        type=str, help='penalty係数分布')
    parser.add_argument('--base-expr',  default="0.01+0.99*x",
                        type=str, help='decay base expr')
    parser.add_argument('--scale-expr',  default="1.0+49.0*x",
                        type=float, help='score scale min')
    parser.add_argument('-s', '--seed', default=0, type=int, help='シード値')
    args = parser.parse_args()
    random.seed(args.seed)
    random_schedule_penalty(args.coeff_expr,
                            args.base_expr, args.scale_expr)


if __name__ == '__main__':
    main()
