#!/usr/bin/env python3
import global_value as gv
import argparse
from asyncio.subprocess import DEVNULL
import re
import toml
import tempfile
import os
import subprocess
import random
import shutil
import numpy as np
import csv
import itertools
import sys
from random_schedule_penalty import random_schedule_penalty
from random_worker import random_worker

from expr import *
import math
import random_job


def to_1b(i0b: int):
    return i0b+1


class OptionItem:
    def __init__(self, name: str, detail: str, type: str, limitation, expr_available, default_value: str, is_array: bool = False) -> None:
        self.name = name
        self.detail = detail
        self.type = type
        self.limitation = limitation
        self.expr_available = expr_available
        self.default_value = default_value
        self.is_array = is_array

    def to_string(self) -> str:
        cm = "# "
        ret = ""
        for s in self.detail.splitlines():
            ret += f'{cm} {s}\n'
        ret += '# \n'
        ret += f'{cm} 型:{self.type}\n'
        ret += f'{cm} 値の範囲:{self.limitation}\n'
        ret += f'{cm} 分布対応:{"YES" if self.expr_available else "NO"}\n'
        ret += f'{self.name} = {self.default_value}\n'
        return ret


OPTIONS = [
    OptionItem('type', '問題のタイプ(A問題,B問題)', '文字列', '"A" or "B"', False, '"B"'),
    OptionItem('seed', 'シード値', '整数',
               '0 以上 2^64-1 以下 ', False, '1234'),
    OptionItem('T_max', '最終時刻(1-based)',
               '整数', '1以上', True,
               '[300, 700, 1000]'),
    OptionItem('map_size', 'マップサイズ', '整数',
               '2のべき乗(1,2,4,...,2^n,...)', True, '2048'),
    OptionItem(
        'map_node_ratio', 'マップ分割率(ノード分割数の理論最大値に対する比率)', '実数', '0以上1以下', True, '"0.45*(0.5**where(map_max_depth>5,map_max_depth-5,0))"'),
    OptionItem('map_max_depth', 'マップ最大再帰深度',
               '整数', '1以上', True, '[5, 6, 7]'),
    OptionItem(
        'unit_dist', '単位距離\n(生成されたマップをノード間距離の最小値がこの値になるようにスケーリングする)', '整数', '1以上', True, '1'),
    OptionItem('map_vertex_num_hard_limit',
               '頂点数のハードリミット[下限,上限](端点含む)', '[整数,整数]', '[1,inf)に含まれる', True, '[[150,2000]]'),
    OptionItem('map_edge_num_hard_limit_coeff',
               '辺の数のハードリミット(頂点数に対する係数で表現)[下限,上限](端点含む)', '[非負実数,非負実数]', '[0,inf)に含まれる', True, '[[1.33333333333333, 2.0]]'),
    OptionItem('worker_num', 'ワーカー数', '整数', '1以上', True,
               '[1, 2, 5, 10]'
               ),
    OptionItem('worker_processable_num', 'ワーカーが処理可能なタイプの個数',
               '整数', '1以上', True, '"1+3*x"'),
    OptionItem('worker_processable_type', 'ワーカーが処理可能なタイプ(0-based)',
               '整数', '0以上', True, '"3*x"'),
    OptionItem('worker_max_task_processing_num',
               'ワーカーの単位時間の最大タスク実行数', '整数', '1以上', True, '"30+71*x"'),
    OptionItem('job_num', '全ジョブの数', '整数', '1以上',
               True,
               '[250, 500, 1000]'),
    OptionItem('job_type', 'ジョブのタイプ分布(0-based)', '整数', '0以上', True, '"3*x"'),
    OptionItem('task_num', '必要タスク数の範囲', '整数', '1以上', True, '"500+1000*x"'),
    OptionItem('job_unfinished_penalty', 'ジョブ未完了ペナルティの範囲',
               '実数', '0以上1以下', True,
               '["0.98+0.02*x", "0.91+0.02*x"]'
               ),
    OptionItem('mandatory_job_num', '必須ジョブ数の範囲', '整数',
               '0以上', True, '"worker_num*T_max/300*x"'),
    OptionItem('reward_interval', '報酬関数の制御点間隔', '整数', '1以上', True, '25'),
    OptionItem('reward_duration', '報酬関数が正である区間長さ',
               '整数', '1以上', True, '"(T_max-100+1)*x+100"'),
    OptionItem('reward_stddev', '報酬関数の標準偏差',
               '実数', '0以上', True, '"0.3+0.08*x"'),
    OptionItem('reward_value', '報酬関数の値の大きさの目安\n(正確には、sqrt(制御点の値の2乗平均) の値)',
               '実数', '0以上', True, '"1000000+1000001*x"'),
    OptionItem('reward_upper_hard_limit',
               '報酬関数の値の上限', '整数', '1以上', True, '10000000'),
    OptionItem('reward_lower_hard_limit',
               '報酬関数の値の下限', '整数', '1以上', True, '1'),
    OptionItem('fundamental_weather_interval_length',
               '基本天候区間の長さ(T_maxを割り切らない場合取り直す。1を含まない場合T_maxの素因数に注意)', '整数', '1以上', True, '"5+16*x"'),
    OptionItem('job_dependency_num', '依存関係の連結成分の大きさ',
               '整数', '1以上', True, '"5*x"'),
    OptionItem('job_max_dep', '1つのジョブが依存するジョブの最大個数', '整数', '0以上', True, '3'),
    OptionItem('map_area_ratio', 'マップの切断面積比',
               '実数', '0以上1以下', True, '"0.3+0.4*x"'),
    OptionItem('map_peak_num', 'マップの地形生成における山の頂点の数', '整数', '1以上', True, '20'),
    OptionItem('schedule_penalty_max', 'スケジュールペナルティ最大値',
               '実数', '0以上1以下', True, '"0.005+0.02*x"'),
    OptionItem('schedule_penalty_decay', 'スケジュールペナルティ減衰率',
               '実数', '0以上1以下', True,
               '"1-0.001*(30.0)**x"'
               ),
    OptionItem('schedule_score_scale', 'スケジュールスコア係数',
               '実数', '0以上', True, '"5**(-1+2*x)"'),
    OptionItem('weather_dependency', '天候依存度',
               '実数', '0以上1以下', True, '"0.15*x"'),
    OptionItem('weather_stationary_dist',
               '''天候の定常分布(正規化前) 各要素は分布で記述可能。
定常分布が一つしかなくても、その定常分布を配列に格納すること。
([定常分布1,定常分布2,...,定常分布n]の形を強制する。定常分布自体が配列なので、他の設定項目と異なり定常分布1しかなくても[定常分布1]と記述する。)''', '実数の配列', '各要素は0以上', True, '''[[
  0.21,
  0.25,
  0.10,
  0.31,
  0.102,
  0.023,
  0.005,
]]''', True),
    OptionItem('weather_prob_nondiag_cutoff_range', '''天候確率行列について、対角成分からこの値より離れた位置の要素を0.0にする
(=この設定値をnとして、(1+2n)重対角行列に確率行列を制限する)''', '整数', '0以上', True, '2'),
    OptionItem('weather_prob_sharpness', '''天候確率行列の成分(i,j)について
この設定値をqとして、初期値にexp(-q*|i-j|^2)で重み付けする
(最終的に生成される確率行列は初期値にかなり近い形になるので、このqを大きくすると自己遷移が増える)''', '実数', '0以上', True, "[2.0,1.5,1.0]"),
    OptionItem('weather_prob_eps', '''天候確率行列の収束判定基準等に使われる微小値
目安:1e-12 ～ 1e-6
(マシンイプシロン(2.220446049250313e-16)より小さくしないこと。)

小さいと計算に時間がかかる場合があるが、収束すれば精度の良い行列が生成される。
大きくすると計算は早いが、定常分布が指定したものと大きくずれた行列が生成される可能性がある。''', '実数', '2.220446049250313e-16 以上 1.0 未満', True, '1e-8'),
    OptionItem('weather_prob_centralize', '天候確率行列を、対角成分が最大になるように制限する',
               '真偽値', 'true or false', False, 'true'),
    OptionItem('weather_limit_const', '天候制限定数(d^weatherの冪)',
               '整数の列', '各要素は0以上。要素数は定常分布の要素数と等しい必要がある。', True, '''[[
  0,
  1,
  2,
  3,
  10,
  14,
  20,
]]''', True)
]


class OptionDetails:
    def __init__(self) -> None:
        self.HEADER = '''分布対応:YES と書いてあるものは
[0,1)の一様分布に従う変数xを入力とした式を指定できる。
また、項目ごとに固定で与えられる乱数 g (これも[0,1)の一様分布) も利用可能。 
設定値を配列に入れて複数記述すると、それらの中からランダム(or 全組み合わせの中で昇順)で選ばれる。この選択も項目ごとに固定である。
設定値が1つの場合配列に格納せずに記述できる。ただし、設定値自体が配列の場合は必ず配列に格納する必要がある。(二重配列になる)
大半の設定項目で定数としてT_max,worker_num,map_max_depthが利用できる。
(処理順:T_max, worker_num, map_max_depth, その他)
例:
"x": [0,1)の一様分布
"1+10*x": [1,11)の一様分布。
"2.0**(10*x)":ある種の指数的な分布
"1.0": 定数1.0
1.0: 定数1.0 (単なる数値の場合文字列として指定しなくても良い)
"1.0-2.0*g": [-1.0,1.0)の一様分布だが、その値はテストケース生成過程を通して変化しない。
"1.0-2.0*g-1.0*x": [-1.0,1.0)の一様分布(固定) - [0.0,1.0)の一様分布(評価の度に変化)
[1.0,2.0]: 定数1.0 または 定数2.0 が選ばれる。選ばれる要素はテストケース生成過程を通して変化しない。
["1.0+2.0*g+1.0*x",10.0,"20.0+8.0*g"]: [1,3)の一様分布(固定)+[0,1)の一様分布(評価の度に変化) または 定数10.0 または [20,28)の一様分布(固定)。選ばれる要素はテストケース生成過程を通して変化しない。
"T_max*(9.0+x)":[T_max*9.0,T_max*10.0)の一様分布
"T_max*worker_num/2*x":[0,T_max*worker_num/2)の一様分布
["T_max*(x**0.5)","T_max*(x**1.5)"]: T_maxに依存した分布2つ
'''
        self.options = OPTIONS
        self.is_array_ = {v.name: v.is_array for v in OPTIONS}

    def print(self, file=sys.stdout):
        for s in self.HEADER.splitlines():
            print(f'# {s}', file=file)
        print(file=file)
        for o in self.options:
            print(o.to_string(), file=file)

    def is_array(self, option_name: str):
        return self.is_array_[option_name]


OPTION_DETAILS = OptionDetails()


class RandomConfig:
    def __init__(self, config) -> None:
        self.config = config
        self.memo = {}
        self.rmemo = {}
        self.picked_all = False

    def as_int(self, key):
        if not key in self.memo:
            self.memo[key] = random_expr_int(f'CONFIG_{key}', self.pick(key))
        return self.memo[key]

    def as_ints(self, key):
        return str(self.as_int(key))

    def as_float(self, key):
        if not key in self.memo:
            self.memo[key] = random_expr_float(f'CONFIG_{key}', self.pick(key))
        return self.memo[key]

    def as_floats(self, key):
        return str(self.as_float(key))

    def pick(self, key):
        return pick_expr(f'CONFIG_{key}', self.config[key])

    def pick_array(self, key):
        return pick_array(f'CONFIG_{key}', self.config[key])

    def raw_(self, key):
        return self.config[key]

    def pattern_n(self):
        if not self.picked_all:
            for key in self.config.keys():
                if OPTION_DETAILS.is_array(key):
                    pick_array(f'CONFIG_{key}', self.config[key])
                else:
                    pick_expr(f'CONFIG_{key}', self.config[key])
            self.picked_all = True
        prod_len = 1
        for v in gv.static_index_map.values():
            prod_len *= v[1]
        return prod_len

    def set_product_position(self):

        prod_len = self.pattern_n()
        prod_iter = itertools.product(*[range(0, v[1])
                                      for v in gv.static_index_map.values()])
        ll = list(v[1]
                  for v in gv.static_index_map.values())
        print(f'Config patterns: {prod_len}', file=sys.stderr)
        idx = random.randrange(0, prod_len)
        ipos = idx % prod_len
        indices = list(itertools.islice(prod_iter, ipos, ipos+1))[0]
        for key, i in zip(gv.static_index_map.keys(), indices):
            gv.static_index_map[key][0] = i
        return ipos


def genseed():
    return max(0, random.getrandbits(31)-1)


def main():
    parser = argparse.ArgumentParser(
        description='ランダムなWorldを設定ファイルから生成', epilog='-g か -c いずれかの指定が必要')
    parser.add_argument('-g', '--generate-config',
                        help='設定ファイルを生成する。これを指定すると設定ファイル生成後に実行を終了する。(-cよりも優先)')
    parser.add_argument('-c', '--config',
                        help='設定ファイル(.toml) これを指定すると、設定項目を使用してWorld構成情報を生成し標準出力に出力する。')
    parser.add_argument('-s', '--seed', help='configのシード値を上書きする。')
    parser.add_argument('--summary-output', help='サマリーの出力先', type=str)
    parser.add_argument('--return-pattern-n', action='store_true',
                        help='これを指定すると設定項目の選び方のパターン数を出力し終了する。')
    parser.add_argument('--return-pattern-pos', action='store_true',
                        help='これを指定すると設定項目のどのパターンが選ばれたかを示すインデックスを出力し終了する。')

    args = parser.parse_args()

    if args.generate_config is None and args.config is None:
        parser.print_help(file=sys.stderr)
        return

    if not args.generate_config is None:
        with open(args.generate_config, 'w') as f:
            OPTION_DETAILS.print(file=f)
        return

    c = RandomConfig(toml.load(args.config))
    if args.return_pattern_n:
        print(f'{c.pattern_n()}')
        return

    seed = c.pick('seed')
    if not args.seed is None:
        seed = int(args.seed)

    random.seed(seed)

    if args.return_pattern_pos:
        print(f'{seed} {c.set_product_position()}')
        return

    c.set_product_position()

    # Define fundamental constants
    world_type = c.pick('type')
    if world_type != 'A' and world_type != 'B':
        raise 'world type must be A or B'

    gv.T_max = t_max = c.as_int('T_max')
    worker_num = gv.worker_num = c.as_int('worker_num')
    depth = gv.map_max_depth = c.as_int('map_max_depth')

    #################
    print(world_type)
    #################

    ############
    print(t_max)
    ############

    this_abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(this_abs_path)
    current_dir = os.path.abspath(os.getcwd())
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    build_dir = script_dir
    
    tmp_nodes = os.path.join(tmp_dir, 'nodes.txt')
    tmp_edges = os.path.join(tmp_dir, 'edges.txt')
    tmp_edge2 = os.path.join(tmp_dir, 'edges2.txt')
    tmp_mapimg = os.path.join(tmp_dir, 'map.png')
    print(f'ratio:{c.as_float("map_node_ratio")}', file=sys.stderr)
    max_nodes = int(int(int(4**(depth+1)-1)/3)*c.as_float('map_node_ratio'))
    vertex_num_hard_limit = c.pick_array('map_vertex_num_hard_limit')
    edge_num_hard_limit_coeff = c.pick_array(
        'map_edge_num_hard_limit_coeff')
    while True:
        subprocess.call(
            ['./map_generator', c.as_ints('map_size'), str(max_nodes), str(genseed()), c.as_ints('map_max_depth'), tmp_nodes, tmp_edges, c.as_floats('map_area_ratio'), c.as_ints('map_peak_num')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=build_dir
        )
        node_num = sum(1 for _ in open(tmp_nodes))
        if node_num < vertex_num_hard_limit[0] or node_num > vertex_num_hard_limit[1]:
            print(f'vertex count:{node_num} (skip)', file=sys.stderr)
            continue
        print(f'vertex count:{node_num} (accepted)', file=sys.stderr)
        edge_num = sum(1 for _ in open(tmp_edges))
        if edge_num < edge_num_hard_limit_coeff[0]*node_num or edge_num > edge_num_hard_limit_coeff[1]*node_num:
            print(f'edge count:{edge_num} (skip)', file=sys.stderr)
            continue
        print(f'edge count:{edge_num} (accepted)', file=sys.stderr)
        shutil.copyfile(tmp_nodes, 'debug_nodes.txt')
        shutil.copyfile(tmp_edges, 'debug_edges.txt')

        unit_dist = c.as_int('unit_dist')

        with open(tmp_edges) as f, open(tmp_nodes) as g, open(tmp_edge2, 'w') as e2:
            reader = csv.reader(f, delimiter=' ')
            minv = -1
            edges = []
            for row in reader:
                v = float(row[2])
                if len(edges) == 0:
                    minv = v
                if v < minv:
                    minv = v
                edges.append([int(row[0]), int(row[1]), v])
            #################################
            for w in [sys.stdout, e2]:
                print(f'{node_num} {len(edges)}', file=w)

            graph_mat_dense = np.full((node_num, node_num), np.inf)
            #################################
            reader_g = csv.reader(g, delimiter=' ')
            for row in reader_g:
                ###################################
                print(
                    f'{row[0]} {unit_dist*float(row[1])/minv} {unit_dist*float(row[2])/minv}')
                ###################################
            for e in edges:
                dist = round(unit_dist*e[2]/minv)
                ###################################
                for w in [sys.stdout, e2]:
                    print(f'{e[0]} {e[1]} {dist}', file=w)
                graph_mat_dense[e[0]-1][e[1]-1] = dist
                graph_mat_dense[e[1]-1][e[0]-1] = dist
                ###################################
        break
    sys.stdout.flush()
# num, job_type_num_expr, job_type_expr, vertex_num, task_expr
    worker_info_arr = random_worker(worker_num, c.pick('worker_processable_num'),
                                    c.pick('worker_processable_type'), node_num, c.pick('worker_max_task_processing_num'))
    all_processable_types = set().union(
        *[w.processable_types for w in worker_info_arr])

    sys.stdout.flush()
    ###################################
    random_job.main([
        '-n', c.as_ints('job_num'),
        '-t', str(c.pick('job_type')),
        '--n-task-expr', str(c.pick('task_num')),
        '-v', str(node_num),
        '--penalty-expr', str(c.pick('job_unfinished_penalty')),
        '-m', c.as_ints('mandatory_job_num'),
        '-s', str(genseed()),
        '--reward-interval-expr', str(c.pick('reward_interval')),
        '--reward-duration-expr', str(c.pick('reward_duration')),
        '--reward-stddev', str(c.pick('reward_stddev')),
        '--reward-value', str(c.pick('reward_value')),
        '--reward-upper-limit-expr', str(c.pick('reward_upper_hard_limit')),
        '--reward-lower-limit-expr', str(c.pick('reward_lower_hard_limit')),
        '--dep-nodes-expr', str(c.pick('job_dependency_num')),
        '--max-dep-expr', str(c.pick('job_max_dep')),
        '--t-max', str(t_max),
        '--weather-dependency-expr', str(c.pick('weather_dependency')),
        '--processable-types', *[str(t) for t in all_processable_types]
    ])
    ###################################

    interval = None
    while True:
        interval = random_expr_int('fundamental_weather_interval_length',
                                   c.pick('fundamental_weather_interval_length'))
        if t_max % interval == 0:
            break

    # 天候
    weather_stationary_dist = [random_expr_float(f'weather_stationary_dist_{i}',
                                                 e) for i, e in enumerate(c.pick_array('weather_stationary_dist'))]
    ############################################################################
    print(f'{len(weather_stationary_dist)} {interval} {random.getrandbits(64)}')
    ############################################################################
    sys.stdout.flush()

    ###################################
    subprocess.call(['./trans_prob_mat_generator',
                     '-d', f'{" ".join(str(v) for v in weather_stationary_dist)}',
                     '-r', c.as_ints('weather_prob_nondiag_cutoff_range'),
                     '-q', c.as_floats('weather_prob_sharpness'),
                     '--eps', str(c.pick('weather_prob_eps')),
                     '-c', str(1 if c.pick('weather_prob_centralize') else 0)], cwd=build_dir)
    ###################################

    limit_const = [random_expr_int(f'weather_limit_const_{i}', e) for i, e in enumerate(
        c.pick_array('weather_limit_const'))]
    if world_type == "A":
        limit_const = [0 for v in weather_stationary_dist]

    ####################################
    print(f'{len(limit_const)}', end='')
    for v in limit_const:
        print(f' {v}', end='')
    print()
    ####################################

    ####################################
    random_schedule_penalty(c.pick('schedule_penalty_max'),
                            c.pick('schedule_penalty_decay'), c.pick('schedule_score_scale'))
    ####################################
    sys.stdout.flush()
    if not args.summary_output is None:
        with open(args.summary_output, 'w') as f:
            s = sorted(set(list(gv.static_index_map.keys()) +
                           list(gv.static_value_map.keys())))
            for k in s:
                print(
                    f'{k},{gv.static_index_map[k] if k in gv.static_index_map else -1},{gv.static_value_map[k] if k in gv.static_value_map else -1.0}', file=f)


if __name__ == '__main__':
    main()
