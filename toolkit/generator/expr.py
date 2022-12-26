import numexpr as ne
import math
import random
import global_value as gv
from typing import List
import collections
import numpy as np


def arr(a, b, c, i):
    return [a, b, c][i]


def random_expr_int(name: str, e: any):
    x = random.random()
    if not name in gv.static_value_map:
        gv.static_value_map[name] = random.random()
    g = gv.static_value_map[name]
    worker_num = gv.worker_num
    T_max = gv.T_max
    map_max_depth = gv.map_max_depth
    return math.floor(ne.evaluate(str(e)))


def random_expr_float(name: str, e: any):
    x = random.random()
    if not name in gv.static_value_map:
        gv.static_value_map[name] = random.random()
    g = gv.static_value_map[name]
    worker_num = gv.worker_num
    T_max = gv.T_max
    map_max_depth = gv.map_max_depth
    return ne.evaluate(str(e))


def pick_expr(name: str, e: any):
    is_list = isinstance(e, (collections.abc.Sequence, np.ndarray)
                         ) and not isinstance(e, str)
    if not name in gv.static_index_map:
        gv.static_index_map[name] = [random.randrange(
            0, len(e)), len(e)] if is_list else [0, 1]
    if is_list:
        return e[gv.static_index_map[name][0]]
    else:
        return e


def pick_array(name: str, e: any):
    is_list = isinstance(e, (collections.abc.Sequence, np.ndarray)
                         ) and not isinstance(e, str)
    if not is_list or len(e) == 0:
        raise 'expected a non-empty array but it was not'

    is_elem_list = all(isinstance(v, (collections.abc.Sequence, np.ndarray)
                                  ) and not isinstance(v, str) for v in e)

    if not is_elem_list:
        raise 'expected a nested array but it was not'

    if not name in gv.static_index_map:
        gv.static_index_map[name] = [random.randrange(
            0, len(e)), len(e)]

    return e[gv.static_index_map[name][0]]
