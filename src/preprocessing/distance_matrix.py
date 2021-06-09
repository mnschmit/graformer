from typing import Mapping, Sequence, List, Dict, Optional, Tuple
import math
import networkx as nx
from collections import defaultdict
from itertools import combinations


def pos_idx(d: int) -> int:
    return 2 * d  # before: - 1


def neg_idx(d: int) -> int:
    return 2 * d + 1


def simple_node_relation(source: int, target: int, length: Mapping[int, Mapping[int, int]]) -> int:
    unreachable = neg_idx(0)

    dpos = length[source].get(target, float('inf'))
    dneg = length[target].get(source, float('inf'))

    if dneg < dpos:
        return neg_idx(dneg)
    elif math.isinf(dpos):
        return unreachable
    else:
        return pos_idx(dpos)


def index_node_relation(source: int, target: int, length: Mapping[int, Mapping[int, int]],
                        same_ent_idx: Mapping[Tuple[int, int], int], same_text_num: int) -> int:
    unreachable = neg_idx(0)
    # same_node = 1
    # same_ent = 2
    # offset = 1  # 2

    # if source == target:
    #     return same_node

    same_ent_score = same_ent_idx[(source, target)]
    if same_ent_score != 0:
        # return same_ent
        if same_ent_score < 0:
            same_ent_score = neg_idx(-same_ent_score) - 1
        else:
            same_ent_score = pos_idx(same_ent_score) - 1
        return same_text_num + same_ent_score

    dpos = length[source].get(target, float('inf'))
    dneg = length[target].get(source, float('inf'))

    if dneg < dpos:
        return neg_idx(dneg)
    elif math.isinf(dpos):
        return unreachable
    else:
        return pos_idx(dpos)


def compute_simple_dm(graph: nx.Graph) -> List[List[int]]:
    length = dict(nx.all_pairs_shortest_path_length(graph))
    matrix = []
    for source in graph:
        row = []
        for target in graph:
            row.append(simple_node_relation(source, target, length))
        matrix.append(row)
    return matrix


def compute_dm(graph: nx.Graph, same_ent_idx: Mapping[Tuple[int, int], int],
               title: Optional[Sequence[int]] = None) -> List[List[int]]:
    title_offset = 0 if title is None else 2
    title = set() if title is None else set(title)

    length = dict(nx.all_pairs_shortest_path_length(graph))
    matrix = []
    for source in graph:
        source_is_title = source in title
        row = []
        for target in graph:
            # NB: ugly hard-coded 1000 gives an offset for same-text pos embeddings
            node_rel_idx = index_node_relation(
                source, target, length, same_ent_idx, 1000)
            if node_rel_idx < 1000:
                node_rel_idx += title_offset
            if source_is_title and target not in title:
                node_rel_idx = 0
            if target in title and not source_is_title:
                node_rel_idx = 1
            row.append(node_rel_idx)
        matrix.append(row)
    return matrix


def compute_same_ent_idx(
        ent2nodeIds: Mapping[str, Sequence[int]]) -> Dict[Tuple[int, int], int]:
    res = defaultdict(int)  # 0 means 'not same ent'
    for ent, nodes in ent2nodeIds.items():
        for (i1, n1), (i2, n2) in combinations(enumerate(nodes), 2):
            res[(n1, n2)] = i2 - i1
            res[(n2, n1)] = i1 - i2
    return res
