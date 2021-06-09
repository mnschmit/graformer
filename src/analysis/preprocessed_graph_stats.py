from typing import List
import numpy as np
import json
import argparse
from tqdm import tqdm
import logging


def print_stats(lengths: List[int], title: str):
    print('{} statistics'.format(title))
    if lengths:
        print('min {} | max {} | mean {:.4f} | std {:.4f}'.format(
            np.min(lengths), np.max(lengths), np.mean(lengths), np.std(lengths)
        ))
    else:
        print('EMPTY')


def idx2len(idx: int, offset=1) -> int:
    idx -= offset
    if idx % 2 == 0:
        return idx // 2
    else:
        return -(idx // 2)


def main(args: argparse.Namespace):
    logger = logging.getLogger(__name__)

    if args.title_graph:
        offset = 2
    else:
        offset = 0

    logger.info('Loading json file')
    with open(args.graph_json) as f:
        graphs = json.load(f)

    logger.info('Extracting path lengths')
    path_nonneg = []
    path_neg = []
    ent_nonneg = []
    ent_neg = []
    for matrix in tqdm(graphs['distance_matrix']):
        for row in matrix:
            for e in row:
                if e > args.same_text_num:
                    ent_len = idx2len(e - args.same_text_num, offset=0)
                    if ent_len < 0:
                        ent_neg.append(ent_len)
                    else:
                        ent_nonneg.append(ent_len)
                else:
                    path_len = idx2len(e, offset=offset)
                    if path_len < 0:
                        path_neg.append(path_len)
                    else:
                        path_nonneg.append(path_len)

    logger.info('Computing path statistics')
    print_stats(path_nonneg, 'nonnegative path index')
    print_stats(path_neg, 'negative path index')
    print_stats(ent_nonneg, 'nonnegative same entity index')
    print_stats(ent_neg, 'negative same entity index')

    if 'positions' in graphs:
        logger.info('Analyzing entity token positions')
        max_pos = -1
        num_entities = []
        for pos_vector in tqdm(graphs['positions']):
            num_entities.append(pos_vector.count(0))
            for pos in pos_vector:
                if pos > max_pos:
                    max_pos = pos
        print('Maximum entity length:', max_pos + 1)
        print('Mean num entities:', sum(num_entities) / len(num_entities))
    else:
        print('No entity token position information present')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_json')
    parser.add_argument('--same-text-num', type=int, default=1000)
    parser.add_argument('--title-graph', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
