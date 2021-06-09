import argparse
from typing import List
import numpy as np
import logging
import json
from tqdm import tqdm


def print_stats(lengths: List[int], title: str):
    print('{} statistics'.format(title))
    print('min {} | max {} | mean {:.4f} | std {:.4f}'.format(
        np.min(lengths), np.max(lengths), np.mean(lengths), np.std(lengths)
    ))


def extract_num_entities(graphs):
    num_entities = []
    for pos_vector in tqdm(graphs['positions']):
        num_entities.append(pos_vector.count(0))
    return num_entities


def main(args):
    logger = logging.getLogger(__name__)
    num_entities = []
    for json_file in args.graph_json:
        logger.info('Loading json file')
        with open(json_file) as f:
            graphs = json.load(f)
        num_entities.extend(extract_num_entities(graphs))
    print_stats(num_entities, 'num entities')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_json', nargs='+')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
