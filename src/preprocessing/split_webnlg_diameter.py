from typing import Dict, List
import argparse
import json
from tqdm import tqdm
import logging

import networkx as nx


def compute_diameter(triples: List[Dict[str, str]], max_value=3) -> int:
    graph = nx.Graph()
    for trip in triples:
        sbj, obj = trip['subject'], trip['object']
        graph.add_node(sbj)
        graph.add_node(obj)
        graph.add_edge(sbj, obj)

    try:
        res = nx.diameter(graph)
    except nx.exception.NetworkXError:
        res = max_value

    return res


def sort_diameter(nlg):
    b1, b2, b3 = [], [], []
    for entry in tqdm(nlg['entries']):
        for e in entry.values():
            diameter = compute_diameter(e["modifiedtripleset"])
            if diameter == 1:
                b1.append(entry)
            elif diameter == 2:
                b2.append(entry)
            elif diameter >= 3:
                b3.append(entry)
            else:
                raise RuntimeError("Wrong assumption about diameter range")
    return [b1, b2, b3]


def sort_num_triples(nlg):
    b1, b2, b3, b4, b5, b6, b7 = [], [], [], [], [], [], []
    for entry in tqdm(nlg['entries']):
        for e in entry.values():
            num_trip = len(e["modifiedtripleset"])
            if num_trip == 1:
                b1.append(entry)
            elif num_trip == 2:
                b2.append(entry)
            elif num_trip == 3:
                b3.append(entry)
            elif num_trip == 4:
                b4.append(entry)
            elif num_trip == 5:
                b5.append(entry)
            elif num_trip == 6:
                b6.append(entry)
            elif num_trip == 7:
                b7.append(entry)
            else:
                raise RuntimeError("Wrong assumption about num_trip range")
    return [b1, b2, b3, b4, b5, b6, b7]


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Loading json file')
    with open(args.json_in) as f:
        nlg = json.load(f)

    if args.num_triples:
        assert len(args.json_out) == 7
        logger.info('Sorting by number of triples')
        bins = sort_num_triples(nlg)
    else:
        assert len(args.json_out) == 3
        logger.info('Sorting by diameter')
        bins = sort_diameter(nlg)

    logger.info('Writing out split jsons')
    for out_fn, bucket in zip(args.json_out, bins):
        with open(out_fn, 'w') as fout:
            json.dump({'entries': bucket}, fout)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('json_in')
    parser.add_argument('json_out', nargs='+')
    parser.add_argument('--num-triples', action='store_true')

    args = parser.parse_args()
    main(args)
