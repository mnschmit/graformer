from typing import List, Iterable
import argparse
import json
from tqdm import tqdm
import logging
from collections import defaultdict
import os
from .prepare_agenda import extract_triple

import networkx as nx


def compute_avg_cc_size(triples: List[str], isolated_nodes: Iterable[str]) -> int:
    graph = nx.Graph()

    graph.add_nodes_from(isolated_nodes)

    for trip in triples:
        sbj, rel, obj = extract_triple(trip)
        graph.add_node(sbj)
        graph.add_node(obj)
        graph.add_edge(sbj, obj)

    res = len(graph) / nx.number_connected_components(graph)

    return res


def sort_into_buckets(agenda, ignore_isolated=False):
    buckets = defaultdict(list)
    for article in tqdm(agenda):
        if ignore_isolated:
            nodes = []
        else:
            nodes = article['entities']

        num_components = compute_avg_cc_size(article["relations"], nodes)
        buckets[num_components].append(article)
    return buckets


def aggregate_bins(buckets, limits):
    res = {}
    for max_lim in limits:
        res[max_lim] = []
    res['rest'] = []

    for avg_size, bucket in buckets.items():
        found_bucket = False
        for l in limits:
            if avg_size < l:
                res[l].extend(bucket)
                found_bucket = True
                break
        if not found_bucket:
            res['rest'].extend(bucket)

    return res


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Loading json file')
    with open(args.json_in) as f:
        agenda = json.load(f)

    logger.info('Start sorting')
    buckets = sort_into_buckets(agenda, ignore_isolated=args.ignore_isolated)

    if args.aggregate:
        buckets = aggregate_bins(buckets, sorted(args.aggregate))

    logger.info('Writing out split jsons')
    for n_comp, bucket in buckets.items():
        out_fn = os.path.join(
            args.out_dir, "num_components_split_{}.json".format(n_comp))
        with open(out_fn, 'w') as fout:
            json.dump(bucket, fout)
        logger.info("Successfully written {}".format(out_fn))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('json_in')
    parser.add_argument('out_dir')
    parser.add_argument('--ignore-isolated', action='store_true')
    parser.add_argument('--aggregate', type=float, nargs='*')

    args = parser.parse_args()
    main(args)
