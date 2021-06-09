import argparse
import networkx as nx
from tqdm import tqdm
import json
import numpy as np
from ..preprocessing.prepare_agenda import extract_triple
from collections import Counter


def generate_graph(triples):
    graph = nx.Graph()
    for trip in triples:
        sbj, pred, obj = trip
        graph.add_node(sbj)
        graph.add_node(obj)
        graph.add_edge(sbj, obj)
    return graph


def generate_triples_webnlg(triples):
    res = []
    for trip in triples:
        res.append((trip['subject'], trip['property'], trip['object']))
    return res


def generate_triples_agenda(triples):
    res = []
    for trip in triples:
        res.append(extract_triple(trip))
    return res


def aggregate_bins(counter, limits):
    res = {}
    for max_lim in limits:
        res[max_lim] = 0
    res['rest'] = 0

    for avg_size, frq in counter.items():
        found_bucket = False
        for l in limits:
            if avg_size < l:
                res[l] += frq
                found_bucket = True
                break
        if not found_bucket:
            res['rest'] += frq

    return res


def main(args):
    avg_cc_sizes = []
    for inp_file in args.input_file:
        with open(inp_file) as f:
            data_in = json.load(f)

        if args.webnlg:
            triples = []
            for entry in tqdm(data_in['entries']):
                for e in entry.values():
                    triples.append(generate_triples_webnlg(
                        e["modifiedtripleset"]))
        else:
            triples = []
            for article in tqdm(data_in):
                triples.append(generate_triples_agenda(article["relations"]))
                if not args.ignore_isolated:
                    for entity in article["entities"]:
                        triples[-1].append((entity, "", entity))

        for tripset in triples:
            graph = generate_graph(tripset)
            num_ccs = nx.number_connected_components(graph)
            num_nodes = len(graph)
            avg_cc_sizes.append(num_nodes / num_ccs)

    print(
        'Min: {} | Max: {} | Mean: {} | Std: {}'.format(
            min(avg_cc_sizes), max(avg_cc_sizes),
            np.mean(avg_cc_sizes), np.std(avg_cc_sizes)
        )
    )
    counter = Counter(avg_cc_sizes)

    if args.aggregate:
        counter = aggregate_bins(counter, args.aggregate)

    for k in sorted(counter.keys(), key=lambda x: float('inf') if x == 'rest' else x):
        print("{}: {}".format(k, counter[k]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='+')
    parser.add_argument('--webnlg', action='store_true')
    parser.add_argument('--ignore-isolated', action='store_true')
    parser.add_argument('--aggregate', '-a', nargs='*', type=float)
    args = parser.parse_args()
    main(args)
