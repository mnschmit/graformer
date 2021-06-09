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


def main(args):
    num_components = []
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
            num_components.append(nx.number_connected_components(graph))

    print(
        'Min: {} | Max: {} | Mean: {} | Std: {}'.format(
            min(num_components), max(num_components),
            np.mean(num_components), np.std(num_components)
        )
    )
    counter = Counter(num_components)
    for k in sorted(counter.keys()):
        print("{}: {}".format(k, counter[k]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='+')
    parser.add_argument('--webnlg', action='store_true')
    parser.add_argument('--ignore-isolated', action='store_true')
    args = parser.parse_args()
    main(args)
