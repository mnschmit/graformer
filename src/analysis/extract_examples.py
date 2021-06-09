import argparse
from collections import defaultdict
import json
from tqdm import tqdm
import networkx as nx
import random
from .count_cc_size import generate_triples_webnlg, generate_triples_agenda, generate_graph


def determine_bin(value2data, min_val, max_val):
    interesting_bin = []
    for val, data in value2data.items():
        if min_val <= val < max_val:
            for d in data:
                interesting_bin.append((val, d))
    return interesting_bin


def main(args):
    value2data = defaultdict(list)
    with open(args.input_file) as f:
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
        value = num_nodes / num_ccs
        value2data[value].append(tripset)

    interesting_bin = determine_bin(value2data, args.min_value, args.max_value)
    print('Found {} matching graphs.'.format(len(interesting_bin)))
    sample = random.sample(interesting_bin, args.sample_size)
    for val, tripset in sorted(sample, key=lambda x: x[0]):
        print("Value:", val)
        for trip in tripset:
            print("    ", trip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('min_value', type=float)
    parser.add_argument('max_value', type=float)
    parser.add_argument('--webnlg', action='store_true')
    parser.add_argument('--ignore-isolated', action='store_true')
    parser.add_argument('--sample-size', type=int, default=10)
    args = parser.parse_args()

    main(args)
