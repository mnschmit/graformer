import argparse
import json
import networkx as nx
from .count_cc_size import generate_triples_webnlg, generate_triples_agenda, generate_graph


def main(args):
    with open(args.input_file) as f:
        data_in = json.load(f)

    if args.webnlg:
        entry = data_in['entries'][args.sample_num]
        for e in entry.values():
            tripset = generate_triples_webnlg(e["modifiedtripleset"])
    else:
        article = data_in[args.sample_num]
        tripset = generate_triples_agenda(article["relations"])
        if not args.ignore_isolated:
            for entity in article["entities"]:
                tripset.append((entity, "", entity))

    graph = generate_graph(tripset)
    num_ccs = nx.number_connected_components(graph)
    num_nodes = len(graph)
    mean_cc_size = num_nodes / num_ccs

    diameters = []
    for subgraph in [graph.subgraph(cc).copy() for cc in nx.connected_components(graph)]:
        diameters.append(nx.diameter(subgraph))

    print('Num nodes:', num_nodes)
    print('Num CCs:', num_ccs)
    print('Mean CC size:', mean_cc_size)
    print('Max diameter:', max(diameters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('sample_num', type=int)
    parser.add_argument('--webnlg', action='store_true')
    parser.add_argument('--ignore-isolated', action='store_true')
    args = parser.parse_args()

    main(args)
