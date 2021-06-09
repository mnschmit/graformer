from typing import List

import argparse
import json
import numpy as np
from tqdm import tqdm


def print_stats(lengths: List[int], title: str):
    print('{} statistics'.format(title))
    print('min {} | max {} | mean {:.4f} | std {:.4f}'.format(
        np.min(lengths), np.max(lengths), np.mean(lengths), np.std(lengths)
    ))


def extract_lengths(json_dict, text_key, multiple_references):
    lengths = []
    for article in tqdm(json_dict[text_key]):
        if multiple_references:
            for t in article:
                lengths.append(len(t))
        else:
            lengths.append(len(article))
    return lengths


def extract_node_stats(json_dict):
    num_nodes = []
    lengths = []
    for x in tqdm(json_dict['node_label']):
        num_nodes.append(len(x))
        for node_label in x:
            try:
                lengths.append(len(node_label))
            except TypeError:
                lengths.append(1)
    return num_nodes, lengths


def compute_tig(node_labels: List[int], text: List[int]):
    graph_tokens = set(node_labels)
    num_in_graph = 0
    for text_token in text:
        if text_token in graph_tokens:
            num_in_graph += 1
    return num_in_graph / len(text)


def compute_git(node_labels: List[int], text: List[int]):
    text_tokens = set(text)
    num_in_text = 0
    for graph_token in node_labels:
        if graph_token in text_tokens:
            num_in_text += 1
    return num_in_text / len(node_labels)


def get_percentages(json_dict, text_key: str, multiple_references: bool, compute):
    percentages = []
    for node_labels, texts in tqdm(zip(json_dict['node_label'], json_dict[text_key])):
        if multiple_references:
            for text in texts:
                percentages.append(compute(node_labels, text))
        else:
            percentages.append(compute(node_labels, texts))
    return percentages


def main(args: argparse.Namespace):
    text_key = 'abstract'
    if args.VG:
        text_key = 'caption'
    elif args.nlg:
        text_key = 'target'

    lengths = []
    num_nodes = []
    node_label_lengths = []
    percentage_text_in_graph = []
    percentage_graph_in_text = []
    for json_file in args.agenda_json:
        with open(json_file) as f:
            agenda = json.load(f)

        lengths.extend(extract_lengths(agenda, text_key, args.nlg))
        tmp_num_nodes, tmp_nll = extract_node_stats(agenda)
        num_nodes.extend(tmp_num_nodes)
        node_label_lengths.extend(tmp_nll)
        percentage_text_in_graph.extend(
            get_percentages(agenda, text_key, args.nlg, compute_tig)
        )
        percentage_graph_in_text.extend(
            get_percentages(agenda, text_key, args.nlg, compute_git)
        )

    if args.count_references:
        print('Number of reference texts:', len(lengths))
    else:
        print('Number of samples (with potentially more than 1 reference text each):', len(
            num_nodes))

    print_stats(lengths, text_key)
    print_stats(num_nodes, 'num nodes')
    print_stats(node_label_lengths, 'node label')
    print_stats(percentage_text_in_graph, '% text tokens in graph')
    print_stats(percentage_graph_in_text, '% graph tokens in text')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agenda_json', nargs='+')
    parser.add_argument('--VG', action='store_true')
    parser.add_argument('--nlg', action='store_true')
    parser.add_argument('--count-references', action='store_true')

    args = parser.parse_args()
    main(args)
