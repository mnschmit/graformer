from typing import Dict, List, Tuple, Sequence, Iterable, Optional
import argparse
import json
from tqdm import tqdm
import logging
from .train_spm_webnlg import ENT_TOKEN, REL_TOKEN  # , inverse_relation
from .distance_matrix import compute_dm, compute_same_ent_idx
from .spm_utils import tokenize_entities

import networkx as nx
import sentencepiece as spm
from itertools import combinations
from collections import OrderedDict, defaultdict
from unidecode import unidecode
import re


def preprocess_text(text: str):
    res = unidecode(text.lower())
    res = ' '.join(re.split(r'(\W)', res))
    res = ' '.join(res.split())
    return res


def preprocess_entity(ent: str, token_sep=' ', prep: bool = True) -> str:
    res = ent.replace('_', token_sep)
    if prep:
        res = preprocess_text(res)
    return res


def preprocess_relation(rel: str, token_sep=' ', prep: bool = True) -> str:
    res_buffer = []
    have_to_lowercase = False
    for char in rel:
        if char == '_':
            res_buffer.append(token_sep)
            have_to_lowercase = True
        elif char.isupper() and have_to_lowercase:
            if res_buffer[-1] != token_sep:
                res_buffer.append(token_sep)
            res_buffer.append(char.lower())
        else:
            res_buffer.append(char)
            have_to_lowercase = char.islower()
    res = ''.join(res_buffer)
    if prep:
        res = preprocess_text(res)
    return res


def graph_gen_helper(
        triples: Iterable[Tuple[str, str, str]],
        entities: Sequence[str],
        sp: spm.SentencePieceProcessor,
        dropout_p: Optional[float] = None
) -> Tuple[Tuple[List[List[int]], List[bool], List[int]], List[List[int]]]:
    graph = nx.DiGraph()

    # Indexed labels for entity nodes
    ent2nodeIds, token_node_labels, positions = tokenize_entities(
        entities, sp, sample_alpha=dropout_p)
    same_ent_idx = compute_same_ent_idx(ent2nodeIds)

    # (1) Entity Nodes
    graph.add_nodes_from(range(len(token_node_labels)))

    do_sample = dropout_p is not None

    # (2)+(3) Relations and Edges
    rel_node_labels = []
    for trip in triples:
        e1 = trip[0]
        e2 = trip[2]
        rel_label = trip[1]

        num_nodes_before = len(graph)
        rel_token_indices = sp.encode(
            rel_label, enable_sampling=do_sample, alpha=dropout_p)
        rel_node_labels.extend(rel_token_indices)
        rel_nodes = list(
            range(num_nodes_before, num_nodes_before+len(rel_token_indices)))
        graph.add_nodes_from(rel_nodes)

        # same text links
        for (i1, n1), (i2, n2) in combinations(enumerate(rel_nodes), 2):
            same_ent_idx[(n1, n2)] = i2 - i1
            same_ent_idx[(n2, n1)] = i1 - i2

        # edges
        for reln in rel_nodes:
            for n1 in ent2nodeIds[e1]:
                graph.add_edge(n1, reln)
            for n2 in ent2nodeIds[e2]:
                graph.add_edge(reln, n2)

    distance_matrix = compute_dm(graph, same_ent_idx)
    is_entity = [True] * (len(token_node_labels) + len(rel_node_labels))

    return (distance_matrix, is_entity, positions), token_node_labels + rel_node_labels


def generate_graph(
        triples: List[Dict[str, str]],
        sp,
        extended_preprocessing: bool,
        bpe_dropout: Optional[float]
) -> Tuple[Tuple[List[List[int]], List[bool], List[int]], List[List[int]]]:
    entities = OrderedDict()
    preprocessed_triples = []
    for t in triples:
        subj = ENT_TOKEN + ' ' + \
            preprocess_entity(t["subject"], prep=extended_preprocessing)
        obj = ENT_TOKEN + ' ' + \
            preprocess_entity(t["object"], prep=extended_preprocessing)
        rel = REL_TOKEN + ' ' + \
            preprocess_relation(t["property"], prep=extended_preprocessing)
        entities[subj] = None
        entities[obj] = None
        preprocessed_triples.append((subj, rel, obj))
        # preprocessed_triples.append((obj, inverse_relation(rel), subj))
    entities = list(entities.keys())

    graph, node_labels = graph_gen_helper(
        preprocessed_triples, entities, sp, bpe_dropout)

    return graph, node_labels


def add_string_to_graph(graph: nx.DiGraph, string: str, sp, bpe_dropout: Optional[float]):
    num_nodes_before = len(graph)
    token_indices = sp.encode(
        string, enable_sampling=bpe_dropout is not None, alpha=bpe_dropout)
    nodes = list(
        range(num_nodes_before, num_nodes_before+len(token_indices)))
    graph.add_nodes_from(nodes)
    for n1, n2 in zip(nodes[:-1], nodes[1:]):
        graph.add_edge(n1, n2)
    return nodes, token_indices


def generate_sequence_graph(
        triples: List[Dict[str, str]],
        sp,
        extended_preprocessing: bool,
        bpe_dropout: Optional[float]
) -> Tuple[Tuple[List[List[int]], List[bool], List[int]], List[List[int]]]:
    graph = nx.DiGraph()

    preprocessed_triples = []
    for t in triples:
        subj = ENT_TOKEN + ' ' + \
            preprocess_entity(t["subject"], prep=extended_preprocessing)
        obj = ENT_TOKEN + ' ' + \
            preprocess_entity(t["object"], prep=extended_preprocessing)
        rel = REL_TOKEN + ' ' + \
            preprocess_relation(t["property"], prep=extended_preprocessing)
        preprocessed_triples.append((subj, rel, obj))
        # preprocessed_triples.append((obj, inverse_relation(rel), subj))

    node_labels = []
    prev_trip_node = None
    for trip in preprocessed_triples:
        e1 = trip[0]
        e2 = trip[2]
        rel_label = trip[1]

        e1_node_ids, e1_token_indices = add_string_to_graph(graph, e1, sp, bpe_dropout)
        node_labels.extend(e1_token_indices)
        rel_node_ids, rel_token_indices = add_string_to_graph(
            graph, rel_label, sp, bpe_dropout)
        node_labels.extend(rel_token_indices)
        e2_node_ids, e2_token_indices = add_string_to_graph(graph, e2, sp, bpe_dropout)
        node_labels.extend(e2_token_indices)

        if prev_trip_node is not None:
            graph.add_edge(prev_trip_node, e1_node_ids[0])

        graph.add_edge(e1_node_ids[-1], rel_node_ids[0])
        graph.add_edge(rel_node_ids[-1], e2_node_ids[0])

        prev_trip_node = e2_node_ids[-1]

    distance_matrix = compute_dm(graph, defaultdict(int))
    is_entity = [True] * len(node_labels)
    positions = list(range(len(node_labels)))

    return (distance_matrix, is_entity, positions), node_labels


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Loading json file')
    with open(args.json_in) as f:
        nlg = json.load(f)

    logger.info('Loading sentencepiece model')
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    # sp.SetEncodeExtraOptions('bos:eos')

    logger.info('Generating graphs')
    dms: List[List[List[int]]] = []
    is_entity: List[List[bool]] = []
    positions: List[List[int]] = []

    node_labels: List[List[int]] = []
    targets: List[List[str]] = []
    for entry in tqdm(nlg['entries']):
        for e in entry.values():
            for i in range(args.sample_factor):
                if args.sequence:
                    graph, nodes = generate_sequence_graph(
                        e["modifiedtripleset"], sp, args.extended_preprocessing,
                        args.bpe_dropout
                    )
                else:
                    graph, nodes = generate_graph(
                        e["modifiedtripleset"], sp, args.extended_preprocessing,
                        args.bpe_dropout
                    )

                dms.append(graph[0])
                is_entity.append(graph[1])
                positions.append(graph[2])
                node_labels.append(nodes)
                targets.append([])
                for lex in e["lexicalisations"]:
                    target_text = lex['lex']
                    if args.extended_preprocessing:
                        target_text = preprocess_text(target_text)
                    targets[-1].append(target_text)

    logger.info('Indexing target texts')
    indexed_targets = [
        # index_strings(target_list, sp)
        sp.encode(
            target_list, add_bos=True, add_eos=True,
            enable_sampling=args.bpe_dropout is not None,
            alpha=args.bpe_dropout
        )
        for target_list in targets
    ]

    text_data = {
        "node_label": node_labels,
        "target": indexed_targets
    }
    graphs = {
        "distance_matrix": dms,
        "is_entity": is_entity,
        "positions": positions
    }

    logger.info('Storing graphs to disk')
    with open(args.graph_out, 'w') as fout:
        json.dump(graphs, fout)
    logger.info('Writing metadata to disk')
    with open(args.text_out, 'w') as fout:
        json.dump(text_data, fout)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('json_in')
    parser.add_argument('graph_out')
    parser.add_argument('text_out')
    parser.add_argument('spm_model')
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--simple-preprocessing',
                        action='store_false', dest='extended_preprocessing')
    parser.add_argument('--bpe-dropout', type=float, default=None)
    parser.add_argument('--sample-factor', type=int, default=1)

    args = parser.parse_args()
    main(args)
