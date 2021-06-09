from typing import Dict, List, Union, Tuple, Optional
import argparse
import json
from tqdm import tqdm
import logging
from .train_spm import AGENDA_RELATIONS, TYPE_REL, agendalize,\
    inverse_relation, ENT_TOKEN, REL_TOKEN
from .distance_matrix import compute_dm, compute_same_ent_idx
from .spm_utils import tokenize_entities  # , index_strings

import networkx as nx

import sentencepiece as spm


def extract_triple(rel_triple: str) -> Tuple[str, str, str]:
    parts = rel_triple.split(' -- ')
    if len(parts) == 3:
        n1, rel, n2 = parts
    else:
        n1, rel, n2 = None, None, None
        for i, p in enumerate(parts):
            if p in AGENDA_RELATIONS:
                assert not any(x in AGENDA_RELATIONS for x in parts[i+1:])
                n1 = ' -- '.join(parts[:i])
                rel = p
                n2 = ' -- '.join(parts[i+1:])
                break
        if rel is None:
            logging.getLogger(__name__).error(
                'Problem with relation triple "{}"'.format(rel_triple))
            exit(-1)

    return n1, rel, n2


def preprocess_entity(ent: str) -> str:
    return ENT_TOKEN + ' ' + ent.lower()


def generate_graph(
        article: Dict[str, Union[str, List[str]]],
        sp,
        bpe_dropout: Optional[float],
        title_graph: bool,
        add_inverse_relations: bool = False
) -> Tuple[Tuple[List[List[int]], List[bool], List[int]], List[List[int]]]:
    graph = nx.DiGraph()

    entities = [preprocess_entity(e) for e in article['entities']]

    tokenize_input = entities
    if title_graph:
        title = article['title'].lower()
        tokenize_input.append(title)

    ent2nodeIds, token_node_labels, positions = tokenize_entities(
        tokenize_input, sp, sample_alpha=bpe_dropout
    )
    same_ent_idx = compute_same_ent_idx(ent2nodeIds)

    # Labels for type nodes
    ent_type_labels_str = article['types'].split()

    num_entities = positions.count(0)
    if title_graph:
        num_entities -= 1
    if num_entities != len(ent_type_labels_str):
        print(
            'ERROR: Entity without type', "#Types:",
            len(ent_type_labels_str), "#Entities:", num_entities)
        print(json.dumps(article, indent=4, sort_keys=True))
        exit(-1)

    # (1) Entity Nodes
    graph.add_nodes_from(range(len(token_node_labels)))

    # (2) Nodes for relations
    rel_node_labels: List[List[int]] = []
    next_rel_node_id = len(token_node_labels)
    for rel_triple in article['relations']:
        e1, rel_label, e2 = extract_triple(rel_triple)

        # relation node
        rel_node_label = REL_TOKEN + agendalize(rel_label)
        # all relations REL are part of the vocabulary; _REL
        new_rel_node_labels = sp.encode(rel_node_label)
        rel_node_labels.extend(new_rel_node_labels)
        new_rel_node_ids = []
        for nl in new_rel_node_labels:
            graph.add_node(next_rel_node_id)
            new_rel_node_ids.append(next_rel_node_id)
            next_rel_node_id += 1

        for rel_node in new_rel_node_ids:
            for n1 in ent2nodeIds[preprocess_entity(e1)]:
                graph.add_edge(n1, rel_node)
            for n2 in ent2nodeIds[preprocess_entity(e2)]:
                graph.add_edge(rel_node, n2)

        if add_inverse_relations:
            # inverse relation node
            rel_node_label = REL_TOKEN + \
                agendalize(inverse_relation(rel_label))
            new_rel_node_labels = sp.encode(rel_node_label)
            rel_node_labels.extend(new_rel_node_labels)
            new_rel_node_ids = []
            for nl in new_rel_node_labels:
                graph.add_node(next_rel_node_id)
                new_rel_node_ids.append(next_rel_node_id)
                next_rel_node_id += 1

            for rel_node in new_rel_node_ids:
                for n1 in ent2nodeIds[preprocess_entity(e1)]:
                    graph.add_edge(rel_node, n1)
                for n2 in ent2nodeIds[preprocess_entity(e2)]:
                    graph.add_edge(n2, rel_node)

    num_relation_nodes_needed = len(ent_type_labels_str)
    num_other_type = ent_type_labels_str.count('<otherscientificterm>')
    num_relation_nodes_needed -= num_other_type
    if add_inverse_relations:
        num_relation_nodes_needed *= 2

    graph.add_nodes_from(
        range(next_rel_node_id, next_rel_node_id + num_relation_nodes_needed))
    next_type_node_id = next_rel_node_id + num_relation_nodes_needed
    type_to_graph_node: Dict[str, int] = {}
    ent_type_labels: List[int] = []
    # (3) Nodes for types
    for ent_label, type_label in zip(entities, ent_type_labels_str):
        if type_label == '<otherscientificterm>':
            continue

        if type_label in type_to_graph_node:
            type_node = type_to_graph_node[type_label]
        else:
            graph.add_node(next_type_node_id)
            type_node_label = agendalize(type_label)
            # type label TYPE is in vocabulary; _TYPE
            ent_type_labels.append(sp.encode(type_node_label)[-1])
            type_to_graph_node[type_label] = next_type_node_id
            type_node = next_type_node_id
            next_type_node_id += 1

        # IS-A relation
        rel_node_label = agendalize(TYPE_REL)
        # HAS_TYPE is in vocabulary; _<HAS-TYPE>
        rel_node_labels.append(sp.encode(rel_node_label)[-1])
        for ent_node_id in ent2nodeIds[ent_label]:
            graph.add_edge(ent_node_id, next_rel_node_id)
        graph.add_edge(next_rel_node_id, type_node)
        next_rel_node_id += 1

        # inverse IS-A relation
        if add_inverse_relations:
            rel_node_label = agendalize(inverse_relation(TYPE_REL))
            rel_node_labels.append(sp.encode(rel_node_label)[-1])
            for ent_node_id in ent2nodeIds[ent_label]:
                graph.add_edge(next_rel_node_id, ent_node_id)
            graph.add_edge(type_node, next_rel_node_id)
            next_rel_node_id += 1

    if title_graph:
        title_nodes = ent2nodeIds[title]
    else:
        title_nodes = None

    distance_matrix = compute_dm(graph, same_ent_idx, title=title_nodes)
    is_entity = [True for _ in token_node_labels] + \
        [False for _ in rel_node_labels + ent_type_labels]

    return (distance_matrix, is_entity, positions),\
        token_node_labels + rel_node_labels + ent_type_labels


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Loading json file')
    with open(args.json_file) as f:
        agenda = json.load(f)

    logger.info('Loading sentencepiece model')
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    logger.info('Generating graphs')
    titles: List[str] = []

    dms: List[List[List[int]]] = []
    is_entity: List[List[bool]] = []
    positions: List[List[int]] = []

    node_labels: List[List[int]] = []
    abstracts: List[str] = []
    for article in tqdm(agenda):
        for i in range(args.sample_factor):
            titles.append(article['title'].lower())
            graph, nodes = generate_graph(
                article, sp, args.bpe_dropout,
                args.title_graph,
                add_inverse_relations=args.add_inverse
            )
            dms.append(graph[0])
            is_entity.append(graph[1])
            positions.append(graph[2])
            node_labels.append(nodes)
            abstracts.append(article['abstract_og'].lower())

    logger.info('Indexing titles and abstracts')
    # index_strings(titles, sp)
    indexed_titles = sp.encode(
        titles, alpha=args.bpe_dropout,
        enable_sampling=args.bpe_dropout is not None,
        add_bos=True, add_eos=True
    )
    # index_strings(abstracts, sp)
    indexed_abstracts = sp.encode(
        abstracts, alpha=args.bpe_dropout,
        enable_sampling=args.bpe_dropout is not None,
        add_bos=True, add_eos=True
    )

    metadata = {
        "title": indexed_titles,
        "abstract": indexed_abstracts,
        "node_label": node_labels
    }
    graphs = {
        "distance_matrix": dms,
        "is_entity": is_entity,
        "positions": positions
    }

    logger.info('Storing graphs to disk')
    with open(args.out_file, 'w') as fout:
        json.dump(graphs, fout)
    logger.info('Writing metadata to disk')
    with open(args.metadata, 'w') as fout:
        json.dump(metadata, fout)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('out_file')
    parser.add_argument('metadata')
    parser.add_argument('spm_model')
    parser.add_argument('--add-inverse', action='store_true')
    parser.add_argument('--bpe-dropout', type=float, default=None)
    parser.add_argument('--sample-factor', type=int, default=1)
    parser.add_argument('--title-graph', action='store_true')

    args = parser.parse_args()
    main(args)
