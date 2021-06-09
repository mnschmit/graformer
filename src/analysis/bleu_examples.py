import argparse
from nlgeval import NLGEval
from tqdm import tqdm
import json
from .count_cc_size import generate_triples_webnlg, generate_triples_agenda, generate_graph
import networkx as nx
import nltk
from collections import Counter


def compute_repetition_rate(s: str, max_n=4):
    res = []
    for n in range(1, max_n+1):
        ngrams = nltk.ngrams(nltk.word_tokenize(s), n)
        counts = Counter(ngrams)

        try:
            res.append(
                float(sum([1 for v in counts.values() if v > 1])) / len(counts))
        except ZeroDivisionError:
            res.append(0.0)

    return tuple(res)


def load_texts(gen_file: str, is_ref: bool):
    samples = []
    with open(gen_file) as f:
        for line in f:
            line = line.rstrip()
            texts = line.split('*#')
            if is_ref:
                samples.append(texts)
            else:
                samples.append(texts[0])
    return samples


def main(args):
    good_hypos = load_texts(args.good_gen, False)
    bad_hypos = load_texts(args.bad_gen, False)
    refs = load_texts(args.ref_file, True)

    with open(args.dataset_file) as f:
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

    nlgeval = NLGEval(
        metrics_to_omit=[
            'METEOR',
            'ROUGE_L',
            'CIDEr'
        ], no_skipthoughts=True, no_glove=True
    )

    interesting_samples = []
    for i, (gh, bh, ref_list, tripset) in tqdm(list(
            enumerate(zip(good_hypos, bad_hypos, refs, triples)))):
        metrics_dict = nlgeval.compute_individual_metrics(ref_list, gh)
        good_bleu = metrics_dict['Bleu_4']
        metrics_dict = nlgeval.compute_individual_metrics(ref_list, bh)
        bad_bleu = metrics_dict['Bleu_4']

        graph = generate_graph(tripset)
        num_ccs = nx.number_connected_components(graph)
        num_nodes = len(graph)
        mean_cc_size = num_nodes / num_ccs

        diameters = []
        for subgraph in [graph.subgraph(cc).copy() for cc in nx.connected_components(graph)]:
            diameters.append(nx.diameter(subgraph))
        max_diameter = max(diameters)

        good_rep_rate = compute_repetition_rate(gh)[-1]
        bad_rep_rate = compute_repetition_rate(bh)[-1]

        if (good_bleu > bad_bleu) and\
           (args.min_mean_cc_size <= mean_cc_size < args.max_mean_cc_size) and\
           (args.min_max_diameter <= max_diameter < args.max_max_diameter) and\
           (good_rep_rate < bad_rep_rate):
            interesting_samples.append(
                (i, good_bleu, bad_bleu, good_rep_rate, bad_rep_rate,
                 mean_cc_size, max_diameter))

    for i, gb, bb, grep, brep, mean_cc, max_dia in sorted(
            interesting_samples, key=lambda x: x[3])[:args.num_samples]:
        print('Sample #', i)
        print('Good BLEU:', gb, 'Bad BLEU:', bb, 'Difference:', gb-bb)
        print('Good RepRate:', grep, 'Bad RepRate:', brep)
        print('Mean CC size:', mean_cc, 'Max diameter:', max_dia)
        print('Good Hypo:', good_hypos[i])
        print('Bad Hypo:', bad_hypos[i])
        for r in refs[i]:
            print('R:', r)
        print('---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    parser.add_argument('good_gen')
    parser.add_argument('bad_gen')
    parser.add_argument('ref_file')
    parser.add_argument('--webnlg', action='store_true')
    parser.add_argument('--ignore-isolated', action='store_true')
    parser.add_argument('--num-samples', '-n', type=int, default=10)
    parser.add_argument('--min-mean-cc-size', type=float, default=0.0)
    parser.add_argument('--max-mean-cc-size', type=float, default=1000.0)
    parser.add_argument('--min-max-diameter', type=int, default=0)
    parser.add_argument('--max-max-diameter', type=int, default=1000)
    args = parser.parse_args()
    main(args)
