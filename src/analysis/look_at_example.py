import argparse
import sentencepiece as spm
import json
import random


def show_example(nodes, targets, sp):
    decoded_nodes = [sp.DecodeIds([n]) for n in nodes]
    if isinstance(targets[0], list):
        decoded_targets = [sp.EncodeAsPieces(
            sp.DecodeIds(target)) for target in targets]
    else:
        decoded_targets = [sp.EncodeAsPieces(sp.DecodeIds(targets))]

    print('Input:')
    print(*nodes, sep=' - ')
    print(*decoded_nodes, sep=' - ')
    print('Possible outputs:')
    for t in decoded_targets:
        print(t)
        print('---')


def main(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)

    with open(args.text_json) as f:
        data = json.load(f)

    if args.sample_num < 0:
        sample_num = random.randint(0, len(data['node_label']))
    else:
        sample_num = args.sample_num

    nl, targets = data['node_label'][sample_num], data[args.text_key][sample_num]

    print('Sample #', sample_num)
    show_example(nl, targets, sp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spm_model')
    parser.add_argument('text_json')
    parser.add_argument('--sample-num', type=int, default=-1)
    parser.add_argument('--text-key', default='target')
    args = parser.parse_args()
    main(args)
