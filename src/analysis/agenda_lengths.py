import argparse
import json
import numpy as np
import nltk
from tqdm import tqdm


def main(args: argparse.Namespace):
    with open(args.agenda_json) as f:
        agenda = json.load(f)

    lengths = []
    for article in tqdm(agenda):
        lengths.append(len(nltk.word_tokenize(article['abstract'])))

    print('statistics')
    print('min {} | max {} | mean {:.4f} | std {:.4f}'.format(
        np.min(lengths), np.max(lengths), np.mean(lengths), np.std(lengths)
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agenda_json')

    args = parser.parse_args()
    main(args)
