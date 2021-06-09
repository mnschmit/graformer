import argparse
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


def main(args: argparse.Namespace):
    with open(args.train_file) as f:
        agenda = json.load(f)

    with open(args.out_file, 'w') as fout:
        for article in tqdm(agenda):
            print(article['title'].lower(), file=fout)
            for sentence in sent_tokenize(article['abstract_og']):
                print(sentence, file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    main(args)
