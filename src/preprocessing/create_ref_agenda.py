import argparse
from tqdm import tqdm
import json


def main(args):
    with open(args.json_file) as f:
        agenda = json.load(f)

    with open(args.ref_file, 'w') as fout:
        for article in tqdm(agenda):
            print(article['abstract_og'].lower(), file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('ref_file')
    args = parser.parse_args()
    main(args)
