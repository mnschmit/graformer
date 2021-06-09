import argparse
from tqdm import tqdm
import json
from .prepare_webnlg import preprocess_text


def main(args):
    with open(args.json_file) as f:
        nlg = json.load(f)

    with open(args.ref_file, 'w') as fout:
        for entry in tqdm(nlg['entries']):
            for e in entry.values():
                targets = [preprocess_text(lex['lex'])
                           for lex in e["lexicalisations"]]
                print(*targets, sep='*#', file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('ref_file')
    args = parser.parse_args()
    main(args)
