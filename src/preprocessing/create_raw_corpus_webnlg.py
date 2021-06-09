import argparse
import json
from tqdm import tqdm
import random
from .prepare_webnlg import preprocess_entity, preprocess_relation, preprocess_text


def treat_lexicalisation(lex, fout, preprocessing):
    text = lex['lex']
    if preprocessing:
        text = preprocess_text(text)
    print(text, file=fout)


def main(args: argparse.Namespace):
    with open(args.train_file) as f:
        nlg = json.load(f)

    rand = random.Random(args.seed)
    with open(args.out_file, 'w') as fout:
        for entry in tqdm(nlg['entries']):
            for e in entry.values():
                if rand.random() > args.subsample:
                    continue

                for triple in e["modifiedtripleset"]:
                    print(
                        preprocess_entity(
                            triple['subject'], prep=args.preprocessing),
                        preprocess_relation(
                            triple['property'], prep=args.preprocessing),
                        preprocess_entity(
                            triple['object'], prep=args.preprocessing),
                        file=fout
                    )

                if args.all_references:
                    for lex in e["lexicalisations"]:
                        treat_lexicalisation(lex, fout, args.preprocessing)
                else:
                    lex = rand.choice(e["lexicalisations"])
                    treat_lexicalisation(lex, fout, args.preprocessing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('out_file')
    parser.add_argument('--no-preprocessing',
                        action='store_false', dest='preprocessing')
    parser.add_argument('--all-references', action='store_true')
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=47110815)
    args = parser.parse_args()

    main(args)
