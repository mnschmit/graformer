import argparse
import json
import subprocess
from tqdm import tqdm
from .prepare_webnlg import preprocess_entity, preprocess_relation


def create_tokenizer(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(
        ['perl', cmd, '-q', '-l', 'en', '-no-escape'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        encoding='utf8'
    )


def tokenize(cmd: str, to_be_tokenized: str) -> str:
    tokenizer = create_tokenizer(cmd)
    tokenizer.stdin.write(to_be_tokenized + '\n')
    tokenizer.stdin.close()
    tokenized = tokenizer.stdout.readline().rstrip()
    tokenizer.terminate()
    return tokenized


def main(args):
    with open(args.json_in) as f:
        document = json.load(f)

    for entry in tqdm(document['entries']):
        for inst in entry.values():
            for lex in inst['lexicalisations']:
                lex['lex'] = tokenize(
                    args.path_to_tokenizer_script, lex['lex'])
            for triple in inst['modifiedtripleset']:
                triple['subject'] = tokenize(
                    args.path_to_tokenizer_script,
                    preprocess_entity(
                        triple['subject'], False, lc=args.lowercase)
                )
                triple['object'] = tokenize(
                    args.path_to_tokenizer_script,
                    preprocess_entity(
                        triple['object'], False, lc=args.lowercase)
                )
                triple['property'] = preprocess_relation(
                    triple['property'], False, lc=args.lowercase
                )

    with open(args.json_out, 'w') as f:
        json.dump(document, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_in')
    parser.add_argument('json_out')
    parser.add_argument('path_to_tokenizer_script')
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()
    main(args)
