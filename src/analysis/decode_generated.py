import argparse
import logging
import sentencepiece as spm


def main(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)

    with open(args.hypo_file, 'w') as fh, open(args.ref_file, 'w') as fr:
        for input_file in args.input_file:
            with open(input_file) as f:
                for line in f:
                    gen, ref = line.split('\t')

                    print(sp.DecodeIds(
                        [int(i) for i in gen.split(' ')]), file=fh
                    )
                    print(sp.DecodeIds(
                        [int(i) for i in ref.split(' ')]), file=fr
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spm_model')
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    parser.add_argument('input_file', nargs='+')
    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args)
