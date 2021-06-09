import argparse
from nlgeval import NLGEval
from tqdm import tqdm


def main(args):
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

    samples = {}
    with open(args.gen_file) as f:
        for line in tqdm(f):
            hypo, refs = line.rstrip().split('\t')
            metrics_dict = nlgeval.compute_individual_metrics(
                refs.split('*#'), hypo)
            samples[(hypo, refs)] = metrics_dict['Bleu_4']

    for hypo, refs in sorted(samples.keys(), key=samples.__getitem__)[:args.num_samples]:
        print('BLEU:', samples[(hypo, refs)])
        print('H:', hypo)
        for r in refs.split('*#'):
            print('R:', r)
        print('---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_file')
    parser.add_argument('--num-samples', '-n', type=int, default=5)
    args = parser.parse_args()
    main(args)
