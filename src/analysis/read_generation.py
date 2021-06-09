import argparse
import random


def pretty_print_hypo_ref(hypo: str, ref: str):
    print('=== GENERATED ===')
    print(hypo)
    print()
    print('=== REFERENCE ===')
    print(ref)


def main(args):
    samples = []
    with open(args.hypo_file) as hf, open(args.ref_file) as rf:
        for hline_rline in zip(hf, rf):
            samples.append(hline_rline)

    s = random.choice(samples)
    pretty_print_hypo_ref(*s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    args = parser.parse_args()

    main(args)
