from argparse import ArgumentParser
import random


def main(args):
    rand = random.Random(args.seed)
    with open(args.input_file) as f, open(args.output_file, 'w') as fout:
        for line in f:
            if rand.random() < args.percentage:
                fout.write(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--percentage', '-p', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=47110815)
    args = parser.parse_args()
    main(args)
