from argparse import ArgumentParser
from ..models.encdec_tokennodes import Graph2Text
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import colors
import logging
import numpy as np

plt.switch_backend('agg')


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    if args.from_checkpoint is None:
        model = Graph2Text(vars(args))
    else:
        model = Graph2Text.load_from_checkpoint(args.from_checkpoint)

    num_params = count_parameters(model)
    print("Total number of parameters:", num_params)

    num_special_graph_embs = 4
    num_graph_embs = model.hparams.max_graph_range * 2 + num_special_graph_embs
    num_heads = args.cut_heads if args.cut_heads else model.hparams.num_heads
    matrix = model.graph_pos_embed.get_weights(
    )[:num_graph_embs, :num_heads].detach()

    special_graph_embs = matrix[0:num_special_graph_embs].numpy()
    pos_graph_embs = matrix[num_special_graph_embs::2].numpy()
    neg_graph_embs = matrix[num_special_graph_embs+1::2].numpy()

    new_matrix = np.concatenate(
        (
            np.flip(neg_graph_embs, axis=0),
            special_graph_embs[-2:-1],
            pos_graph_embs,
            special_graph_embs[:-2],
            special_graph_embs[-1:]
        ),
        axis=0
    )

    if args.transpose:
        new_matrix = new_matrix.transpose()

    if args.normalize:
        min_weight = np.min(new_matrix)
        max_weight = np.max(new_matrix)
        normalized_matrix = (new_matrix - min_weight) / \
            (max_weight - min_weight)
        new_matrix = normalized_matrix

    pos_labels = [
        str(i+1)
        for i in range((num_graph_embs - num_special_graph_embs) // 2)
    ]
    neg_labels = ['-' + l for l in pos_labels]

    if args.transpose:
        title_labels = [r'$T^-$', r'$T$']
    else:
        title_labels = ['title2txt', 'txt2title']

    gpos_labels = [''] + list(reversed(neg_labels)) + ['0'] + \
        pos_labels + title_labels + [r'$\infty$']

    norm = colors.TwoSlopeNorm(0.0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(new_matrix, interpolation='nearest',
                     cmap=plt.cm.seismic, norm=norm)  # Greys
    fig.colorbar(cax)

    if args.transpose:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_ylabel("attention head")
        ax.set_xlabel(r"$R_{ij}$")
        ax.set_xticklabels(gpos_labels)
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlabel("attention head")
        ax.set_ylabel(r"$R_{ij}$")
        ax.set_yticklabels(gpos_labels)

    plt.savefig('interpret.pdf')


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', default='')
    parser.add_argument('--from-checkpoint', default=None)
    parser.add_argument('--cut-heads', type=int, default=None)
    parser.add_argument('--lines', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--transpose', action='store_true')
    parser = Graph2Text.add_model_specific_args(parser)

    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args)
