import argparse
from ..data_loader.webnlg_dataset import WebNLG
from ..data_loader.agenda_dataset import Agenda
from ..data_loader.token_batch_sampler import TokenBatchSampler


def main(args):
    if args.webnlg:
        dataset = WebNLG(args.json_graphs, args.json_texts)
    else:
        dataset = Agenda(args.json_graphs, args.json_texts,
                         title_graph=args.title_graph)

    sampler = TokenBatchSampler(
        dataset, args.batch_size, dataset.num_tokens_fun)
    num_epochs = args.num_steps / len(sampler)
    num_steps = len(sampler) * args.num_epochs
    avg_num_tokens_per_sample = sampler.num_total_tokens / len(dataset)
    print(
        (
            'With a batch size of {} tokens,' +
            ' you will have {} steps in {} epochs and you will need {} epochs to achieve {} steps.'
        ).format(
            args.batch_size, num_steps, args.num_epochs, num_epochs, args.num_steps
        )
    )
    print('In average, one sample contains {} tokens.'.format(
        avg_num_tokens_per_sample))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_graphs')
    parser.add_argument('json_texts')
    parser.add_argument('--num-steps', type=int, default=300000)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--webnlg', action='store_true')
    parser.add_argument('--title-graph', action='store_true')
    args = parser.parse_args()
    main(args)
