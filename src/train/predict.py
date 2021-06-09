from argparse import ArgumentParser
from pytorch_lightning import Trainer
from ..models.encdec_tokennodes import Graph2Text
import logging
import torch


def main(args):
    gpus = [int(c) for c in args.gpus]
    distributed = len(gpus) > 1

    model = Graph2Text.load_from_checkpoint(checkpoint_path=args.checkpoint)
    # model = Graph2Text.load_from_metrics(args.checkpoint, args.meta_tags)

    if hasattr(model.hparams, 'default_save_path') and model.hparams.default_save_path is None:
        model.hparams.default_save_path = './default'
    # print(model.hparams)

    model.distributed = distributed
    model.set_test_output(args.output_file)
    model.set_test_data(args.graph_file, args.text_file)
    model.activate_meteor()
    model.set_test_ref_file(args.ref_file)
    if args.length_penalty is not None:
        model.length_penalty = args.length_penalty
    if args.coverage_penalty is not None:
        model.coverage_penalty = args.coverage_penalty
    if args.top_p is not None:
        model.top_p = args.top_p
    if args.batch_size is not None:
        model.set_inference_batch_size(args.batch_size)
    if args.beam_size is not None:
        model.num_beams = args.beam_size
    if args.min_length is not None:
        model.min_length = args.min_length

    trainer = Trainer(
        gpus=gpus,
        distributed_backend='ddp' if distributed else None
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.test(model)
    # print(model.test_scores)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default='', type=str)
    parser.add_argument('checkpoint')
    # parser.add_argument('meta_tags')
    parser.add_argument('graph_file')
    parser.add_argument('text_file')
    parser.add_argument('ref_file')
    parser.add_argument('--output-file', '-o', default=None)
    parser.add_argument('--length-penalty', type=float, default=None)
    parser.add_argument('--coverage-penalty', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--beam-size', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=None)

    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args)
