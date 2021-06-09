from test_tube import HyperOptArgumentParser
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import torch
# import numpy as np
# import random

from ..models.encdec_tokennodes import Graph2Text


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_checkpoint_dir(experiment_name: str):
    version_num = 0
    current_path = os.path.join(
        'checkpoints', experiment_name, 'version_{}'.format(version_num))
    while os.path.exists(current_path):
        version_num += 1
        current_path = os.path.join(
            'checkpoints', experiment_name, 'version_{}'.format(version_num))
    return current_path


def main(args):

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = Graph2Text(vars(args))

    num_params = count_parameters(model)
    if num_params > 67e6:
        logging.getLogger(__name__).warning('Very high number of parameters')
        exit(1)

    gpus = [int(c) for c in args.gpus]
    distributed = len(gpus) > 1

    logger = TestTubeLogger("tt_logs", name=args.experiment_name)

    checkpoint_dir = find_checkpoint_dir(args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir,
        monitor='val_loss' if args.light_validation else 'bleu',
        mode='min' if args.light_validation else 'max'
    )

    trainer = Trainer(
        deterministic=True,
        logger=logger,
        gpus=gpus,
        accelerator='ddp' if distributed else None,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accum_count,
        gradient_clip_val=args.grad_clip,
        limit_val_batches=args.val_percent_check,
        limit_train_batches=args.train_percent_check,
        # min_epochs=10,
        max_epochs=args.num_epochs,
        # num_sanity_val_steps=5 if args.VG else 2,
        progress_bar_refresh_rate=20
        # track_grad_norm=2
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=47110815)
    parser.add_argument('--gpus', default='')

    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--val-percent-check', default=1.0, type=float)
    parser.add_argument('--train-percent-check', default=1.0, type=float)
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
    parser.add_argument('--accum-count', type=int, default=2)
    # recommendation by witwicky is 1.0
    parser.add_argument('--grad-clip', type=float, default=0)

    parser.add_argument('--experiment-name', default='default')
    parser.add_argument('--light-validation',
                        action='store_true', dest='light_validation')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-validation', action='store_true')

    parser = Graph2Text.add_model_specific_args(parser)

    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args)
