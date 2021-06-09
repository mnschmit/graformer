import torch
from typing import Callable, Optional, Any, Sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import random
from .utils import batch_gen


class BucketBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_sizes: Sequence[int],
                 shuffle: Optional[Sequence[bool]] = None,
                 ratios: Optional[Sequence[float]] = None,
                 key_fun: Optional[Callable[[Any], int]] = len,
                 distributed: bool = False):
        super().__init__(dataset)
        self.batch_sizes = batch_sizes
        self.distributed = distributed
        self.precomputed_keys = [key_fun(x) for x in dataset]

        if shuffle is not None:
            assert len(shuffle) == len(
                batch_sizes), "inconsistent number of buckets with shuffle"
            self.shuffle = shuffle
        else:
            self.shuffle = [True] * len(batch_sizes)

        if ratios is not None:
            assert len(ratios) == len(
                batch_sizes) - 1, "ratios has to be one shorter than batch_sizes"
            self.ratios = ratios
        else:
            self.ratios = [1. / len(batch_sizes)] * (len(batch_sizes) - 1)

        if distributed:
            if not dist.is_available():
                raise ValueError(
                    "When distributed=True, distributed package is required")
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            num_samples = int(
                math.ceil(len(dataset) * 1.0 / self.num_replicas))
            self.total_size = num_samples * self.num_replicas

        self.buckets, self.num_batches_per_epoch = self.sort_into_buckets(
            list(range(len(dataset))))

    def sort_in_two_buckets(self, indices, ratio):
        indices.sort(key=lambda i: self.precomputed_keys[i])
        thr = int(len(indices) * ratio)
        b1, b2 = indices[:thr], indices[thr:]

        num_batches_bucket1 = (
            len(b1) + self.batch_sizes[0] - 1) // self.batch_sizes[0]
        num_batches_bucket2 = (
            len(b2) + self.batch_sizes[1] - 1) // self.batch_sizes[1]

        return (b1, b2), num_batches_bucket1 + num_batches_bucket2

    def sort_into_buckets(self, indices: Sequence[int]):
        indices.sort(key=lambda i: self.precomputed_keys[i])
        thresholds = [int(len(indices) * r) for r in self.ratios]
        buckets = []
        cumsum = 0
        for thr in thresholds:
            buckets.append(indices[cumsum:cumsum+thr])
            cumsum += thr
        buckets.append(indices[cumsum:])

        assert len(buckets) == len(self.batch_sizes)
        num_batches_per_bucket = [
            (len(b) + bs - 1) // bs
            for b, bs in zip(buckets, self.batch_sizes)
        ]

        return buckets, sum(num_batches_per_bucket)

    def __iter__(self):
        for bucket, should_shuffle, bs in zip(self.buckets, self.shuffle, self.batch_sizes):
            if should_shuffle:
                random.shuffle(bucket)
            yield from batch_gen(bucket, bs, False)

    def __len__(self):
        return self.num_batches_per_epoch

    def on_epoch_start(self, epoch):
        if self.distributed:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(epoch)
            indices = torch.randperm(
                len(self.precomputed_keys), generator=g).tolist()
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            self.buckets, self.num_batches_per_epoch = self.sort_into_buckets(
                indices)
