import torch
from typing import Callable, Optional, Any
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import random


class BucketSampler(Sampler):
    def __init__(self, dataset: Dataset, shuffle1: bool = True, shuffle2: bool = True,
                 ratio: float = 0.8, key_fun: Optional[Callable[[Any], int]] = len,
                 distributed: bool = False):
        super().__init__(dataset)
        self.shuffle = [shuffle1, shuffle2]
        self.dataset = dataset
        self.epoch = 0
        self.distributed = distributed
        self.precomputed_keys = [key_fun(x) for x in dataset]
        self.ratio = ratio

        if distributed:
            if not dist.is_available():
                raise ValueError(
                    "When distributed=True, distributed package is required")
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            self.num_samples = int(
                math.ceil(len(dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = len(dataset)

        self.buckets = self.sort_in_two_buckets(
            list(range(len(dataset))), ratio)

    def sort_in_two_buckets(self, indices, ratio):
        indices.sort(key=lambda i: self.precomputed_keys[i])
        thr = int(len(indices) * ratio)
        return indices[:thr], indices[thr:]

    def __iter__(self):
        if self.distributed:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            # subsample
            indices = indices[self.rank:self.total_size():self.num_replicas]
            buckets = self.sort_in_two_buckets(indices, self.ratio)
        else:
            buckets = self.buckets

        for bucket, should_shuffle in zip(buckets, self.shuffle):
            if should_shuffle:
                random.shuffle(bucket)
            yield from bucket

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
