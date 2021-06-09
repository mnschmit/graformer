from typing import Optional, Callable, Any, Tuple
import random
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes
import torch.distributed as dist
import math
from collections import defaultdict
from itertools import islice, chain


class CurriculumBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int,
                 start_competence: float = 0.1, full_competence_time: int = 50000,
                 att_fun: str = 'len', extract_fun: Optional[Callable[[Any], str]] = None,
                 num_pretraining_steps: int = 0, distributed: bool = False):
        super().__init__(dataset)
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        implemented_att_funs = {
            'len': self.diff_len
        }

        if att_fun not in implemented_att_funs:
            raise ValueError("att_fun should be one of {}, but got att_fun={}".format(
                str(implemented_att_funs.keys()), att_fun))

        if distributed:
            if not dist.is_available():
                raise ValueError(
                    "When distributed=True, distributed package is required")
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            self.num_samples = int(
                math.ceil(len(dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
            self.T = max(1, int(
                math.ceil(full_competence_time * 1.0 / self.num_replicas)))
        else:
            self.num_samples = len(dataset)
            self.T = max(1, full_competence_time)

        if not isinstance(full_competence_time, _int_classes) or full_competence_time < 0:
            raise ValueError("full_competence_time should be a nonnegative integer value, "
                             "but got full_competence_time={}".format(full_competence_time))
        if not isinstance(start_competence, float) or start_competence <= 0.0 or\
           start_competence > 1.0:
            raise ValueError("start_competence should be a float value x with 0.0 < x <= 1.0, "
                             "but got start_competence={}".format(start_competence))
        if not isinstance(num_pretraining_steps, _int_classes) or num_pretraining_steps < 0:
            raise ValueError("num_pretraining_steps should be a nonnegative integer value, "
                             "but got num_pretraining_steps={}".format(num_pretraining_steps))

        self.dataset = dataset
        self.batch_size = batch_size
        self.att_fun = implemented_att_funs[att_fun]
        self.distributed = distributed
        self.num_pretraining_steps = num_pretraining_steps

        self.extract_fun = extract_fun
        self.c0_square = start_competence ** 2
        self.competence = start_competence

        self.compute_difficulties()
        self.available_samples = []
        self.num_batches_given = 0
        self.num_batches_per_epoch = (
            self.num_samples + self.batch_size - 1) // self.batch_size

    def is_fully_competent(self):
        return (self.num_batches_given < self.num_pretraining_steps)\
            or (self.num_batches_given >= self.T or math.isclose(1.0, self.competence))

    def iter_fully(self):
        batch = []
        for idx in torch.randperm(self.num_samples).tolist():
            batch.append(idx)
            if len(batch) == self.batch_size:
                self.num_batches_given += 1
                yield batch
                batch = []
        if len(batch) > 0:
            self.num_batches_given += 1
            yield batch

    def iter_curricully(self):
        bucket_idx, next_diff_bucket = self.compute_available_samples()
        self.num_batches_given += 1
        while self.num_batches_given % self.num_batches_per_epoch != 0:
            try:
                batch = random.sample(self.available_samples, self.batch_size)
            except ValueError:
                batch = random.choices(
                    self.available_samples, k=self.batch_size)
            yield batch

            self.num_batches_given += 1
            self.evaluate_competence()
            if self.diff_cdf[next_diff_bucket] < self.competence\
               or math.isclose(self.diff_cdf[next_diff_bucket], self.competence):
                self.available_samples.extend(
                    self.diff2textids[next_diff_bucket])
                if not self.is_fully_competent():
                    bucket_idx += 1
                    next_diff_bucket = self.difficulties[bucket_idx]
        yield random.sample(self.available_samples, self.batch_size)

    def __iter__(self):
        if self.is_fully_competent():
            yield from self.iter_fully()
        else:
            yield from self.iter_curricully()

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def diff_len(self, text: str) -> int:
        return len(text)

    def compute_difficulties(self):
        if self.distributed:
            # so every process repeats the same samples
            rand = random.Random(0)
            dataset = islice(
                chain(
                    range(len(self.dataset)),
                    rand.sample(
                        range(len(self.dataset)),
                        self.total_size - len(self.dataset)
                    )
                ),
                self.rank, self.total_size, self.num_replicas
            )
        else:
            dataset = range(len(self.dataset))

        data = [
            self.dataset[i] if self.extract_fun is None
            else self.extract_fun(self.dataset[i])
            for i in dataset
        ]

        diff_data = defaultdict(list)
        for idx, d in enumerate(data):
            diff_data[self.att_fun(d)].append(idx)
        diff_prob = {d: len(diff_data[d])/len(data) for d in diff_data}
        diff_cdf = {}
        prev_d = None
        for d in sorted(diff_prob.keys()):
            sum_before = diff_cdf.get(prev_d, 0)
            prev_d = d
            diff_cdf[d] = sum_before + diff_prob[d]

        self.diff2textids = diff_data
        self.diff_cdf = diff_cdf
        self.difficulties = sorted(self.diff_cdf.keys())

    def compute_available_samples(self) -> Tuple[int, Optional[int]]:
        self.available_samples.clear()
        for d_idx, d in enumerate(self.difficulties):
            if self.diff_cdf[d] < self.competence\
               or math.isclose(self.diff_cdf[d], self.competence):
                self.available_samples.extend(self.diff2textids[d])
            else:
                return d_idx, d
        return -1, None

    def evaluate_competence(self):
        if self.num_batches_given < self.num_pretraining_steps:
            self.competence = 1.0
        else:
            self.competence = min(
                1.,
                math.sqrt(
                    self.num_batches_given * (1 - self.c0_square) / self.T + self.c0_square)
            )
