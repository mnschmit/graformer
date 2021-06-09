from typing import Optional, Callable, Any, Tuple
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes
import torch.distributed as dist
import math
from collections import defaultdict
from itertools import islice, chain
from .token_batch_sampler import TokenBatchSampler


class CurriculumTokenBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, token_per_batch: int,
                 num_tokens_fun: Callable[[Any], int],
                 start_competence: float = 0.1, full_competence_time: int = 100000,
                 distributed: bool = False, batches_per_epoch: int = 10000):
        super().__init__(dataset)
        if not isinstance(token_per_batch, _int_classes) or isinstance(token_per_batch, bool) or \
                token_per_batch <= 0:
            raise ValueError("token_per_batch should be a positive integer value, "
                             "but got token_per_batch={}".format(token_per_batch))
        if not isinstance(batches_per_epoch, _int_classes)\
                or isinstance(batches_per_epoch, bool) or batches_per_epoch < 0:
            raise ValueError("batches_per_epoch should be a positive integer value, "
                             "but got batches_per_epoch={}".format(batches_per_epoch))
        if not isinstance(distributed, bool):
            raise ValueError("distributed should be a boolean value, but got "
                             "distributed={}".format(distributed))
        if not isinstance(full_competence_time, _int_classes) or full_competence_time < 0:
            raise ValueError("full_competence_time should be a nonnegative integer value, "
                             "but got full_competence_time={}".format(full_competence_time))
        if not isinstance(start_competence, float) or start_competence <= 0.0 or\
           start_competence > 1.0:
            raise ValueError("start_competence should be a float value x with 0.0 < x <= 1.0, "
                             "but got start_competence={}".format(start_competence))

        num_total_tokens = sum(num_tokens_fun(x) for x in dataset)

        if distributed:
            if not dist.is_available():
                raise ValueError(
                    "When distributed=True, distributed package is required")
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            num_total_tokens = int(
                math.ceil(num_total_tokens * 1.0 / self.num_replicas))
            num_samples = int(
                math.ceil(len(dataset) * 1.0 / self.num_replicas))
            self.total_size = num_samples * self.num_replicas
            batches_per_epoch = int(
                math.ceil(batches_per_epoch * 1.0 / self.num_replicas))
        else:
            num_samples = len(dataset)

        self.T = max(1, full_competence_time)

        self.dataset = dataset
        self.token_per_batch = token_per_batch
        self.get_num_tokens = num_tokens_fun
        self.distributed = distributed

        self.c0_square = start_competence ** 2
        self.competence = start_competence

        self.num_batches_per_epoch = max(
            batches_per_epoch,
            num_total_tokens // self.token_per_batch
        )
        # avg sample-based batch size
        self.avg_num_samples_per_batch = (
            num_samples * token_per_batch) // num_total_tokens

        self.full_batch_sampler = TokenBatchSampler(
            dataset, token_per_batch, num_tokens_fun,
            shuffle=True, drop_last=False, distributed=distributed,
            batches_per_epoch=self.num_batches_per_epoch
        )

        self.compute_difficulties()
        self.available_samples = []
        self.num_batches_given = 0

    def is_fully_competent(self):
        return self.num_batches_given >= self.T or math.isclose(1.0, self.competence)

    def iter_fully(self):
        for batch in self.full_batch_sampler:
            self.num_batches_given += 1
            yield batch

    def compose_legal_batch(self):
        try:
            batch = random.sample(
                self.available_samples, self.avg_num_samples_per_batch)
        except ValueError:
            batch = random.choices(
                self.available_samples, k=self.avg_num_samples_per_batch)

        num_tokens_in_batch = sum(
            self.get_num_tokens(self.dataset[x]) for x in batch)

        if num_tokens_in_batch > self.token_per_batch:
            while num_tokens_in_batch > self.token_per_batch:
                removed = batch.pop()
                num_tokens_in_batch -= self.get_num_tokens(
                    self.dataset[removed])
        else:
            new_sample = random.choice(self.available_samples)
            num_new_tokens = self.get_num_tokens(self.dataset[new_sample])
            while num_tokens_in_batch + num_new_tokens < self.token_per_batch:
                batch.append(new_sample)
                num_tokens_in_batch += num_new_tokens
                new_sample = random.choice(self.available_samples)
                num_new_tokens = self.get_num_tokens(self.dataset[new_sample])

        return batch

    def iter_curricully(self):
        bucket_idx, next_diff_bucket = self.compute_available_samples()
        self.num_batches_given += 1
        while self.num_batches_given % self.num_batches_per_epoch != 0:
            batch = self.compose_legal_batch()
            yield batch

            self.evaluate_competence()
            if self.diff_cdf[next_diff_bucket] < self.competence\
               or math.isclose(self.diff_cdf[next_diff_bucket], self.competence):
                self.available_samples.extend(
                    self.diff2textids[next_diff_bucket])
                if not self.is_fully_competent():
                    bucket_idx += 1
                    next_diff_bucket = self.difficulties[bucket_idx]

            self.num_batches_given += 1
        yield self.compose_legal_batch()

    def __iter__(self):
        if self.is_fully_competent():
            yield from self.iter_fully()
        else:
            yield from self.iter_curricully()

    def __len__(self) -> int:
        return self.num_batches_per_epoch

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

        data = [(i, self.dataset[i]) for i in dataset]
        diff_data = defaultdict(list)

        for idx, d in data:
            diff_data[self.get_num_tokens(d)].append(idx)
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
            if self.diff_cdf[d] > self.competence:
                return d_idx, d
            else:
                self.available_samples.extend(self.diff2textids[d])
        return -1, None

    def evaluate_competence(self):
        self.competence = min(
            1.,
            math.sqrt(
                self.num_batches_given * (1 - self.c0_square) / self.T + self.c0_square)
        )
