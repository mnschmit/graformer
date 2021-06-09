from typing import Callable, Any
from torch.utils.data import Dataset, DistributedSampler, RandomSampler,\
    SequentialSampler, BatchSampler
from torch.utils.data.sampler import Sampler, _int_classes
from tqdm import tqdm


class TokenBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, token_per_batch: int,
                 num_tokens_fun: Callable[[Any], int],
                 shuffle: bool = True, drop_last: bool = False,
                 distributed: bool = False, batches_per_epoch: int = 0):
        super().__init__(dataset)
        if not isinstance(token_per_batch, _int_classes) or isinstance(token_per_batch, bool) or \
                token_per_batch <= 0:
            raise ValueError("token_per_batch should be a positive integer value, "
                             "but got token_per_batch={}".format(token_per_batch))
        if not isinstance(batches_per_epoch, _int_classes)\
           or isinstance(batches_per_epoch, bool) or batches_per_epoch < 0:
            raise ValueError("batches_per_epoch should be a positive integer value, "
                             "but got batches_per_epoch={}".format(batches_per_epoch))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle should be a boolean value, but got "
                             "shuffle={}".format(shuffle))
        if not isinstance(distributed, bool):
            raise ValueError("distributed should be a boolean value, but got "
                             "distributed={}".format(distributed))
        self.dataset = dataset
        self.get_num_tokens = num_tokens_fun
        self.num_total_tokens = self.count_total_tokens()
        self.token_per_batch = token_per_batch
        self.drop_last = drop_last
        if distributed:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
        self.bucket_sampler = BatchSampler(
            self.sampler,
            min(
                int(100 * len(self.dataset) * token_per_batch /
                    self.num_total_tokens),
                len(self.sampler)
            ),
            False
        )
        self.batches_per_epoch = max(
            batches_per_epoch,
            self.num_total_tokens // self.token_per_batch +
            (0 if drop_last else len(self.bucket_sampler))
        )

    def count_total_tokens(self):
        return sum(self.get_num_tokens(x) for x in tqdm(self.dataset))

    def __iter__(self):
        num_batches_given = 0
        while num_batches_given < self.batches_per_epoch:
            for bucket in self.bucket_sampler:
                num_tokens_in_batch = 0
                batch = []
                for idx in sorted(bucket, key=lambda i: self.get_num_tokens(self.dataset[i])):
                    num_tokens_for_idx = self.get_num_tokens(self.dataset[idx])
                    if num_tokens_in_batch + num_tokens_for_idx > self.token_per_batch:
                        num_batches_given += 1
                        yield batch
                        batch = [idx]
                        num_tokens_in_batch = num_tokens_for_idx
                    else:
                        batch.append(idx)
                        num_tokens_in_batch += num_tokens_for_idx
                if len(batch) > 0 and not self.drop_last:
                    num_batches_given += 1
                    yield batch

    def __len__(self):
        return self.batches_per_epoch

        # if self.batches_per_epoch > 0:
        #     return self.batches_per_epoch
        # elif self.drop_last:
        #     return self.num_total_tokens // self.token_per_batch
        # else:
        #     return self.num_total_tokens // self.token_per_batch + len(self.bucket_sampler)
