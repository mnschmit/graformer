from typing import Iterable, Iterator, List


def batch_gen(iterable: Iterable[int], batch_size: int, drop_last: bool) -> Iterator[List[int]]:
    batch = []
    for idx in iterable:
        batch.append(idx)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0 and not drop_last:
        yield batch
