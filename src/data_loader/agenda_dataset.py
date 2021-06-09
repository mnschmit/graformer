from typing import List, Iterable
from torch.utils.data import Dataset
import json
import random


class Agenda(Dataset):
    def __init__(self, graph_file, metadata_file, title_graph=False, for_testing=False,
                 word_dropout=0.0, unk_id=3, special_ids=[0, 1, 2]):
        super().__init__()

        with open(graph_file) as f:
            graphs = json.load(f)

        self.dm = graphs['distance_matrix']
        self.is_entity = graphs['is_entity']
        self.pos = graphs['positions']

        with open(metadata_file) as f:
            metadata = json.load(f)

        self.node_labels = metadata['node_label']
        self.texts = metadata['abstract']
        self.title_graph = title_graph
        if not title_graph:
            self.titles = metadata['title']
        self.for_testing = for_testing
        self.drop_p = word_dropout
        self.unk_id = unk_id
        self.special_ids = set(special_ids)

    def _drop_words(self, ids: Iterable[int]) -> List[int]:
        res = []
        for i in ids:
            if i in self.special_ids or random.random() > self.drop_p:
                res.append(i)
            else:
                res.append(self.unk_id)
        return res

    def __getitem__(self, idx: int):
        if self.for_testing:
            target = [self.texts[idx]]
        else:
            target = self.texts[idx]

        node_labels = self.node_labels[idx]

        if not self.for_testing and self.drop_p > 0.0:
            target = self._drop_words(target)
            node_labels = self._drop_words(node_labels)

        if self.title_graph:
            sample = (self.dm[idx], self.is_entity[idx], self.pos[idx],
                      node_labels, target)
        else:
            sample = (self.dm[idx], self.is_entity[idx], self.pos[idx],
                      node_labels, self.titles[idx], target)
        return sample

    def __len__(self) -> int:
        return len(self.texts)

    def extract_fun(self, sample) -> str:
        return sample[-1]

    def num_tokens_fun(self, sample) -> int:
        node_labels = sample[3]
        target = sample[-1]
        res = len(node_labels) + len(target)

        if not self.title_graph:
            res += len(sample[-2])

        return res
