from typing import List, Iterable
from torch.utils.data import Dataset
import json
import random


class WebNLG(Dataset):
    def __init__(self, graph_file, text_file, for_testing=False,
                 word_dropout=0.0, unk_id=3, special_ids=[0, 1, 2]):
        super().__init__()

        with open(graph_file) as f:
            graphs = json.load(f)

        self.dm = graphs['distance_matrix']
        self.is_entity = graphs['is_entity']
        self.pos = graphs['positions']

        with open(text_file) as f:
            textdata = json.load(f)

        self.node_labels = textdata['node_label']
        self.texts = textdata['target']
        self.for_testing = for_testing

        if not for_testing:
            self.idx2idx = []
            for i, t in enumerate(self.texts):
                for j in range(len(t)):
                    self.idx2idx.append((i, j))

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
            internal_idx = idx
            target = self.texts[idx]
        else:
            internal_idx, target_idx = self.idx2idx[idx]
            target = self.texts[internal_idx][target_idx]

        node_labels = self.node_labels[internal_idx]

        if not self.for_testing and self.drop_p > 0.0:
            target = self._drop_words(target)
            node_labels = self._drop_words(node_labels)

        sample = (self.dm[internal_idx], self.is_entity[internal_idx], self.pos[internal_idx],
                  node_labels, target)
        return sample

    def __len__(self) -> int:
        if self.for_testing:
            return len(self.texts)
        else:
            return len(self.idx2idx)

    def extract_fun(self, sample) -> str:
        return sample[-1]

    def num_tokens_fun(self, sample) -> int:
        node_labels = sample[-2]
        target = sample[-1]
        return len(node_labels) + len(target)
