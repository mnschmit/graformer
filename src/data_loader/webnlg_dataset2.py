from typing import FrozenSet, Tuple, List
from torch.utils.data import Dataset
import json
import sentencepiece as spm
from collections import OrderedDict
from ..preprocessing.prepare_webnlg import preprocess_entity, preprocess_relation,\
    preprocess_text, graph_gen_helper
from ..preprocessing.train_spm_webnlg import ENT_TOKEN, REL_TOKEN


class WebNLG(Dataset):
    def __init__(self, json_file, spm_model_file, for_testing=False,
                 bpe_dropout=0.0):
        super().__init__()

        with open(json_file) as f:
            nlg = json.load(f)
        entries = [next(iter(e.values())) for e in nlg['entries']]

        self.triplesets: List[FrozenSet[Tuple[str, str, str]]] = []
        self.entity_lists: List[List[str]] = []
        self.targets = []
        for e in entries:
            triples = []
            entities = OrderedDict()
            for trip in e["modifiedtripleset"]:
                sbj = ENT_TOKEN + preprocess_entity(trip['subject'])
                rel = REL_TOKEN + preprocess_relation(trip['property'])
                obj = ENT_TOKEN + preprocess_entity(trip['object'])
                entities[sbj] = None
                entities[obj] = None
                triples.append((sbj, rel, obj))
            self.entity_lists.append(list(entities.keys()))
            self.triplesets.append(frozenset(triples))

            targets = [preprocess_text(lex['lex'])
                       for lex in e["lexicalisations"]]
            self.targets.append(targets)

        self.for_testing = for_testing

        if not for_testing:
            self.idx2idx = []
            for i, t in enumerate(self.targets):
                for j in range(len(t)):
                    self.idx2idx.append((i, j))

        self.sp = spm.SentencePieceProcessor(model_file=spm_model_file)
        if for_testing:
            self.dropout_p = None
            self.use_bpe_dropout = False
        else:
            self.dropout_p = bpe_dropout
            self.use_bpe_dropout = 0.0 < bpe_dropout < 1.0

    def __getitem__(self, idx: int):
        if self.for_testing:
            internal_idx = idx
            target = self.targets[idx]
        else:
            internal_idx, target_idx = self.idx2idx[idx]
            target = self.targets[internal_idx][target_idx]

        target = self.sp.encode(
            target, enable_sampling=self.use_bpe_dropout,
            alpha=self.dropout_p, add_bos=True, add_eos=True
        )

        (dm, is_entity, pos), node_labels = graph_gen_helper(
            self.triplesets[internal_idx],
            self.entity_lists[internal_idx],
            self.sp,
            dropout_p=self.dropout_p
        )

        sample = (dm, is_entity, pos, node_labels, target)
        return sample

    def __len__(self) -> int:
        if self.for_testing:
            return len(self.targets)
        else:
            return len(self.idx2idx)

    def extract_fun(self, sample) -> str:
        return sample[-1]

    def num_tokens_fun(self, sample) -> int:
        node_labels = sample[-2]
        target = sample[-1]
        return len(node_labels) + len(target)
