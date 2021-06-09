from torch.utils.data import Dataset
from collections import defaultdict
import math


class CurriculumDataset(Dataset):
    def __init__(self, start_competence: float, full_competence_time: int, for_testing: bool):
        super().__init__()

        self.c0_square = start_competence ** 2
        self.T = full_competence_time
        self.competence = start_competence
        self.for_testing = for_testing

    def convert_index(self, idx: int):
        if self.for_testing:
            return idx
        else:
            return self.available_samples[idx]

    def compute_difficulties(self, data, diff_fun):
        if self.for_testing:
            self.available_samples = range(len(data))
            return
        else:
            self.available_samples = []

        diff_data = defaultdict(list)
        for idx, d in enumerate(data):
            diff_data[diff_fun(d)].append(idx)
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

        self.compute_available_samples()

    def compute_available_samples(self):
        self.available_samples.clear()
        for d in self.difficulties:
            if self.diff_cdf[d] < self.competence\
               or math.isclose(self.diff_cdf[d], self.competence):
                self.available_samples.extend(self.diff2textids[d])
            else:
                break

    def evaluate_competence(self, epoch: int):
        self.competence = min(1., math.sqrt(
            epoch * (1 - self.c0_square) / self.T + self.c0_square))

    def end_of_epoch(self, epoch: int):
        self.evaluate_competence(epoch)
        self.compute_available_samples()
