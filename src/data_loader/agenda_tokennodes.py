from .curriculum_dataset import CurriculumDataset
import json


class AgendaCurriculum(CurriculumDataset):
    def __init__(self, graph_file, metadata_file, start_competence=0.1, full_competence_time=15,
                 for_testing=False):
        super().__init__(start_competence, full_competence_time, for_testing)

        with open(graph_file) as f:
            graphs = json.load(f)

        self.dm = graphs['distance_matrix']
        self.is_entity = graphs['is_entity']
        self.pos = graphs['positions']

        with open(metadata_file) as f:
            metadata = json.load(f)

        self.titles = metadata['title']
        self.node_labels = metadata['node_label']
        self.texts = metadata['abstract']

        self.compute_difficulties(self.texts, len)

    def __getitem__(self, idx):
        new_idx = self.convert_index(idx)
        sample = (self.dm[new_idx], self.is_entity[new_idx], self.pos[new_idx],
                  self.node_labels[new_idx], self.titles[new_idx], self.texts[new_idx])
        return sample

    def __len__(self):
        return len(self.available_samples)
