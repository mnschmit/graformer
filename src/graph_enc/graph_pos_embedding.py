import torch
from torch.nn import Embedding
from overrides import overrides


class GraphPosEmbedding(Embedding):
    @overrides
    def __init__(self, num_graph_embeddings, num_text_embeddings,
                 num_heads, num_special_gpos, static_extremes=False,
                 no_gpos=False, **kwargs):
        num_embeddings = num_graph_embeddings + num_text_embeddings
        super().__init__(num_embeddings, num_heads, **kwargs)

        self.num_graph_embeddings = num_graph_embeddings
        self.num_special_gpos = num_special_gpos
        self.static_extremes = static_extremes
        self.no_gpos = no_gpos

        if static_extremes:
            neg_inf = -1e9
            weights = torch.zeros(num_embeddings, num_heads)
            weights[1:num_special_gpos+1, :num_heads//2] = neg_inf
            weights[num_special_gpos+2:, :num_heads//2] = neg_inf
            self.weight = torch.nn.Parameter(weights, False)
        elif no_gpos:
            weights1 = torch.zeros(num_graph_embeddings, num_heads)
            weights2 = torch.empty(num_text_embeddings, num_heads)
            torch.nn.init.uniform_(weights2, -1.0, 1.0)
            self.weight1 = torch.nn.Parameter(weights1, False)
            self.weight2 = torch.nn.Parameter(weights2, True)
        else:
            torch.nn.init.uniform_(self.weight, -1.0, 1.0)

    def get_weights(self):
        if self.no_gpos:
            return torch.cat([self.weight1, self.weight2])
        else:
            return self.weight  # * self.weight

    def gpos_loss(self):
        if self.static_extremes or self.no_gpos:
            return self.get_weights().new_zeros(1)

        start = self.num_special_gpos
        end = self.num_graph_embeddings

        # (num_graph_embeddings//2, pos/neg, num_heads//2)
        constrained_weights = self.get_weights()[start:end, :self.embedding_dim//2].reshape(
            (end-start)//2, 2, -1
        )

        bigger_pos = constrained_weights[:-1, 0, :]
        smaller_pos = constrained_weights[1:, 0, :]
        bigger_neg = constrained_weights[:-1, 1, :]
        smaller_neg = constrained_weights[1:, 1, :]

        # uncomment to also put constraints for second half of heads
        # bigger2 = self.weight[start+1:end, self.embedding_dim//2:]
        # smaller2 = self.weight[start:end-1, self.embedding_dim//2:]
        # torch.max(self.ideal_ratio, smaller2 / bigger2)

        # return torch.log(torch.max(self.ideal_ratio, smaller_pos / bigger_pos)
        #                  + torch.max(self.ideal_ratio, smaller_neg / bigger_neg)).mean()

        return (torch.clamp_min(smaller_pos - bigger_pos, 0.0)
                + torch.clamp_min(smaller_neg - bigger_neg, 0.0)).mean()

    def _compute_head_factors(self, rel_pos_encodings, reverse=False):
        special_mask = rel_pos_encodings < self.num_special_gpos
        halfed = ((rel_pos_encodings - self.num_special_gpos + 2) / 2).float()

        if reverse:
            largest_input = float(
                (self.num_embeddings - self.num_special_gpos + 1) // 2)
            halfed = largest_input - halfed + 1

        return torch.where(special_mask, torch.ones_like(halfed), 1/halfed)

    def _compute_factors(self, rel_pos_encodings):
        factor_matrix = []
        for head in range(self.embedding_dim):
            factor_matrix.append(self._compute_head_factors(
                rel_pos_encodings, reverse=head % 2 == 0).unsqueeze(-1))
        return torch.cat(factor_matrix, dim=-1)

    def forward(self, x):
        if self.no_gpos:
            weight = torch.cat([self.weight1, self.weight2])
            return weight[x]
        else:
            return super().forward(x)
