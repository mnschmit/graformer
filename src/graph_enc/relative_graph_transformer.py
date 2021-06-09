from typing import List, Optional
import torch
from torch import nn
from .graph_pos_embedding import GraphPosEmbedding
from ..models.relative_attention import RelativeMultiheadAttention
from ..models.relative_attention_scalars import RelativeMultiheadAttention as RMHAScalars
from ..models.CopyCatDecoder import _get_activation_fn


class RelativeGraphTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int,
                 num_graph_pos: int, same_text_range: int, same_text_num: int,
                 num_special_gpos: int,
                 dim_feedforward: int, dropout: float, attention_dropout: float,
                 activation="gelu",
                 use_full_pos_embeddings=False,
                 rel_pos_embed: Optional[nn.Embedding] = None,
                 prenorm=False):
        super().__init__()

        self.prenorm = prenorm

        self.max_range = num_graph_pos - 1
        self.same_text_num = same_text_num
        self.same_text_range = same_text_range

        if use_full_pos_embeddings:
            self.mha = RelativeMultiheadAttention(
                hidden_dim, num_heads, num_graph_pos + same_text_range,
                dropout=attention_dropout
            )
        else:
            if rel_pos_embed is None:
                rel_pos_embed = GraphPosEmbedding(
                    num_graph_pos, same_text_range,
                    self.hparams.num_heads, num_special_gpos
                )
            else:
                assert list(rel_pos_embed.weight.size()) == [
                    num_graph_pos + same_text_range, num_heads]

            self.mha = RMHAScalars(hidden_dim, num_heads,
                                   rel_pos_embed, -1, dropout=attention_dropout,
                                   scalar_matrix_buffer=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, node_features, distance_matrix, padding_mask):
        '''
        node_features: (batch_size, num_nodes, hidden_dim)
        distance_matrix: (batch_size, num_nodes, num_nodes)
        padding_mask: (batch_size, num_nodes)
        '''

        dm = torch.where(
            distance_matrix > self.same_text_num,
            torch.clamp_max(
                distance_matrix + self.max_range - self.same_text_num,
                self.same_text_range + self.max_range
            ),
            torch.clamp_max(distance_matrix, self.max_range)
        )

        # (num_nodes, batch_size, hidden_dim)
        feat = node_features.transpose(0, 1)

        if self.prenorm:
            feat = self.norm1(feat)  # norm before layer

        src2 = self.mha(feat, scalar_matrix=dm,
                        key_padding_mask=padding_mask)[0]
        # (batch_size, num_nodes, hidden_dim)
        src2 = src2.transpose(0, 1)

        src = node_features + self.dropout1(src2)

        if not self.prenorm:
            src = self.norm1(src)  # norm after layer

        # Feedforward net
        residual = src
        if self.prenorm:
            src = self.norm2(src)  # norm before layer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.prenorm:
            src = self.norm2(src)  # norm after layer

        return src


class RelativeGraphTransformerEncoder(nn.Module):

    def __init__(self, d_model,
                 encoder_layers: List[RelativeGraphTransformerEncoderLayer],
                 norm=None,
                 gru_combine=False):
        super().__init__()

        self.layers = nn.ModuleList(encoder_layers)
        self.norm = norm

        self.use_gru = gru_combine
        if gru_combine:
            self.gru = nn.GRUCell(d_model, d_model)

    def forward(self, node_label_features, distance_matrix, padding_mask):
        '''
        Input:
            - node_label_features: node embeddings based on their labels
                size: (batch_size, num_nodes, hidden_dim)
            - distance_matrix: (batch_size, num_nodes, num_nodes)
            - padding_mask: (batch_size, num_nodes); True for real tokens, False for pad tokens

        Output: 
            - node embeddings: (batch_size, num_nodes, hidden_dim)
        '''

        # (batch_size, num_nodes, hidden_dim)
        node_embeddings = node_label_features
        batch_size, num_nodes, hidden_dim = node_label_features.size()
        for layer in self.layers:
            new_node_embeddings = layer(
                node_embeddings, distance_matrix, padding_mask)

            if self.use_gru:
                node_embeddings = self.gru(
                    new_node_embeddings.reshape(batch_size * num_nodes, -1),
                    node_embeddings.reshape(batch_size * num_nodes, -1)
                ).reshape(batch_size, num_nodes, -1)
            else:
                node_embeddings = new_node_embeddings

        if self.norm:
            node_embeddings = self.norm(node_embeddings)

        return node_embeddings
