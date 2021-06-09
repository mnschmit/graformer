from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.modules.transformer import MultiheadAttention


def generate_square_rel_pos_scalar_matrix(seq_len: int, max_range: int) -> torch.Tensor:
    matrix = torch.zeros(seq_len, seq_len, dtype=torch.long)
    for i in range(seq_len):
        uidx = torch.triu_indices(seq_len, seq_len, i+1)
        matrix[uidx[0], uidx[1]] += 1
        lidx = torch.tril_indices(seq_len, seq_len, -i-1)
        matrix[lidx[0], lidx[1]] -= 1
    return matrix.clamp(min=-max_range, max=max_range) + max_range


class RelativeMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, num_pos_embeddings: int,
                 dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, scalar_matrix_buffer=None):
        super().__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias,
            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim)

        self.scaling = float(self.head_dim) ** -0.5
        self.pos_embed1 = torch.nn.Embedding(num_pos_embeddings, self.head_dim)
        # self.pos_embed2 = torch.nn.Embedding(num_pos_embeddings, self.head_dim)

        if scalar_matrix_buffer is not None:
            max_length, max_range = scalar_matrix_buffer
            self.register_buffer(
                'scalar_matrix',
                generate_square_rel_pos_scalar_matrix(max_length, max_range)
            )

    def obtain_pos_embeddings(self, scalar_matrix: Optional[torch.LongTensor],
                              bsz: int, tgt_len: int, src_len: int,
                              pos_embeddings: torch.nn.Embedding):
        # (src_len, tgt_len)
        if scalar_matrix is None:
            relative_positions = self.scalar_matrix[:src_len, :tgt_len]
            # (src_len, tgt_len, head_dim)
            pos_emb = pos_embeddings(relative_positions)
            # (batch_size * num_heads, tgt_len, head_dim, src_len)
            pos_emb = pos_emb.permute(1, 2, 0).unsqueeze(0).expand(
                bsz * self.num_heads, -1, -1, -1)
        else:
            # (batch_size, src_len, tgt_len, head_dim)
            pos_emb = pos_embeddings(scalar_matrix)
            # (batch_size * num_heads, tgt_len, head_dim, src_len)
            pos_emb = pos_emb.permute(0, 2, 3, 1).repeat(
                self.num_heads, 1, 1, 1)

        # (bsz * num_heads, tgt_len, head_dim, src_len)
        return pos_emb

    def forward(self, qkv, key_padding_mask=None,
                need_weights=True, attn_mask=None, scalar_matrix=None):
        tgt_len, bsz, embed_dim = qkv.size()
        assert embed_dim == self.embed_dim

        # self-attention (apparently chunk separates the views)
        q, k, v = F.linear(qkv, self.in_proj_weight,
                           self.in_proj_bias).chunk(3, dim=-1)
        q *= self.scaling

        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)

        q = q.reshape(tgt_len, bsz * self.num_heads,
                      self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.reshape(-1, bsz * self.num_heads,
                          self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.reshape(-1, bsz * self.num_heads,
                          self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()
                                          [2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()
                                          [2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                              dtype=attn_mask.dtype,
                                                              device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len]

        # !RELATIVE! add position attention weight corrections
        # (bsz * num_heads, tgt_len, head_dim, src_len)
        pos_emb1 = self.obtain_pos_embeddings(
            scalar_matrix, bsz, tgt_len, src_len, self.pos_embed1)

        # q = (bsz * self.num_heads, tgt_len, head_dim)
        # k = (bsz * self.num_heads, src_len, head_dim)
        # (batch_size * num_heads, tgt_len, src_len)
        weight_corrections = torch.matmul(
            q.unsqueeze(2),
            pos_emb1
        ).squeeze(2)
        attn_output_weights += weight_corrections

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training)

        # v = (bsz * self.num_heads, src_len, head_dim)
        # (bsz * num_heads, tgt_len, head_dim)
        attn_output = torch.bmm(attn_output_weights, v)

        # # !RELATIVE! add position embedding weighted sum
        # # (bsz * num_heads, tgt_len, head_dim, src_len)
        # pos_emb2 = self.obtain_pos_embeddings(
        #     scalar_matrix, bsz, tgt_len, src_len, self.pos_embed2)

        # # (bsz * num_heads, tgt_len, 1, head_dim)
        # weighted_pos_sum = torch.matmul(
        #     attn_output_weights.unsqueeze(2),
        #     pos_emb2.transpose(2, 3)
        # )
        # attn_output += weighted_pos_sum.squeeze(2)

        assert list(attn_output.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(
            0, 1).reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None
