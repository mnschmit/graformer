import torch
import math
from torch.nn import functional as F
from torch.nn.modules.activation import PReLU
from torch.nn.modules.transformer import MultiheadAttention
from .relative_attention import RelativeMultiheadAttention
from .relative_attention_scalars import RelativeMultiheadAttention as RMHAScalars
from torch_scatter import scatter_logsumexp


def _get_activation_fn(activation, num_parameters: int = 1):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "prelu":
        return PReLU(num_parameters=num_parameters)
    else:
        raise RuntimeError(
            "activation should be relu/gelu/prelu, not %s." % activation)


class CopyCatDecoderLayer(torch.nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, max_range, max_length,
                 dim_feedforward=2048, dropout=0.1, activation="gelu", attention_dropout=0.0,
                 use_full_pos_embeddings=False, rel_pos_embed=None,
                 use_gate=False, use_scaled_interattention=False, prenorm=False):  # prelu
        super().__init__()

        self.prenorm = prenorm

        if use_full_pos_embeddings:
            self.self_attn = RelativeMultiheadAttention(
                d_model, nhead, max_range * 2 + 1, dropout=attention_dropout,
                scalar_matrix_buffer=(max_length, max_range))
        else:
            if rel_pos_embed is None:
                rel_pos_embed = torch.nn.Embedding(max_range * 2 + 1, nhead)
            else:
                assert list(rel_pos_embed.weight.size()) == [
                    max_range * 2 + 1, nhead]

            self.self_attn = RMHAScalars(
                d_model, nhead, rel_pos_embed, max_length, dropout=attention_dropout)

        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=attention_dropout)
        self.multihead_attn2 = MultiheadAttention(
            d_model, nhead, dropout=attention_dropout)

        if use_scaled_interattention:
            self.multiplier = torch.nn.parameter.Parameter(
                torch.Tensor(1).fill_(1.0))
        else:
            self.multiplier = None

        self.use_gate = use_gate
        if use_gate:
            self.gate_linear = torch.nn.Linear(d_model, 1)

        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.norm4 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.dropout4 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def compute_cov_loss(self, normalized_attention_scores):
        # (batch_size, trg_len, src_len)
        cumulative_att = torch.cumsum(normalized_attention_scores, 1)
        max_att = torch.max(normalized_attention_scores, 1)[0].mean()
        return torch.where(
            normalized_attention_scores < cumulative_att,
            normalized_attention_scores,
            cumulative_att).sum(2).mean() - max_att
        # 0.5 * (a_g + c_g - torch.sqrt((a_g - c_g)**2))

    def compute_cov_pen(self, normalized_attention_scores):
        # (batch_size, src_len)
        aggregated_scores = torch.sum(normalized_attention_scores, 1)
        clipped_logs = torch.log(torch.min(aggregated_scores,
                                           torch.ones_like(aggregated_scores)))
        return clipped_logs.sum(1)

    def encoder_attention(self, target, memory, memory_mask, memory_key_padding_mask,
                          mhattn, dropout, norm, gate, multiplier):
        residual = target
        if self.prenorm:
            target = norm(target)
        intermediate, att = mhattn(
            target, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        intermediate = dropout(intermediate)

        if gate is not None:
            intermediate *= gate
            att = gate.transpose(0, 1) * att
        if multiplier is not None:
            intermediate *= multiplier

        intermediate += residual
        if not self.prenorm:
            target = norm(intermediate)
        else:
            target = intermediate

        return target, intermediate, att

    def forward(self, target, graph_memory,
                target_mask=None, graph_memory_mask=None,
                target_key_padding_mask=None,
                graph_memory_key_padding_mask=None, lm_pretraining=False):
        r"""Pass the inputs (and masks) through the decoder layer.

        Args:
            target: the sequence to the decoder layer (required).
            graph_memory: the sequnce from the last layer of the graph encoder (required).
            target_mask: the mask for the target sequence (optional).
            graph_memory_mask: the mask for the graph_memory sequence (optional).
            target_key_padding_mask: the mask for the target keys per batch (optional).
            graph_memory_key_padding_mask: the mask for the graph_memory keys per batch (optional).

        Shape:
            TODO
        """
        # self-attention
        residual = target
        if self.prenorm:
            target = self.norm1(target)
        target2 = self.self_attn(target, attn_mask=target_mask,
                                 key_padding_mask=target_key_padding_mask)[0]
        target = residual + self.dropout1(target2)
        if not self.prenorm:
            target = self.norm1(target)

        if lm_pretraining:
            gate = torch.zeros(
                (target.size(0), target.size(1), 1), device=target.device)
        else:
            # gate computation before any other attention
            if self.use_gate:
                gate = torch.sigmoid(self.gate_linear(target))
            else:
                gate = None

        # graph encoder attention
        target, intermediate, a_g = self.encoder_attention(
            target, graph_memory, graph_memory_mask, graph_memory_key_padding_mask,
            self.multihead_attn2, self.dropout3, self.norm3, gate, self.multiplier
        )

        # coverage calculations
        coverage_loss = self.compute_cov_loss(a_g)
        coverage_penalty = self.compute_cov_pen(a_g)

        # Feedforward Network
        residual = target
        if self.prenorm:
            target = self.norm4(target)
        target2 = self.linear2(self.dropout(
            self.activation(self.linear1(target))))
        target = residual + self.dropout4(target2)
        if not self.prenorm:
            target = self.norm4(target)

        return target, intermediate, a_g, coverage_loss, coverage_penalty


class CopyCatDecoder(torch.nn.Module):
    r"""CopyCatDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, vocab_size, d_model, decoder_layers, norm=None, tie_weights=None,
                 copynet=False, kgsum=False, gru_combine=False,
                 no_copy=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(decoder_layers)
        self.norm = norm

        self.is_copynet = copynet
        self.is_kgsum = kgsum
        self.use_gru = gru_combine
        self.do_copy = not no_copy

        if gru_combine:
            self.gru = torch.nn.GRUCell(d_model, d_model)

        if self.do_copy:
            self.w_gen = torch.nn.Linear(d_model, 1)

        if tie_weights is None:
            self.gen_prob_weight = torch.nn.Parameter(
                torch.FloatTensor(vocab_size, d_model)
            )
            torch.nn.init.kaiming_uniform_(
                self.gen_prob_weight, a=math.sqrt(5))
        else:
            self.gen_prob_weight = tie_weights

        self.gen_prob_bias = torch.nn.Parameter(
            torch.FloatTensor(vocab_size), True)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            self.gen_prob_weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.gen_prob_bias, -bound, bound)

        self.small = -1e9  # -10000.0
        self.epsilon = 1e-9

    def compute_coverage_regularizer(self, mean_normalized_attention_scores):
        # (batch_size, src_len)
        aggregated_scores = torch.sum(mean_normalized_attention_scores, 1)
        clipped_logs = torch.log(
            torch.clamp(aggregated_scores, min=self.epsilon, max=1.0)
        )
        return -clipped_logs.mean()  # mean over batch_size and src_len (before sum)

    def forward(self, target, graph_memory, graph_ids,
                target_mask=None, graph_memory_mask=None,
                target_key_padding_mask=None,
                graph_memory_key_padding_mask=None, copy_graph_padding_mask=None,
                lm_pretraining=False):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            target: the sequence to the decoder (required).
            target_mask: the mask for the target sequence (optional).
            memory_mask: the mask for the title_memory sequence (optional).
            target_key_padding_mask: the mask for the target keys per batch (optional).
            graph_memory_key_padding_mask: the mask for the title_memory keys per batch (optional).

        Shape:
            target: (trg_len, batch_size, hidden_dim)

            output: (batch_size, trg_len, vocab_size)
        """
        # (trg_len, batch_size, hidden_dim)
        output = target
        trg_len, batch_size, hidden_dim = target.size()
        # (trg_len, batch_size, hidden_dim)
        intermediate = None

        total_coverage_loss = None
        total_cov_pen = None
        a_g = None
        attention_accumulator = None
        for layer in self.layers:
            new_output, intermediate, a_g, layer_cov_loss, layer_cov_pen = layer(
                output, graph_memory, target_mask=target_mask,
                graph_memory_mask=graph_memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                graph_memory_key_padding_mask=graph_memory_key_padding_mask,
                lm_pretraining=lm_pretraining
            )

            if self.use_gru:
                output = self.gru(
                    new_output.reshape(trg_len * batch_size, -1),
                    output.reshape(trg_len * batch_size, -1)
                ).reshape(trg_len, batch_size, -1)
            else:
                output = new_output

            if attention_accumulator is None:
                attention_accumulator = a_g
            else:
                attention_accumulator += a_g

            if total_coverage_loss is None:
                total_coverage_loss = layer_cov_loss
            else:
                total_coverage_loss += layer_cov_loss
            if total_cov_pen is None:
                total_cov_pen = layer_cov_pen
            else:
                total_cov_pen += layer_cov_pen
        total_coverage_loss /= len(self.layers)
        total_cov_pen /= len(self.layers)
        mean_attention_scores = attention_accumulator / len(self.layers)
        coverage_reg = self.compute_coverage_regularizer(mean_attention_scores)

        if self.norm:
            output = self.norm(output)

        logits = F.linear(
            output, self.gen_prob_weight, bias=self.gen_prob_bias)
        logits = logits.transpose(0, 1)

        # copy attention
        # intermediate: (trg_len, batch_size, hidden_dim)
        # title_memory: (src_len, batch_size, hidden_dim)
        # graph_memory: (num_nodes, batch_size, hidden_dim)

        # copy attention computation
        if self.do_copy and not self.is_kgsum:
            if copy_graph_padding_mask is None:
                copy_graph_padding_mask = graph_memory_key_padding_mask
            a_g = torch.bmm(intermediate.transpose(0, 1),
                            graph_memory.permute(1, 2, 0))
            a_g.masked_fill_(
                copy_graph_padding_mask.unsqueeze(1).expand_as(a_g),
                self.small
            )

        if self.do_copy:
            if self.is_copynet:
                logits = self.copynet_logits(
                    logits, a_g, graph_ids
                )
            elif self.is_kgsum:
                logits = self.norm_copyprob_logits(
                    logits, a_g, output, graph_ids
                )
            else:
                a_g = F.softmax(a_g, dim=2)
                logits = self.norm_copyprob_logits(
                    logits, a_g, intermediate, graph_ids
                )

        return logits, coverage_reg, total_cov_pen

    def copynet_logits(self, logits, a_g, graph_ids):
        gen_probs = torch.empty(
            (logits.size(0), logits.size(1), logits.size(2), 2),
            device=logits.device
        )
        gen_probs[:, :, :, 0] = logits

        # (batch_size, trg_len, vocab_size) with -inf
        gen_probs[:, :, :, 1] = scatter_logsumexp(
            a_g,
            graph_ids.unsqueeze(1).expand_as(a_g),
            dim=2,
            dim_size=gen_probs.size(2)
        )

        return torch.logsumexp(gen_probs, 3)

    def norm_copyprob_logits(self, logits, a_g, intermediate, graph_ids):
        pr_gen = torch.sigmoid(self.w_gen(intermediate)).transpose(0, 1)
        gen_probs = F.softmax(logits, dim=2) * pr_gen

        gen_probs.scatter_add_(
            2,
            graph_ids.unsqueeze(1).expand_as(a_g),
            a_g * (1 - pr_gen)
        )

        return gen_probs
