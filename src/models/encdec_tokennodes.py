from argparse import ArgumentParser
import os

import math
import torch
import torch.distributed as dist

from ..graph_enc.relative_graph_transformer import RelativeGraphTransformerEncoder,\
    RelativeGraphTransformerEncoderLayer
from ..graph_enc.graph_pos_embedding import GraphPosEmbedding
from torch.nn.modules.transformer import LayerNorm
from .CopyCatDecoder import CopyCatDecoderLayer, CopyCatDecoder
from .beam_search import BeamHypotheses, top_k_top_p_filtering
from .adafactor import Adafactor
from .loss import LabelSmoothingLoss
from .sinusoidal import SinusoidalPositionalEmbedding

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from typing import Dict, Any, List, Optional, Union
import pytorch_lightning as pl
from torch.nn import functional as F

from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, RandomSampler
from ..data_loader.bucket_batch_sampler import BucketBatchSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
# from ..data_loader.agenda_tokennodes import AgendaCurriculum
from ..data_loader.agenda_dataset import Agenda
from ..data_loader.webnlg_dataset import WebNLG
from ..data_loader.webnlg_dataset2 import WebNLG as WebNLG_BPEDropout
from ..data_loader.curriculum_batch_sampler import CurriculumBatchSampler
from ..data_loader.token_batch_sampler import TokenBatchSampler
from ..data_loader.curriculum_token_batch_sampler import CurriculumTokenBatchSampler

from ..data_loader.handle_data_tokennodes import collate_webnlg
from nlgeval import NLGEval

import random
import sentencepiece as spm


def generate_square_subsequent_mask(sz, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, float(0.0))
    return mask


class Graph2Text(pl.LightningModule):

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters(hparams)

        # self.hparams = Namespace(**hparams) if type(hparams) is dict else hparams

        try:
            light_validation = self.hparams.light_validation
        except AttributeError:
            light_validation = False
        try:
            verbose = self.hparams.verbose
        except AttributeError:
            verbose = False
        try:
            no_validation = self.hparams.no_validation
        except AttributeError:
            no_validation = False
        try:
            activation = self.hparams.activation
        except AttributeError:
            activation = "gelu"   # activation="prelu"

        # if isinstance(hparams, Mapping):
        #     self.hparams = SimpleNamespace(**hparams)
        # else:
        #     self.hparams = hparams

        # self.hparams = hparams

        if self.hparams.webNLG:
            min_length = 7
            max_length = 115  # 110
            max_ent_len = 62  # 61
            max_path_idx = 2 * 8 + 1
        else:
            # i.e. AGENDA
            min_length = 13
            max_length = 560
            max_ent_len = 46

        self.spm_model_path = os.path.join(
            self.hparams.data_root, 'bpe.model')
        sp = self.load_spm_model()

        self.vocab_size = len(sp)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        del sp

        try:
            self.lr_reduce_patience = self.hparams.lr_reduce_patience
        except AttributeError:
            self.lr_reduce_patience = 3

        try:
            self.lr_warmup_steps = self.hparams.lr_warmup_steps
        except AttributeError:
            self.lr_warmup_steps = 0

        self.repetition_penalty = self.hparams.repetition_penalty
        self.coverage_weight = self.hparams.coverage_weight
        self.gpos_regularizer = self.hparams.gpos_regularizer
        self.l2_regularizer = self.hparams.l2_regularizer

        if hasattr(self.hparams, 'bpe_dropout') and self.hparams.bpe_dropout > 0.0:
            self.train_data = (os.path.join(
                self.hparams.data_root, 'train.json'),)
            self.val_data = (os.path.join(
                self.hparams.data_root, 'dev.json'),)
            self.test_data = (os.path.join(
                self.hparams.data_root, 'test.json'),)
        else:
            self.train_data = (
                os.path.join(self.hparams.data_root, 'train-graphs.json'),
                os.path.join(self.hparams.data_root, 'train-texts.json')
            )
            self.val_data = (
                os.path.join(self.hparams.data_root, 'val-graphs.json'),
                os.path.join(self.hparams.data_root, 'val-texts.json')
            )
            self.test_data = (
                os.path.join(self.hparams.data_root, 'test-graphs.json'),
                os.path.join(self.hparams.data_root, 'test-texts.json')
            )

        self.val_ref_file = os.path.join(
            self.hparams.data_root, 'val-ref.txt'
        )
        self.test_ref_file = os.path.join(
            self.hparams.data_root, 'test-ref.txt'
        )

        self.distributed = len(str(self.hparams.gpus)) > 1
        self.verbose = verbose
        self.no_validation = no_validation
        self.light_validation = light_validation
        self.use_webNLG = self.hparams.webNLG

        self.batch_size = self.hparams.batch_size
        self.inference_batch_size = None

        self.hidden_dim = self.hparams.hidden_dim
        self.num_beams = min(self.hparams.beam_size, self.hparams.batch_size)
        self.length_penalty = self.hparams.length_penalty
        self.coverage_penalty = self.hparams.coverage_penalty

        try:
            self.temperature = self.hparams.temperature
        except AttributeError:
            self.temperature = None
        try:
            self.top_k = self.hparams.top_k
        except AttributeError:
            self.top_k = 0
        try:
            self.top_p = self.hparams.top_p
        except AttributeError:
            self.top_p = 1.0

        self.shuffle_data = self.hparams.shuffle_data

        self.max_length = max_length
        self.min_length = min_length
        # dim_feedforward = min(5 * self.hparams.hidden_dim, 2000)
        # dim_feedforward = min(2048, 4 * self.hparams.hidden_dim)
        dim_feedforward = self.hparams.dim_feedforward

        self.word_embeddings = torch.nn.Embedding(
            self.vocab_size, self.hparams.hidden_dim, padding_idx=self.pad_id)

        torch.nn.init.kaiming_uniform_(self.word_embeddings.weight)

        try:
            input_dropout_p = self.hparams.input_dropout
        except AttributeError:
            input_dropout_p = 0.0
        self.input_dropout = torch.nn.Dropout(input_dropout_p)

        try:
            ls_epsilon = self.hparams.label_smoothing
        except AttributeError:
            ls_epsilon = 0.1
        self.loss = LabelSmoothingLoss(
            ls_epsilon, self.vocab_size, ignore_index=self.pad_id)

        num_special_gpos = 4

        num_graph_pos_embeddings = max_path_idx + 1 if self.hparams.max_graph_range is None\
            else self.hparams.max_graph_range * 2 + num_special_gpos
        num_same_text_embeddings = (max_ent_len-1) * 2 if self.hparams.same_text_range is None\
            else self.hparams.same_text_range * 2

        self.no_gpos = self.hparams.no_gpos\
            if hasattr(self.hparams, 'no_gpos') else False

        if self.hparams.share_pos_across_layers and not self.hparams.full_pos_embeddings\
           and not self.hparams.sinusoidals:
            self.graph_pos_embed = GraphPosEmbedding(
                num_graph_pos_embeddings, num_same_text_embeddings,
                self.hparams.num_heads, num_special_gpos,
                static_extremes=self.hparams.gpos_static_extremes,
                no_gpos=self.no_gpos
            )
        else:
            self.graph_pos_embed = None
        graph_encoder_norm = LayerNorm(self.hparams.hidden_dim, eps=1e-6)

        try:
            self.transformer_prenorm = self.hparams.prenorm
        except AttributeError:
            self.transformer_prenorm = False

        if self.hparams.sinusoidals:
            self.graph_pos_embed = SinusoidalPositionalEmbedding(
                self.hparams.hidden_dim, self.pad_id
            )
            self.graph_enc = torch.nn.modules.transformer.TransformerEncoder(
                torch.nn.modules.transformer.TransformerEncoderLayer(
                    self.hparams.hidden_dim, self.hparams.num_heads,
                    dim_feedforward=self.hparams.dim_feedforward,
                    dropout=self.hparams.dropout,
                    activation=activation
                ),
                self.hparams.num_encoder_layers,
                norm=graph_encoder_norm
            )
        else:
            self.graph_enc = RelativeGraphTransformerEncoder(
                self.hparams.hidden_dim,
                [
                    RelativeGraphTransformerEncoderLayer(
                        self.hparams.hidden_dim, self.hparams.num_heads,
                        num_graph_pos_embeddings, num_same_text_embeddings, 1000,
                        num_special_gpos,
                        dim_feedforward, self.hparams.dropout,
                        self.hparams.attention_dropout,
                        activation=activation,
                        use_full_pos_embeddings=self.hparams.full_pos_embeddings,
                        rel_pos_embed=self.graph_pos_embed,
                        prenorm=self.transformer_prenorm
                    )
                    for _ in range(self.hparams.num_encoder_layers)
                ],
                norm=graph_encoder_norm
            )

        if hasattr(self.hparams, 'bpe_dropout') and self.hparams.bpe_dropout > 0.0:
            max_length *= 2

        if self.hparams.share_pos_across_layers and not self.hparams.full_pos_embeddings:
            self.rel_pos_embed = torch.nn.Embedding(
                self.hparams.max_text_range * 2 + 1, self.hparams.num_heads
            )
        else:
            self.rel_pos_embed = None

        decoder_norm = LayerNorm(self.hparams.hidden_dim, eps=1e-6)
        tied_weights = None
        if self.hparams.tie_weights:
            tied_weights = self.word_embeddings.weight

        try:
            self.no_copy = self.hparams.no_copy
        except AttributeError:
            self.no_copy = False

        self.decoder = CopyCatDecoder(
            self.vocab_size, self.hparams.hidden_dim,
            [
                CopyCatDecoderLayer(
                    self.hparams.hidden_dim, self.hparams.num_heads,
                    self.hparams.max_text_range, max_length,
                    dim_feedforward=dim_feedforward,
                    dropout=self.hparams.dropout, activation=activation,
                    attention_dropout=self.hparams.attention_dropout,
                    use_full_pos_embeddings=self.hparams.full_pos_embeddings,
                    rel_pos_embed=self.rel_pos_embed,
                    use_gate=self.hparams.with_gate,
                    use_scaled_interattention=self.hparams.scaled_interattention,
                    prenorm=self.transformer_prenorm
                ) for _ in range(self.hparams.num_decoder_layers)
            ],
            norm=decoder_norm, tie_weights=tied_weights,
            copynet=self.hparams.copynet,
            kgsum=self.hparams.kgsum,
            no_copy=self.no_copy
        )

        self.nlgeval = NLGEval(metrics_to_omit=[
            'ROUGE_L', 'CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilarity',
            'VectorExtremaCosineSimilarity', 'GreedyMatchingScore', 'METEOR'
        ])

        self.test_output_fn = None
        self.test_scores = None

        trainset, collate = self.determine_dataset(*self.train_data)
        self.num_training_steps = (
            len(trainset) // self.batch_size + 1) * self.hparams.num_epochs

        self.trainsampler = None

    def load_spm_model(self):
        sp = spm.SentencePieceProcessor(model_file=self.spm_model_path)
        return sp

    def activate_meteor(self):
        self.nlgeval = NLGEval(metrics_to_omit=[
            'ROUGE_L', 'CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilarity',
            'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'
        ])

    def set_test_ref_file(self, filename):
        self.test_ref_file = filename

    def set_test_output(self, filename):
        self.test_output_fn = filename

    def set_test_data(self, graph_file, text_file):
        self.test_data = (graph_file, text_file)

    def reset_test_scores(self):
        self.test_scores = None

    def set_hparams(self, hparams: Dict[str, Optional[Union[int, float]]]):
        self.num_beams = hparams.get('beam_size', 1)
        self.length_penalty = hparams.get('length_penalty', 0.0)
        self.coverage_penalty = hparams.get('coverage_penalty', 0.0)
        self.repetition_penalty = hparams.get('repetition_penalty', 1.0)
        self.temperature = hparams.get('temperature', None)
        self.top_k = hparams.get('top_k', 0)
        self.top_p = hparams.get('top_p', 1.0)

    def encode_input(self, dm, is_entity_mask, node_pos, node_labels, nl_pad_mask):
        # (batch_size, num_nodes, hidden_dim)
        node_embeddings = self.word_embeddings(node_labels)

        if not node_embeddings.requires_grad and self.training:
            print("NODE EMBEDDINGS DO NOT REQUIRE GRADIENTS !")
            print(node_labels)

        node_embeddings = self.input_dropout(node_embeddings)

        if not node_embeddings.requires_grad and self.training:
            print("NODE EMBEDDINGS AFTER DROPOUT DO NOT REQUIRE GRADIENTS !")
            print(node_embeddings)

        if self.hparams.sinusoidals:
            node_embeddings += self.graph_pos_embed(node_labels)
            self.graph_enc.forward(
                node_embeddings.transpose(0, 1), src_key_padding_mask=nl_pad_mask)
        else:
            # (batch_size, num_nodes, hidden_dim)
            node_embeddings = self.graph_enc(
                node_embeddings, dm, nl_pad_mask
            )

        return node_embeddings.transpose(0, 1)

    def forward(self, dm, is_entity_mask, node_positions, node_labels, nl_pad_mask,
                text, text_pad_mask):
        graph_memory = self.encode_input(
            dm, is_entity_mask, node_positions, node_labels, nl_pad_mask
        )

        if not graph_memory.requires_grad and self.training:
            print("GRAPH MEMORY DOES NOT REQUIRE GRADIENTS !")

        subsequent_mask = generate_square_subsequent_mask(
            text.size(1), text.device)

        logits, coverage_term, _ = self.step(
            text, text_pad_mask, subsequent_mask,
            graph_memory, nl_pad_mask, node_labels
        )

        return logits, coverage_term

    def step(self, context, tgt_key_padding_mask, tgt_mask,
             graph_memory, graph_memory_padding_mask, node_labels):
        embedded_context = self.input_dropout(self.word_embeddings(context))

        lm_pretraining = self.current_epoch < self.hparams.num_lm_epochs

        # (batch_size, trg_len, vocab_size)
        logits, coverage_term, cov_pen = self.decoder(
            embedded_context.transpose(0, 1),
            graph_memory,
            node_labels,
            target_mask=tgt_mask, target_key_padding_mask=tgt_key_padding_mask,
            graph_memory_key_padding_mask=graph_memory_padding_mask,
            lm_pretraining=lm_pretraining
        )

        return logits, coverage_term, cov_pen

    def sample_sequence(self, *args):
        if self.num_beams > 1:
            return self.sample_sequence_beamsearch(*args)
        else:
            return self.sample_sequence_greedy(*args)

    def sample_sequence_greedy(self, dm, is_entity, positions,
                               node_labels, nl_pad_mask):
        batch_size = dm.size(0)

        generated = torch.full(
            (batch_size, self.max_length), self.pad_id,
            dtype=torch.long, device=nl_pad_mask.device
        )
        generated[:, 0] = self.bos_id

        # length of generated sentences / unfinished sentences
        unfinished_sents = node_labels.new(batch_size).fill_(1)

        # input encodings
        graph_memory = self.encode_input(
            dm, is_entity, positions, node_labels, nl_pad_mask
        )

        # decoding loop
        for cur_len in range(1, self.max_length):
            # TODO: build in past for a speed up in decoding
            next_token_logits, _, cp = self.step(
                generated[:, :cur_len], None, None,
                graph_memory, nl_pad_mask, node_labels
            )
            # only need last predictions
            next_token_logits = next_token_logits[:, -1, :]

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            if self.repetition_penalty != 1.0:
                self.enforce_repetition_penalty(
                    next_token_logits, batch_size, 1, generated[:, :cur_len])

            # set eos token prob to zero if min_length is not reached
            if cur_len < self.min_length:
                next_token_logits[:, self.eos_id] = -float("inf")

            temperature = self.temperature
            # select next words: greedy or sample
            if temperature is None or temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                if temperature != 1.0:
                    next_token_logits /= temperature

                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    next_token_logits, top_k=self.top_k, top_p=self.top_p
                )  # (batch_size, vocab_size)

                next_token = torch.multinomial(
                    F.softmax(scores, dim=-1), num_samples=1
                ).squeeze(-1)

            tokens_to_add = next_token * unfinished_sents + \
                (self.pad_id) * (1 - unfinished_sents)

            generated[:, cur_len] = tokens_to_add
            unfinished_sents.mul_((tokens_to_add != self.eos_id).long())

            # stop when we are done with each sentence
            if unfinished_sents.max() == 0:
                break
        cur_len += 1

        return generated[:, :cur_len]

    def sample_sequence_beamsearch(self, dm, is_entity, positions,
                                   node_labels, nl_pad_mask):
        batch_size = dm.size(0)
        do_not_sample = self.temperature is None or self.temperature == 0

        generated = torch.full(
            (batch_size * self.num_beams, 1), self.bos_id,
            dtype=torch.long, device=nl_pad_mask.device
        )

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(
                self.num_beams, self.max_length,
                self.length_penalty, self.coverage_penalty,
                early_stopping=True
            ) for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros(
            (batch_size, self.num_beams), dtype=torch.float, device=nl_pad_mask.device)
        if do_not_sample:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        done = [False for _ in range(batch_size)]

        # input encodings
        graph_memory = self.encode_input(
            dm, is_entity, positions, node_labels, nl_pad_mask
        )
        # expand (encoded) input to num beams

        num_nodes = graph_memory.size(0)
        node_labels = node_labels.unsqueeze(1).expand(
            batch_size, self.num_beams, num_nodes).contiguous(
        ).view(batch_size * self.num_beams, num_nodes)
        graph_memory = graph_memory.unsqueeze(2).expand(
            num_nodes, batch_size, self.num_beams, self.hidden_dim).contiguous(
        ).view(num_nodes, batch_size * self.num_beams, self.hidden_dim)
        graph_memory_padding_mask = nl_pad_mask.unsqueeze(1).expand(
            batch_size, self.num_beams, num_nodes).contiguous().view(
                batch_size * self.num_beams, num_nodes)

        # decoding loop
        for cur_len in range(1, self.max_length):
            # TODO: build in past for a speed up in decoding
            next_token_logits, _, cp = self.step(
                generated, None, None,
                graph_memory, graph_memory_padding_mask, node_labels
            )
            # only need last predictions
            next_token_logits = next_token_logits[:, -1, :]

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            if self.repetition_penalty != 1.0:
                self.enforce_repetition_penalty(
                    next_token_logits, batch_size, self.num_beams, generated
                )

            # (batch_size * num_beams, vocab_size)
            scores = F.log_softmax(next_token_logits, dim=-1)

            # set eos token prob to zero if min_length is not reached
            if cur_len < self.min_length:
                next_token_logits[:, self.eos_id] = -float("inf")

            # select next words: greedy or sample
            if do_not_sample:
                # Add the log prob of the new beams to the log prob
                # of the beginning of the sequence (sum of logs == log of the product)
                # (batch_size * num_beams, vocab_size)
                next_scores = scores + beam_scores[:, None].expand_as(scores)

                # re-organize to group the beam together (we keep top hypothesis across beams)
                next_scores = next_scores.view(
                    batch_size, self.num_beams * self.vocab_size
                )  # (batch_size, num_beams * vocab_size)

                _next_scores, next_tokens = torch.topk(
                    next_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True)

                next_scores = _next_scores
            else:
                if self.temperature != 1.0:
                    next_token_logits /= self.temperature

                # (batch_size * num_beams, vocab_size)
                _scores = scores + beam_scores[:, None].expand_as(scores)
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, self.num_beams * self.vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam
                # (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                # (batch_size, num_beams * 2)
                next_tokens = torch.multinomial(
                    probs, num_samples=2 * self.num_beams)
                # Compute next scores
                # (batch_size, num_beams * 2)
                next_scores = torch.gather(_scores, -1, next_tokens)
                # sort the sampled vector to make sure
                # that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(
                    next_scores, descending=True, dim=1)
                # (batch_size, num_beams * 2)
                next_tokens = torch.gather(
                    next_tokens, -1, next_scores_indices)

                # # Top-p/top-k filtering
                # scores = top_k_top_p_filtering(
                #     next_token_logits, top_k=self.top_k, top_p=self.top_p,
                #     min_tokens_to_keep=max(2, self.num_beams)
                # )  # (batch_size * num_beams, vocab_size)

                # # proc_rank = self.get_proc_rank()
                # # if proc_rank == 1:
                # #     print('-----', next_token_logits, scores, sep='\n')

                # # Sample 2 next words for each beam
                # # (so we have some spare tokens and match output of greedy beam search)
                # next_words = torch.multinomial(
                #     F.softmax(scores, dim=-1), num_samples=2
                # )  # (batch_size * num_beams, 2)
                # # Compute next scores
                # # (batch_size * num_beams, vocab_size)
                # _scores = F.log_softmax(scores, dim=-1)
                # # (batch_size * num_beams, 2)
                # _scores = torch.gather(_scores, -1, next_words)
                # # (batch_size * num_beams, 2)
                # next_scores = _scores + beam_scores[:, None].expand_as(_scores)
                # # Match shape of greedy beam search
                # # (batch_size, 2 * num_beams)
                # next_words = next_words.view(batch_size, 2 * self.num_beams)
                # # (batch_size, 2 * num_beams)
                # next_scores = next_scores.view(batch_size, 2 * self.num_beams)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * self.num_beams)

            # next batch beam content
            # list length: (batch_size * num_beams)
            # list elements: tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= self.num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(
                        self.num_beams)

                    next_batch_beam.extend(
                        [(0, self.pad_id, 0)] * self.num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (
                        beam_token_id, beam_token_score) in enumerate(
                            zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    # get beam and word IDs
                    beam_id = beam_token_id // self.vocab_size
                    token_id = beam_token_id % self.vocab_size

                    effective_beam_id = batch_idx * self.num_beams + beam_id

                    cov_pen = cp[effective_beam_id]

                    # end of sentence, or next word
                    if token_id.item() == self.eos_id:  # or cur_len + 1 == self.max_length:
                        # if beam_token does not belong to top num_beams tokens,
                        # it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            generated[effective_beam_id, :cur_len].clone(),
                            beam_token_score.item(),
                            cov_pen.item()
                        )
                    else:
                        next_sent_beam.append(
                            (beam_token_score, token_id, effective_beam_id)
                        )

                    # is the beam for the next step full ?
                    if len(next_sent_beam) == self.num_beams:
                        break

                # decide if we are done with this sentence
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # update next beam content
                assert len(
                    next_sent_beam) == self.num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == self.num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * self.num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = generated.new([x[1] for x in next_batch_beam])
            beam_idx = generated.new([x[2] for x in next_batch_beam])

            # re-order batch
            generated = generated[beam_idx, :]
            generated = torch.cat(
                [generated, beam_tokens.unsqueeze(1)], dim=-1)

            # TODO: REORDER INTERNAL STATES IF YOU USE past

        cur_len += 1

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores
            # if not eos and batch_idx not done
            if all(
                (token_id % self.vocab_size).item() != self.eos_id
                    for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[
                        batch_idx, :self.num_beams
                    ] == beam_scores.view(
                        batch_size, self.num_beams
                    )[batch_idx]
                ), (
                    "If batch_idx is not done, final next scores:"
                    + " {} have to equal to accumulated beam_scores: {}"
                ).format(
                    next_scores[:, :self.num_beams][batch_idx],
                    beam_scores.view(batch_size, self.num_beams)[batch_idx]
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = generated[effective_beam_id]
                cov_pen = cp[effective_beam_id]
                generated_hyps[batch_idx].add(
                    final_tokens, final_score, cov_pen)

        # select the best hypotheses
        sent_lengths = generated.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.beams, key=lambda x: x[0])[1]
            sent_lengths[i] = len(best_hyp)  # + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
            decoded = generated.new(
                batch_size, sent_max_len).fill_(self.pad_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_length:
                    decoded[i, sent_lengths[i]] = self.eos_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == self.max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(
                next(self.parameters()).device)

        # # generate target batch
        # decoded = generated.new(
        #     batch_size, sent_lengths.max().item()).fill_(self.pad_id)
        # for i, hypo in enumerate(best):
        #     decoded[i, :sent_lengths[i]] = hypo
        #     decoded[i, sent_lengths[i]] = self.eos_id

        # sanity check
        # num_eos = (decoded == self.eos_id).sum()
        # assert num_eos == batch_size or num_eos == 0

        return decoded  # generated[:, :cur_len], logits  # [:, :cur_len-1, :]

    def enforce_repetition_penalty(self, next_token_logits, batch_size, num_beams, generated):
        for i in range(batch_size * num_beams):
            for previous_token in set(generated[i].tolist()):
                # if score < 0 then repetition penalty has to be
                # multiplied to reduce the previous token probability
                if next_token_logits[i, previous_token] < 0:
                    next_token_logits[i,
                                      previous_token] *= self.repetition_penalty
                else:
                    next_token_logits[i,
                                      previous_token] /= self.repetition_penalty

    def unpack_batch(self, batch):
        dm, is_entity, pos, node_labels, text = batch

        text_pad_mask = (text == self.pad_id).bool()
        node_pad_mask = (node_labels == self.pad_id).bool()

        return dm, is_entity, pos, node_labels, node_pad_mask,\
            text, text_pad_mask

    def on_epoch_start(self):
        if not self.hparams.curriculum and not self.hparams.no_buckets\
           and not self.hparams.token_batching:
            if self.trainsampler is not None:
                self.trainsampler.on_epoch_start(self.current_epoch)

    def compute_loss(self, logits, text, coverage_term):
        flat_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        flat_targets = text[:, 1:].contiguous().view(-1)

        if self.hparams.tldr:
            flat_probs = torch.gather(
                F.softmax(flat_logits, dim=1),
                1,
                flat_targets.unsqueeze(1)
            ).squeeze(1)
            tl_ce = self.loss(flat_logits, flat_targets, average=False)
            tl_cosw = torch.cos(flat_probs * math.pi) + 1
            ce_loss = torch.sum(tl_cosw * tl_ce)  # before: mean
        else:
            ce_loss = self.loss(flat_logits, flat_targets, average=False).sum()

        # token normalization
        num_tokens = flat_targets.ne(self.pad_id).sum()
        ce_loss /= num_tokens

        cov_loss = self.coverage_weight * coverage_term
        if self.graph_pos_embed is None or self.hparams.sinusoidals:
            gpos_reg = ce_loss.new_zeros(1)
        else:
            gpos_reg = self.gpos_regularizer * self.graph_pos_embed.gpos_loss()

        if not ce_loss.requires_grad and self.training:
            print("CE LOSS DOES NOT REQUIRE GRADIENTS !!!!!")
        if not cov_loss.requires_grad and self.training:
            print("COVERAGE LOSS DOES NOT REQUIRE GRADIENTS !!!!!")
        if not gpos_reg.requires_grad and self.training and\
           not self.no_gpos and self.graph_pos_embed is not None\
           and not self.hparams.sinusoidals:
            print("GPOS REGULARIZER DOES NOT REQUIRE GRADIENTS !!!!!")

        total_loss = ce_loss + cov_loss + gpos_reg
        return total_loss, cov_loss, gpos_reg

    def training_step(self, batch, batch_nb):
        dm, is_entity, pos, node_labels, nl_pad_mask,\
            text, text_pad_mask = self.unpack_batch(batch)

        next_token_logits, coverage_term = self.forward(
            dm, is_entity, pos, node_labels, nl_pad_mask,
            text, text_pad_mask
        )

        if not next_token_logits.requires_grad:
            print("NEXT_TOKEN_LOGITS DO NOT REQUIRE GRADIENTS !!!!!")
        if not coverage_term.requires_grad:
            print("COVERAGE TERM DOES NOT REQUIRE GRADIENTS !!!!!")

        loss, cov_loss, gpos_reg = self.compute_loss(
            next_token_logits, text, coverage_term)

        # tensorboard_logs = {'train_loss': loss, 'coverage_loss': cov_loss,
        #                     'gpos_regularization': gpos_reg}
        # if self.hparams.curriculum:
        #     tensorboard_logs['competence'] = self.trainsampler.competence

        self.log('train_loss', loss)
        self.log('coverage_loss', cov_loss)
        self.log('gpos_regularization', gpos_reg)
        if self.hparams.curriculum:
            self.log('competence', self.trainsampler.competence)

        return loss

    def validation_step(self, batch, batch_nb):
        if self.no_validation:
            return None

        res = {}

        dm, is_entity, pos, node_labels, nl_pad_mask,\
            text, text_pad_mask = self.unpack_batch(batch)

        if not self.light_validation:
            generated_text = self.sample_sequence(
                dm, is_entity, pos, node_labels, nl_pad_mask
            )

            res['hypo_batch'] = generated_text.detach().cpu().tolist()
            res['ref_batch'] = text.detach().cpu().tolist()

        next_token_logits, coverage_term = self.forward(
            dm, is_entity, pos, node_labels, nl_pad_mask,
            text[:, 0], text_pad_mask[:, 0]
        )
        loss, cov_loss, gpos_reg = self.compute_loss(
            next_token_logits, text[:, 0], coverage_term)

        res['val_loss'] = loss - gpos_reg
        res['coverage_loss'] = cov_loss

        return res

    def test_step(self, batch, batch_nb):
        dm, is_entity, pos, node_labels, nl_pad_mask,\
            text, text_pad_mask = self.unpack_batch(batch)

        generated_text = self.sample_sequence(
            dm, is_entity, pos, node_labels, nl_pad_mask
        )

        generated = generated_text.detach().cpu().tolist()
        reference = text.detach().cpu().tolist()

        res = {'sequences': zip(generated, reference)}

        return res

    def get_proc_rank(self):
        try:
            proc_rank = self.trainer.proc_rank
        except Exception:
            proc_rank = 0
        return proc_rank

    def load_references_from_file(self, filename: str) -> List[List[str]]:
        res = []
        with open(filename) as f:
            for line in f:
                refs = line.rstrip().split('*#')
                res.append(refs)
        return res

    def test_epoch_end(self, outputs):
        if self.test_output_fn is None:
            fp = None
        else:
            proc_rank = self.get_proc_rank()
            fp = open(self.test_output_fn + '-{}'.format(proc_rank), 'w')

        sp = self.load_spm_model()

        hypo_collection, reference_collection = [], []
        loss_collection, cov_loss_collection = [], []
        for batch in outputs:
            for gen, ref in batch['sequences']:
                decoded_gen = sp.DecodeIds(gen)
                decoded_refs = [sp.DecodeIds(r) for r in ref]
                # filter empty strings
                filtered_refs = [r for r in decoded_refs if r]
                hypo_collection.append(decoded_gen)
                reference_collection.append(filtered_refs)
                if fp is not None:
                    self.print(decoded_gen, file=fp)  # , end='\t')
                    # print(*filtered_refs, sep='*#', file=fp)

            loss_collection.append(batch.get('loss', -torch.ones(1)))
            cov_loss_collection.append(
                batch.get('coverage_loss', -torch.ones(1)))

        if fp is not None:
            fp.close()

        avg_loss = torch.stack(loss_collection).mean()
        avg_cov_loss = torch.stack(cov_loss_collection).mean()
        if self.distributed:
            dist.all_reduce(avg_loss)
            avg_loss /= dist.get_world_size()
            dist.all_reduce(avg_cov_loss)
            avg_cov_loss /= dist.get_world_size()

        reference_collection = self.load_references_from_file(
            self.test_ref_file)

        reformatted_refs = self.reformat_ref_lists(reference_collection)
        scores = self.nlgeval.compute_metrics(
            reformatted_refs, hypo_collection)

        scores['loss'] = avg_loss
        scores['coverage_loss'] = avg_cov_loss

        self.test_scores = scores
        self.log_dict(scores)
        return scores

    def validation_epoch_end(self, outputs):
        if self.no_validation:
            return

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_cov_loss = torch.stack([x['coverage_loss']
                                    for x in outputs]).mean()
        if self.distributed:
            dist.all_reduce(avg_loss)
            avg_loss /= dist.get_world_size()
            dist.all_reduce(avg_cov_loss)
            avg_cov_loss /= dist.get_world_size()

        if not self.light_validation:
            sp = self.load_spm_model()
            hypo: List[str] = []
            refs: List[List[str]] = []
            for x in outputs:
                hypo.extend([sp.DecodeIds(h) for h in x['hypo_batch']])
                new_refs = [
                    [sp.DecodeIds(single_ref) for single_ref in r]
                    for r in x['ref_batch']
                ]
                # filter empty refs (caused by padding, corrupts the scores)
                refs.extend([r for r in ref_list if r]
                            for ref_list in new_refs)

            # make sure this is no test
            if len(refs) > 400:
                refs = self.load_references_from_file(self.val_ref_file)

            # reformat ref list
            reformatted_refs: List[List[str]] = self.reformat_ref_lists(refs)
            scores = self.nlgeval.compute_metrics(reformatted_refs, hypo)

            bleu = torch.tensor([scores['Bleu_4']]).cuda(avg_loss.device)
            if self.distributed:
                dist.all_reduce(bleu)
                bleu /= dist.get_world_size()

        if self.verbose and not self.light_validation:
            if not self.distributed or self.trainer.proc_rank == 0:
                sample = random.choice(outputs)

                generated = sample['hypo_batch'][0]
                ref = sample['ref_batch'][0][0]

                print('Example generation:')
                print(sp.DecodeIds(generated))
                print('Reference:')
                print(sp.DecodeIds(ref))

        self.log('val_loss', avg_loss)
        self.log('val_cov_loss', avg_cov_loss)

        if not self.light_validation:
            self.log('bleu', bleu)

    def reformat_ref_lists(self, refs) -> List[List[str]]:
        max_num_refs = max([len(ref_list) for ref_list in refs])
        reformatted_refs: List[List[str]] = []
        for _ in range(max_num_refs):
            reformatted_refs.append([])
        for ref_list in refs:
            for ref_bucket, ref_str in enumerate(ref_list):
                reformatted_refs[ref_bucket].append(ref_str)
            for i in range(max_num_refs - len(ref_list)):
                reformatted_refs[len(ref_list) + i].append(ref_list[0])
        return reformatted_refs

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'adafactor':
            # NO start learning rate
            opt = Adafactor(self.parameters(),
                            weight_decay=self.l2_regularizer)
            return opt
        elif self.hparams.optimizer_name == 'adam':
            opt = AdamW(self.parameters(), lr=self.hparams.learning_rate,
                        weight_decay=self.l2_regularizer)
            scheduler = get_linear_schedule_with_warmup(
                opt, self.lr_warmup_steps, self.num_training_steps)

            return [opt], [scheduler]
        else:
            opt = torch.optim.SGD(
                self.parameters(), self.hparams.learning_rate,
                weight_decay=self.l2_regularizer
            )
            return opt

    def determine_dataset(self, *data, is_test=False):
        try:
            word_dropout_p = self.hparams.word_dropout
        except AttributeError:
            word_dropout_p = 0.0

        if self.use_webNLG:
            if hasattr(self.hparams, 'bpe_dropout') and self.hparams.bpe_dropout > 0.0:
                dataset = WebNLG_BPEDropout(
                    data[0], self.spm_model_path, for_testing=is_test,
                    bpe_dropout=self.hparams.bpe_dropout
                )
            else:
                dataset = WebNLG(
                    *data, for_testing=is_test,
                    word_dropout=word_dropout_p, unk_id=self.unk_id,
                    special_ids=[self.pad_id, self.bos_id, self.eos_id]
                )
        else:
            dataset = Agenda(
                *data, title_graph=True, for_testing=is_test,
                word_dropout=word_dropout_p, unk_id=self.unk_id,
                special_ids=[self.pad_id, self.bos_id, self.eos_id]
            )

        return dataset, collate_webnlg

    def train_dataloader(self):
        trainset, collate = self.determine_dataset(*self.train_data)

        if self.hparams.curriculum:
            fct = self.hparams.full_competence_time
            if fct < 0:
                fct = self.hparams.num_epochs

            if self.hparams.batches_per_epoch > 0:
                fct *= self.hparams.batches_per_epoch
            elif self.hparams.token_batching:
                # TODO if time: change to non-hard-coded heuristic
                heuristic_batches_per_epoch = (
                    len(trainset) * 32) // self.batch_size
                fct *= heuristic_batches_per_epoch
                print(('Using {} as heuristic number of batches per epoch'
                       + ' to determine full competence time.'
                       ).format(heuristic_batches_per_epoch))
            else:
                fct *= len(trainset) // self.batch_size

            if self.hparams.token_batching:
                batch_sampler = CurriculumTokenBatchSampler(
                    trainset, self.batch_size,
                    trainset.num_tokens_fun,
                    start_competence=self.hparams.start_competence,
                    full_competence_time=fct,
                    distributed=self.distributed,
                    batches_per_epoch=self.hparams.batches_per_epoch
                )
            else:
                batch_sampler = CurriculumBatchSampler(
                    trainset, self.batch_size,
                    start_competence=self.hparams.start_competence,
                    full_competence_time=fct,
                    extract_fun=trainset.extract_fun,
                    num_pretraining_steps=self.hparams.num_lm_epochs *
                    len(trainset) // self.batch_size,
                    distributed=self.distributed
                )
        elif self.hparams.token_batching:
            batch_sampler = TokenBatchSampler(
                trainset, self.batch_size,
                trainset.num_tokens_fun, shuffle=self.shuffle_data,
                distributed=self.distributed,
                batches_per_epoch=self.hparams.batches_per_epoch
            )
        elif self.hparams.no_buckets:
            if self.distributed:
                sampler = DistributedSampler(
                    trainset, shuffle=self.shuffle_data)
            else:
                if self.shuffle_data:
                    sampler = RandomSampler(trainset, replacement=False)
                else:
                    sampler = SequentialSampler(trainset)
            batch_sampler = BatchSampler(sampler, self.batch_size, False)
        else:
            batch_sampler = BucketBatchSampler(
                trainset,
                [self.batch_size] * 4,
                shuffle=[self.shuffle_data] * 4,
                ratios=[0.4, 0.4, 0.1],
                key_fun=lambda x: len(trainset.extract_fun(x)),
                distributed=self.distributed,
            )

        self.trainsampler = batch_sampler

        return DataLoader(
            trainset, batch_sampler=batch_sampler,
            collate_fn=partial(
                collate, pad_elem=self.pad_id
            ), pin_memory=True,
            num_workers=2
        )

    def set_inference_batch_size(self, bs):
        self.inference_batch_size = bs

    def determine_inference_batch_size(self):
        if hasattr(self, 'inference_batch_size') and self.inference_batch_size is not None:
            batch_size = self.inference_batch_size
        elif self.use_webNLG:
            batch_size = 32
        else:
            batch_size = 8
        return max(1, batch_size // self.num_beams)

    def val_dataloader(self):
        valset, collate = self.determine_dataset(*self.val_data, is_test=True)

        if self.distributed:
            sampler = DistributedSampler(valset, shuffle=False)
        else:
            sampler = SequentialSampler(valset)

        batch_size = self.determine_inference_batch_size()

        return DataLoader(
            valset, batch_size=batch_size,
            collate_fn=partial(collate, pad_elem=self.pad_id),
            pin_memory=True, sampler=sampler,
            num_workers=2
        )

    def test_dataloader(self):
        testset, collate = self.determine_dataset(
            *self.test_data, is_test=True)

        if self.distributed:
            sampler = DistributedSampler(testset, shuffle=False)
        else:
            sampler = SequentialSampler(testset)

        batch_size = self.determine_inference_batch_size()

        return DataLoader(
            testset, batch_size=batch_size,
            collate_fn=partial(collate, pad_elem=self.pad_id),
            pin_memory=True, sampler=sampler,
            num_workers=2
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=False,
            conflict_handler='resolve'
        )

        parser.add_argument('data_root')

        parser.add_argument('--batches-per-epoch', type=int, default=0)

        parser.add_argument('--hidden-dim', type=int, default=512)
        parser.add_argument('--num-heads', type=int, default=8)
        parser.add_argument('--num-encoder-layers', type=int, default=6)
        parser.add_argument('--num-title-encoder-layers', type=int, default=3)
        parser.add_argument('--num-decoder-layers', type=int, default=6)
        parser.add_argument('--dim-feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=float, default=0.1)  # 0.3
        parser.add_argument('--attention-dropout', type=float, default=0.1)
        parser.add_argument('--input-dropout', type=float, default=0.0)
        parser.add_argument('--word-dropout', type=float, default=0.0)
        parser.add_argument('--bpe-dropout', type=float, default=0.0)
        parser.add_argument('--max-text-range', type=int, default=10)  # 100
        parser.add_argument('--max-graph-range', type=int, default=10)  # 20
        parser.add_argument('--same-text-range', type=int, default=10)

        parser.add_argument('--label-smoothing', type=float, default=0.1)
        parser.add_argument('--activation', default="gelu")  # prelu

        parser.add_argument('--tie-weights', type=bool, default=True)
        parser.add_argument('--separate-pos-per-layer',
                            action='store_false', dest='share_pos_across_layers')
        parser.add_argument('--full-pos-embeddings', action='store_true')
        parser.add_argument('--sinusoidals', action='store_true')
        parser.add_argument('--gpos-static-extremes', action='store_true')
        parser.add_argument('--no-gpos', action='store_true')
        parser.add_argument('--prenorm', action='store_true')

        parser.add_argument('--webNLG', action='store_true')

        parser.add_argument('--with-gate', action='store_true')
        parser.add_argument('--scaled-interattention', action='store_true')
        parser.add_argument('--copynet', action='store_true')
        parser.add_argument('--kgsum', action='store_true')
        parser.add_argument('--no-copy', action='store_true')
        parser.add_argument('--tldr', action='store_true')

        parser.add_argument('--coverage-weight', type=float, default=0.0)
        parser.add_argument('--gpos-regularizer', type=float, default=0.0)
        parser.add_argument('--l2-regularizer', type=float, default=0.0)

        parser.add_argument('--num-lm-epochs', type=int, default=0)

        parser.add_argument('--start-competence', type=float, default=0.1)
        parser.add_argument('--full-competence-time', type=int, default=10)

        parser.add_argument('--repetition-penalty', type=float, default=1.0)
        parser.add_argument('--length-penalty', type=float, default=0.0)  # 0.2
        parser.add_argument('--coverage-penalty',
                            type=float, default=0.0)  # 0.2
        parser.add_argument('--beam-size', type=int, default=1)

        parser.add_argument('--temperature', type=float, default=0.0)
        parser.add_argument('--top-k', type=int, default=0)  # 10
        parser.add_argument('--top-p', type=float, default=1.0)  # 0.95

        # training params
        parser.add_argument('--optimizer-name', default='adafactor', type=str)
        parser.add_argument('--batch-size', type=int, default=8)  # 12 ?
        parser.add_argument('--learning-rate', type=float, default=3e-4)
        parser.add_argument('--lr-reduce-patience', type=int, default=3)
        parser.add_argument('--lr-warmup-steps', type=int, default=0)

        parser.add_argument('--no-shuffling',
                            action='store_false', dest='shuffle_data')
        parser.add_argument('--curriculum', action='store_true')
        parser.add_argument('--no-buckets', action='store_true')
        parser.add_argument('--token-batching', action='store_true')

        return parser
