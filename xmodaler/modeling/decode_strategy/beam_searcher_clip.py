"""
From original at https://github.com/aimagelab/meshed-memory-transformer/blob/master/models/beam_search/beam_search.py
Original copyright of AImageLab code below, modifications by Yehao Li, Copyright 2021.
"""
# Copyright (c) 2019, AImageLab
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import expand_tensor
from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY
from xmodaler.functional import load_vocab
from xmodaler.tokenization import BertTokenizer

from ..clip import load as load_clip
from ..clip import tokenize as clip_tokenize
from xmodaler.functional import load_vocab, decode_sequence, decode_sequence_bert


@DECODE_STRATEGY_REGISTRY.register()
class BeamSearcherCLIP(DecodeStrategy):
    @configurable
    def __init__(
            self,
            *,
            vocab_path,
            beam_size,
            max_seq_len,
            bert_tokenizer,
            bos_token_id,
            eos_token_id,
            clip=None,
    ):
        super().__init__()
        self.beam_size = beam_size
        if bert_tokenizer is None:
            self.vocab = load_vocab(vocab_path)
        else:
            self.vocab = None

        self.max_seq_len = max_seq_len
        self.bert_tokenizer = bert_tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.clip_model = clip

    @classmethod
    def from_config(cls, cfg):
        tokenizer_map = {'BERT': BertTokenizer}

        tokenizer_cls = tokenizer_map.get(cfg.INFERENCE.VOCAB, None)
        if tokenizer_cls is None:
            bert_tokenizer = None
            bos_token_id = 0
            eos_token_id = 0
        else:
            bert_tokenizer = tokenizer_cls.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME, do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE)
            if cfg.INFERENCE.VOCAB == 'BERT':
                bos_token_id = bert_tokenizer.vocab["[CLS]"]
                eos_token_id = bert_tokenizer.vocab["[SEP]"]

        if cfg.DECODE_STRATEGY.USE_CLIP_PROB:
            clip_model = load_clip(cfg.DECODE_STRATEGY.CLIP_MODEL)
        else:
            clip_model = None

        return {
            "vocab_path": cfg.INFERENCE.VOCAB,
            "beam_size": cfg.DECODE_STRATEGY.BEAM_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            'bert_tokenizer': bert_tokenizer,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "clip": clip_model
        }

    def _select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def _expand_state(self, states, selected_beam, batch_size, beam_size, cur_beam_size):
        for i in range(len(states)):
            shape = list(states[i].shape)
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            states[i] = torch.gather(states[i].view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                beam.expand(*([batch_size, beam_size] + shape[1:])))
            states[i] = states[i].view(*([-1, ] + shape[1:]))

    def clip_text_prob(self, img_feature, pred_sents_ids):
        """
            img_feature: shape[batch, C]
            pred_sents_ids: list [bs, beam_size, 1]
        """
        pred_sents_ids = torch.cat(pred_sents_ids, -1)
        bs, beam_size, seq_length = pred_sents_ids.shape
        pred_sents_ids = pred_sents_ids.reshape(-1, seq_length)

        if self.vocab:
            sents = decode_sequence(self.vocab, pred_sents_ids)
        else:
            sents = decode_sequence_bert(self.bert_tokenizer, pred_sents_ids, self.eos_token_id)

        sent_embed = clip_tokenize(sents).to(self.device)
        text_feature = self.clip_model.encode_text(sent_embed)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature.reshape(bs, beam_size, -1).transpose(1, 2)

        if len(img_feature.shape) == 2:
            img_feature = img_feature.unsqueeze(1)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * torch.bmm(img_feature, text_feature).squeeze(1)
        prob_per_text = logits_per_image.softmax(dim=-1)
        return prob_per_text

    def _forward(self, batched_inputs, model):
        batch_size = batched_inputs[kfg.ATT_FEATS].size(0)
        out_size = batched_inputs.get('OUT_SIZE', 1)
        beam_size = self.beam_size
        log_probs = []
        selected_words = None
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id

        inputs = batched_inputs
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = model.visual_embed(batched_inputs)
        inputs.update(ve_out)

        encoder_out_v = model.encoder(inputs, mode='v')
        inputs.update(encoder_out_v)
        inputs = model.decoder.preprocess(inputs)

        clip_img_feats = self.clip_model.encode_image(batched_inputs['img'].to(self.device))
        clip_img_feats = clip_img_feats / clip_img_feats.norm(dim=-1, keepdim=True)

        outputs = []
        for t in range(self.max_seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            inputs.update({ kfg.G_TOKENS_IDS: wt, kfg.TIME_STEP: t })
            te_out = model.token_embed(inputs)
            inputs.update(te_out)

            encoder_out_t = model.encoder(inputs, mode='t')
            inputs.update(encoder_out_t)

            decoder_out = model.decoder(inputs)
            inputs.update(decoder_out)

            if t > 1:
                clip_prob = self.clip_text_prob(clip_img_feats, outputs)
            else:
                clip_prob = 0

            logit = model.predictor(inputs)[kfg.G_LOGITS]
            word_logprob = F.log_softmax(logit, dim=-1)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.eos_token_id).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self._select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
 
            if kfg.HISTORY_STATES in inputs:
                expand_keys = [kfg.HISTORY_STATES]
                if kfg.ENC_HISTORY_STATES in inputs:
                    expand_keys.append(kfg.ENC_HISTORY_STATES)
            else:
                expand_keys = [kfg.G_HIDDEN_STATES, kfg.G_CELL_STATES]

            for key in expand_keys:
                if key in inputs:
                    states = inputs[key]
                    self._expand_state(states, selected_beam, batch_size, beam_size, cur_beam_size)
                    inputs.update({key: states})

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                expand_keys = { 
                    kfg.ATT_FEATS, 
                    kfg.GLOBAL_FEATS, 
                    kfg.ATT_MASKS, 
                    kfg.EXT_ATT_MASKS, 
                    kfg.P_ATT_FEATS, 
                    kfg.EXT_G_TOKENS_MASKS,
                    kfg.G_TOKENS_TYPE,
                    kfg.SEMANTICS_FEATS,
                    kfg.EXT_SEMANTICS_MASKS
                }
                for key in expand_keys:
                    if key in inputs:
                        if isinstance(inputs[key], list):
                            inputs[key] = inputs[key][-1] # usually is ATT_FEATS in TDEN
                        tensor = expand_tensor(inputs[key], beam_size)
                        inputs.update({ key: tensor })

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.max_seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.max_seq_len))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)
        
        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: outputs,
            kfg.G_LOGP: log_probs
        }