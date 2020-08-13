import torch.nn as nn
from model.hss_seq2seq import HSSSeq2SeqModel
from model.hss_classifier import HSSClassifier


class HSSModel(nn.Module):
    # contains HSSSeq2Seq and HSSClassifier
    def __init__(self, opt):
        super(HSSModel, self).__init__()
        memory_bank_size = 2 * opt.encoder_size if opt.bidirectional else opt.encoder_size
        self.seq2seq_model = HSSSeq2SeqModel(opt)
        self.classifier_model = HSSClassifier(memory_bank_size, opt.num_classes, opt.classifier_dropout, opt.ordinal)
        self.model_type = opt.model_type

    def set_embedding(self, embedding):
        self.seq2seq_model.set_embedding(embedding)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, rating, src_sent_positions, src_sent_nums, src_sent_mask):
        decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, sentiment_context, encoder_memory_bank = \
            self.seq2seq_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, src_sent_positions, src_sent_nums, src_sent_mask)
        # encoder_memory_bank: [batch, src_len, 2 * encoder_size]
        # sentiment_context: [batch, trg_len, 2 * encoder_size]

        # forward classification model
        # 1. mask the memory bank vector of each padded src token as -inf
        # [batch, src_len, 1]
        expand_src_mask = src_mask.unsqueeze(-1)
        adding_src_mask = (1 - expand_src_mask).masked_fill((1 - expand_src_mask).byte(), -float('inf'))
        encoder_memory_bank = encoder_memory_bank * expand_src_mask + adding_src_mask

        # 2. mask the sentiment context vector of each padded trg token as -inf
        # [batch, trg_len, 1]
        expand_trg_mask = trg_mask.unsqueeze(-1)
        adding_trg_mask = (1 - expand_trg_mask).masked_fill((1 - expand_trg_mask).byte(), -float('inf'))
        sentiment_context = sentiment_context * expand_trg_mask + adding_trg_mask

        logit = self.classifier_model(encoder_memory_bank, sentiment_context)
        classifier_attention_dist = None
        return decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, logit, classifier_attention_dist
