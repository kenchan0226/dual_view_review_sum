import torch.nn as nn
from model.seq2seq import Seq2SeqModel
from model.pooling_classifier import MaxPoolClassifier, MeanPoolClassifier
from model.word_attn_classifier import WordAttnClassifier
from model.word_multi_hop_attn_classifier import WordMultiHopAttnClassifier
from model.word_attn_no_query_classifier import WordAttnNoQueryClassifier


class MultiTaskBasicModel(nn.Module):
    # contains HSSSeq2Seq and HSSClassifier
    def __init__(self, opt):
        super(MultiTaskBasicModel, self).__init__()
        memory_bank_size = 2 * opt.encoder_size if opt.bidirectional else opt.encoder_size
        self.seq2seq_model = Seq2SeqModel(opt)
        if opt.classifier_type == "max":
            self.classifier_model = MaxPoolClassifier(memory_bank_size, opt.num_classes, opt.classifier_dropout, opt.ordinal)
        elif opt.classifier_type == "word_attn":
            self.classifier_model = WordAttnClassifier(opt.query_hidden_size, memory_bank_size, opt.num_classes, opt.attn_mode, opt.classifier_dropout, opt.ordinal)
        elif opt.classifier_type == "word_attn_no_query":
            self.classifier_model = WordAttnNoQueryClassifier(memory_bank_size, opt.num_classes, opt.classifier_dropout,
                                                        opt.ordinal)
        elif opt.classifier_type == "word_multi_hop_attn":
            self.classifier_model = WordMultiHopAttnClassifier(opt.query_hidden_size, memory_bank_size, opt.num_classes,
                                                         opt.attn_mode, opt.classifier_dropout, opt.ordinal)
        else:
            raise ValueError
        self.model_type = opt.model_type
        self.classifier_type = opt.classifier_type

    def set_embedding(self, embedding):
        self.seq2seq_model.set_embedding(embedding)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, rating, src_sent_positions, src_sent_nums, src_sent_mask, rating_tokens_tensor):
        decoder_dist, h_t, seq2seq_attention_dist, encoder_final_states, coverage, encoder_memory_banks = \
            self.seq2seq_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, src_sent_positions, src_sent_nums, src_sent_mask)
        # encoder_memory_bank: [batch, src_len, 2 * encoder_size]
        # forward classification model
        # 1. mask the memory bank vector of each padded src token as -inf
        # [batch, src_len, 1]
        #if isinstance(self.classifier_model, MaxPoolClassifier):
        #    expand_src_mask = src_mask.unsqueeze(-1)
        #    adding_src_mask = (1 - expand_src_mask).masked_fill((1 - expand_src_mask).byte(), -float('inf'))
        #    encoder_memory_bank = encoder_memory_bank * expand_src_mask + adding_src_mask
        if self.seq2seq_model.separate_layer_enc:
            classifier_memory_bank = encoder_memory_banks[1]
        elif self.seq2seq_model.hr_enc:
            #(word_memory_bank, sent_memory_bank), (encoder_last_layer_final_state, sent_enc_ll_final_state)
            word_memory_bank, sent_memory_bank = encoder_memory_banks
            #classifier_memory_bank = sent_memory_bank
            classifier_memory_bank = word_memory_bank
        else:
            classifier_memory_bank = encoder_memory_banks[0]

        classifier_output = self.classifier_model(classifier_memory_bank, src_mask)
        if isinstance(classifier_output, tuple):
            logit = classifier_output[0]
            classify_attn_dist = classifier_output[1]
        else:
            logit = classifier_output
            classify_attn_dist = None

        return decoder_dist, h_t, seq2seq_attention_dist, encoder_final_states[0], coverage, logit, classify_attn_dist

