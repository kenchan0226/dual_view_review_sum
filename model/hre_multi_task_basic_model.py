import torch.nn as nn
from model.seq2seq import HirEncSeq2SeqModel
from model.pooling_classifier import MaxPoolClassifier, MeanPoolClassifier


class HirEncMultiTaskBasicModel(nn.Module):
    # contains HSSSeq2Seq and HSSClassifier
    def __init__(self, opt):
        super(HirEncMultiTaskBasicModel, self).__init__()
        memory_bank_size = 2 * opt.encoder_size if opt.bidirectional else opt.encoder_size
        self.seq2seq_model = HirEncSeq2SeqModel(opt)
        if opt.model_type == "max" or opt.model_type == "hre_max":
            self.classifier_model = MaxPoolClassifier(memory_bank_size, opt.num_classes, opt.classifier_dropout)
        elif opt.model_type == "mean":
            self.classifier_model = MeanPoolClassifier(memory_bank_size, opt.num_classes, opt.classifier_dropout)
        else:
            raise ValueError("invalid type")
        self.model_type = opt.model_type

    def set_embedding(self, embedding):
        self.seq2seq_model.set_embedding(embedding)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums, src_sent_mask):
        decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, encoder_memory_bank = \
            self.seq2seq_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, src_sent_positions, src_sent_nums, src_sent_mask)
        # encoder_memory_bank: [batch, src_len, 2 * encoder_size]
        # forward classification model
        # # 1. mask the memory bank vector of each padded src token as -inf
        # # [batch, src_len, 1]
        # if isinstance(self.classifier_model, MaxPoolClassifier):
        #     expand_src_mask = src_mask.unsqueeze(-1)
        #     adding_src_mask = (1 - expand_src_mask).masked_fill((1 - expand_src_mask).byte(), -float('inf'))
        #     encoder_memory_bank = encoder_memory_bank * expand_src_mask + adding_src_mask

        logit = self.classifier_model(encoder_memory_bank, src_mask)
        classifier_attention_dist = None
        return decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, logit, classifier_attention_dist

