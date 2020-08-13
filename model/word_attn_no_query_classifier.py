import torch
import torch.nn as nn
from torch.nn import init
from model.attention import Attention
from utils.masked_softmax import MaskedSoftmax

class WordAttnNoQueryClassifier(nn.Module):

    def __init__(self, memory_bank_size, num_classes, dropout=0.0, ordinal=False, hr_enc=False):
        super(WordAttnNoQueryClassifier, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.num_classes = num_classes
        self.hr_enc = hr_enc

        self.self_attention_layer = nn.Linear(memory_bank_size, 1)
        if self.hr_enc:
            self.sent_self_attention_layer = nn.Linear(memory_bank_size, 1)
        self.softmax = MaskedSoftmax(dim=1)

        self.ordinal = ordinal
        self.expanded_memory_size = memory_bank_size if not hr_enc else 2 * memory_bank_size
        if ordinal:
            self.classifier = nn.Sequential(nn.Linear(self.expanded_memory_size, self.expanded_memory_size),
                                            nn.Dropout(p=dropout),
                                            nn.ReLU(),
                                            nn.Linear(self.expanded_memory_size, num_classes),
                                            nn.Sigmoid())
        else:
            self.classifier = nn.Sequential(nn.Linear(self.expanded_memory_size, self.expanded_memory_size),
                                            nn.Dropout(p=dropout),
                                            nn.ReLU(),
                                            nn.Linear(self.expanded_memory_size, num_classes),
                                            nn.LogSoftmax(dim=1))

    def self_attention(self, memory_bank, src_mask, attn_grad='word'):
        memory_bank = memory_bank.contiguous()
        if attn_grad == 'word':
            attention_scores = self.self_attention_layer(memory_bank)  # [batch, src_len, 1]
        elif attn_grad == 'sent':
            attention_scores = self.sent_self_attention_layer(memory_bank) # [batch, sent_num, 1]
        else:
            raise ValueError
        attention_dist = self.softmax(attention_scores.squeeze(2), src_mask)  # [batch, src_len] or [batch, sent_num]
        attention_dist = attention_dist.unsqueeze(1)  # [batch, 1, src_len]
        context = torch.bmm(attention_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size,  memory_bank_size]
        attention_dist = attention_dist.squeeze(1)  # [batch, src_len] or [batch, sent_num]
        return context, attention_dist

    def forward(self, encoder_memory_bank, src_mask, sent_memory_bank=None, sent_mask=None):
        """
        :param encoder_memory_bank: [batch, src_len, memory_bank_size]
        :param sent_memory_bank: [batch, sent_num, memory_bank_size]
        :return:
        """
        batch_size = encoder_memory_bank.size(0)
        context, attn_dist = self.self_attention(encoder_memory_bank, src_mask)

        attn_dist_tuple = (attn_dist, None)
        if self.hr_enc:
            sent_context, sent_attn_dist = self.self_attention(sent_memory_bank, sent_mask, attn_grad='sent')
            # [batch, 2 * memory_bank_size]
            context = torch.cat([context, sent_context], dim=1)
            attn_dist_tuple = (attn_dist, sent_attn_dist)

        logit = self.classifier(context)
        return logit, attn_dist_tuple
