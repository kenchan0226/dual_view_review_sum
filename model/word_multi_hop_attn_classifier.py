import torch
import torch.nn as nn
from torch.nn import init
from model.attention import Attention

class WordMultiHopAttnClassifier(nn.Module):

    def __init__(self, query_hidden_size, memory_bank_size, num_classes, attn_mode, dropout=0.0, ordinal=False, hr_enc=False):
        super(WordMultiHopAttnClassifier, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.query_hidden_size = query_hidden_size
        self.num_classes = num_classes
        self.hr_enc = hr_enc

        self._query_vector = nn.Parameter(torch.zeros(1, query_hidden_size))
        init.uniform_(self._query_vector, -0.1, 0.1)
        self.attention_layer = Attention(memory_bank_size, memory_bank_size, coverage_attn=False, attn_mode=attn_mode)
        self.glimpse_layer = Attention(query_hidden_size, memory_bank_size, coverage_attn=False, attn_mode=attn_mode)
        if self.hr_enc:
            self._sent_query_vector = nn.Parameter(torch.zeros(1, query_hidden_size))
            init.uniform_(self._sent_query_vector, -0.1, 0.1)
            self.sent_attention_layer = Attention(memory_bank_size, memory_bank_size, coverage_attn=False, attn_mode=attn_mode)
            self.sent_glimpse_layer = Attention(query_hidden_size, memory_bank_size, coverage_attn=False, attn_mode=attn_mode)

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

    def forward(self, encoder_memory_bank, src_mask, sent_memory_bank=None, sent_mask=None):
        """
        :param encoder_hidden_states: [batch, src_len, memory_bank_size]
        :param sent_memory_bank: [batch, sent_num, memory_bank_size]
        :return:
        """
        batch_size = encoder_memory_bank.size(0)
        query_vector_expanded = self._query_vector.expand(batch_size, self.query_hidden_size)  # [batch, query_hidden_size]
        glimpse_vector, _, _ = self.glimpse_layer(query_vector_expanded, encoder_memory_bank, src_mask)  # [batch, memory_bank_size]
        context, attn_dist, _ = self.attention_layer(glimpse_vector, encoder_memory_bank, src_mask)

        attn_dist_tuple = (attn_dist, None)
        if self.hr_enc:
            sent_query_vector_expanded = self._sent_query_vector.expand(batch_size, self.query_hidden_size)  # [batch, query_hidden_size]
            sent_glimpse_vector, _, _ = self.sent_glimpse_layer(sent_query_vector_expanded, sent_memory_bank, sent_mask)  # [batch, memory_bank_size]
            sent_context, sent_attn_dist, _ = self.sent_attention_layer(sent_glimpse_vector, sent_memory_bank, sent_mask)
            # [batch, 2 * memory_bank_size]
            context = torch.cat([context, sent_context], dim=1)
            attn_dist_tuple = (attn_dist, sent_attn_dist)

        logit = self.classifier(context)
        return logit, attn_dist_tuple

