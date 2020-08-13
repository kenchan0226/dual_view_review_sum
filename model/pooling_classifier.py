import torch
import torch.nn as nn


class MaxPoolClassifier(nn.Module):

    def __init__(self, memory_bank_size, num_classes, dropout=0.0, ordinal=False, hr_enc=False):
        super(MaxPoolClassifier, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.num_classes = num_classes
        self.ordinal = ordinal
        self.hr_enc = hr_enc
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

    def forward(self, encoder_hidden_states, src_mask, sent_enc_h_states=None, sent_mask=None):
        """
        :param encoder_hidden_states: [batch, src_len, memory_bank_size]
        :param sent_enc_h_states: [batch, sent_num, memory_bank_size]
        :return:
        """
        # 1. mask the word memory bank vector of each padded src token as -inf
        # [batch, src_len, 1]
        expand_src_mask = src_mask.unsqueeze(-1)
        adding_src_mask = (1 - expand_src_mask).masked_fill((1 - expand_src_mask).byte(), -float('inf'))
        encoder_hidden_states = encoder_hidden_states * expand_src_mask + adding_src_mask
        # 2. max pooling
        batch_size = encoder_hidden_states.size(0)
        word_r, _ = torch.max(encoder_hidden_states, dim=1)  # [batch, memory_bank_size]
        assert word_r.size() == torch.Size([batch_size, self.memory_bank_size])

        if self.hr_enc:
            # 3. if hr_enc, mask the sent memory bank vector of each padded sent as -inf
            # [batch, sent_num, 1]
            expand_sent_mask = sent_mask.unsqueeze(-1)
            adding_sent_mask = (1 - expand_sent_mask).masked_fill((1 - expand_sent_mask).byte(), -float('inf'))
            sent_enc_h_states = sent_enc_h_states * expand_sent_mask + adding_sent_mask
            # 2. max pooling
            batch_size = sent_enc_h_states.size(0)
            sent_r, _ = torch.max(sent_enc_h_states, dim=1)  # [batch, memory_bank_size]
            assert sent_r.size() == torch.Size([batch_size, self.memory_bank_size])
            # [batch, 2 * memory_bank_size]
            r = torch.cat([word_r, sent_r], dim=1)
        else:
            # [batch, 2 * memory_bank_size]
            r = word_r
        # 2 layer fc, relu
        logit = self.classifier(r)  # [batch, num_classes]
        return logit


class MeanPoolClassifier(nn.Module):

    def __init__(self, memory_bank_size, num_classes, dropout=0.0):
        super(MeanPoolClassifier, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.num_classes = num_classes
        self.classifier = nn.Sequential(nn.Linear(memory_bank_size, memory_bank_size),
                                        nn.Dropout(p=dropout),
                                        nn.ReLU(),
                                        nn.Linear(memory_bank_size, num_classes),
                                        nn.LogSoftmax(dim=1))

    def forward(self, encoder_hidden_states, src_mask=None):
        """
        :param encoder_hidden_states: [batch, src_len, memory_bank_size]
        :return:
        """
        batch_size = encoder_hidden_states.size(0)
        r = torch.mean(encoder_hidden_states, dim=1)  # [batch, memory_bank_size]
        assert r.size() == torch.Size([batch_size, self.memory_bank_size])
        # 2 layer fc, relu
        logit = self.classifier(r)  # [batch, num_classes]
        return logit

