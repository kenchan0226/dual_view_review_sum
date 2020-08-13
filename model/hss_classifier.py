import torch
import torch.nn as nn


class HSSClassifier(nn.Module):

    def __init__(self, memory_bank_size, num_classes, dropout=0.0, ordinal=False):
        super(HSSClassifier, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.num_classes = num_classes
        self.ordinal = ordinal
        if ordinal:
            self.classifier = nn.Sequential(nn.Linear(memory_bank_size, memory_bank_size),
                                            nn.Dropout(p=dropout),
                                            nn.ReLU(),
                                            nn.Linear(memory_bank_size, num_classes),
                                            nn.Sigmoid())
        else:
            self.classifier = nn.Sequential(nn.Linear(memory_bank_size, memory_bank_size),
                                            nn.Dropout(p=dropout),
                                            nn.ReLU(),
                                            nn.Linear(memory_bank_size, num_classes),
                                            nn.LogSoftmax(dim=1))

    def forward(self, encoder_hidden_states, sentiment_context):
        """
        :param encoder_hidden_states: [batch, src_len, memory_bank_size]
        :param sentiment_context: [batch, trg_len, memory_bank_size]
        :return:
        """
        assert encoder_hidden_states.size(0) == sentiment_context.size(0)
        batch_size = encoder_hidden_states.size(0)
        concated_representation = torch.cat([sentiment_context, encoder_hidden_states], dim=1)
        r, _ = torch.max(concated_representation, dim=1)  # [batch, memory_bank_size]
        assert r.size() == torch.Size([batch_size, self.memory_bank_size])
        # 2 layer fc, relu
        logit = self.classifier(r)  # [batch, num_classes]
        return logit

