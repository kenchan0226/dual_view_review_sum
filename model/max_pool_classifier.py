import torch
import torch.nn as nn


class MaxPoolClassifier(nn.Module):

    def __init__(self, memory_bank_size, num_classes, dropout=0.0):
        super(MaxPoolClassifier, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.num_classes = num_classes
        self.classifier = nn.Sequential(nn.Linear(memory_bank_size, memory_bank_size),
                                        nn.Dropout(p=dropout),
                                        nn.ReLU(),
                                        nn.Linear(memory_bank_size, num_classes),
                                        nn.LogSoftmax(dim=1))

    def forward(self, encoder_hidden_states):
        """
        :param encoder_hidden_states: [batch, src_len, memory_bank_size]
        :return:
        """
        batch_size = encoder_hidden_states.size(0)
        r, _ = torch.max(encoder_hidden_states, dim=1)  # [batch, memory_bank_size]
        assert r.size() == torch.Size([batch_size, self.memory_bank_size])
        # 2 layer fc, relu
        logit = self.classifier(r)  # [batch, num_classes]
        return logit

