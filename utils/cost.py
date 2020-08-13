import numpy as np
from utils.string_helper import _make_n_gram
from collections import Counter
import torch


def compute_batch_cost(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, cost_types, device):
    with torch.no_grad():
        #num_cost_types = len(cost_types)
        batch_cost_list = []
        #batch_cost_2d = np.zeros((batch_size, num_cost_types))
        for i, cost_type in enumerate(cost_types):
            if cost_type == 0:
                cost_func = has_three_gram_repeat
            elif cost_type == 1:
                cost_func = min_len_cost
            else:
                raise ValueError("No matched cost function type.")
            batch_cost_list.append(cost_func(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device))
            #batch_cost_2d[:, i] = cost_func(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size)

        batch_cost_2d = torch.stack(batch_cost_list, dim=1)  # tensor: [batch, num_cost_types]
    return batch_cost_2d


def num_three_gram_repeat(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    # return the number of 3-gram repeat for each prediciton sequence in the batch
    batch_cost = np.zeros(batch_size)
    for batch_i, pred_str in enumerate(pred_str_list):
        three_gram_counter = Counter()
        three_gram_counter.update(_make_n_gram(pred_str, n=3))
        three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
        batch_cost[batch_i] = three_gram_repeat
    return torch.from_numpy(batch_cost).type(torch.FloatTensor).to(device)  # tensor: [batch_size]


def has_three_gram_repeat(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    # return the number of 3-gram repeat for each prediciton sequence in the batch
    #batch_cost = np.zeros(batch_size)
    batch_cost = []
    for batch_i, pred_str in enumerate(pred_str_list):
        three_gram_counter = Counter()
        three_gram_counter.update(_make_n_gram(pred_str, n=3))
        three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
        if three_gram_repeat > 0:
            cost = 1.0
        else:
            cost = 0.0
        batch_cost.append(cost)
    return torch.FloatTensor(batch_cost).to(device)  # tensor: [batch_size]


def min_len_cost(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    batch_cost = []
    for batch_i, pred_str in enumerate(pred_str_list):
        if len(pred_str) < 20:
            cost = 1.0
        else:
            cost = 0.0
        batch_cost.append(cost)
    return torch.FloatTensor(batch_cost).to(device)  # tensor: [batch_size]

