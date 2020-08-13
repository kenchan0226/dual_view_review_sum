import numpy as np
from utils.string_helper import *
import torch
from utils import io
from metric import compute_rouge_l_summ, compute_rouge_n
import nltk
from bert_score import score
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.


def sample_list_to_str_list(sample_list, oov_lists, idx2word, vocab_size, eos_idx, unk_idx=None, replace_unk=False, src_str_list=None):
    """Convert a list of sample dict to a 2d list of predicted keyphrases"""
    pred_str_list = []  #  a 2dlist, len(pred_str_2d_list)=batch_size, len(pred_str_2d_list[0])=
    for sample, oov, src_word_list in zip(sample_list, oov_lists, src_str_list):
        # sample['prediction']: list of 0-dim tensor, len=trg_len
        # sample['attention']: tensor with size [trg_len, src_len]
        pred_word_list = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, sample['attention'])
        pred_str_list.append(pred_word_list)
    return pred_str_list


def compute_batch_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, reward_type, regularization_factor=0.0, regularization_type=0, entropy=None, device='cpu'):
    with torch.no_grad():
        # A reward function returns a tensor of size [batch_size]
        if reward_type == 0:
            reward_func = xsum_mixed_rouge_reward
        elif reward_type == 1:
            reward_func = rouge_l_reward
        return reward_func(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device)


def rouge_l_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    #reward = np.zeros(batch_size)
    reward = []
    for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
            zip(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list)):
        #reward[idx] = compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f')
        reward.append(compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f'))
    return torch.FloatTensor(reward).to(device)  # tensor: [batch_size]


def xsum_mixed_rouge_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    #reward = np.zeros(batch_size)
    reward = []
    for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
            zip(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list)):
        #reward[idx] = 0.2 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.5 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.3 * compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f')
        reward.append(0.2 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.5 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.3 * compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f'))
    return torch.FloatTensor(reward).to(device)  # tensor: [batch_size]


def bert_score_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    P, R, F1 = score(' '.join(pred_str_list), ' '.join(trg_str_list))
    return F1.to(device)


def bert_score_reward_no_stop_word(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    stop_words = stopwords.words('english')
    # remove stop words
    pred_str_list = [w for w in pred_str_list if w not in stop_words]
    trg_str_list = [w for w in trg_str_list if w not in stop_words]
    P, R, F1 = score(' '.join(pred_str_list), ' '.join(trg_str_list))
    return F1.to(device)


def compute_batch_reward_old(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, reward_type, regularization_factor=0.0, regularization_type=0, entropy=None):
    reward = np.zeros(batch_size)

    if regularization_type == 2:
        if entropy is None:
            raise ValueError('Entropy should not be none when regularization type is 2')
        assert reward.shape[0] == entropy.shape[0]

    for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(zip(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list)):
        if entropy is None:
            entropy_idx = None
        else:
            entropy_idx = entropy[idx]
        reward[idx] = compute_reward_old(pred_str, pred_sent_list, trg_str, trg_sent_list, reward_type, regularization_factor, regularization_type, entropy_idx)
    return reward


def compute_reward_old(pred_str, pred_sent_list, trg_str, trg_sent_list, reward_type, regularization_factor=0.0, regularization_type=0, entropy=None):

    if regularization_type == 1:
        raise ValueError("Not implemented.")
    elif regularization_type == 2:
        regularization = entropy
    else:
        regularization = 0.0

    if reward_type == 0:
        tmp_reward = 0.3 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.2 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.5 * compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f')
    elif reward_type == 1:
        tmp_reward = compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f')
    else:
        raise ValueError

    # Add the regularization term to the reward only if regularization type != 0
    if regularization_type == 0 or regularization_factor == 0:
        reward = tmp_reward
    else:
        reward = (1 - regularization_factor) * tmp_reward + regularization_factor * regularization
    return reward


def compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, max_num_phrases, reward_shaping, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    phrase_reward = np.zeros((batch_size, max_num_phrases))
    if reward_shaping:
        for t in range(max_num_phrases):
            pred_str_2dlist_at_t = [pred_str_list[:t + 1] for pred_str_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_batch_reward(pred_str_2dlist_at_t, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
    else:
        phrase_reward[:, -1] = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type,
                                                               topk, match_type, regularization_factor, regularization_type, entropy)
    return phrase_reward


def compute_phrase_reward_backup(pred_str_2dlist, trg_str_2dlist, batch_size, num_predictions, reward_shaping, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    phrase_reward = np.zeros((batch_size, num_predictions))
    if reward_shaping:
        for t in range(num_predictions):
            pred_str_2dlist_at_t = [pred_str_list[:t + 1] for pred_str_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_batch_reward(pred_str_2dlist_at_t, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
    else:
        phrase_reward[:, num_predictions - 1] = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type,
                                                               topk, match_type, regularization_factor, regularization_type, entropy)
    return phrase_reward


def shape_reward(reward_np_array):
    batch_size, seq_len = reward_np_array.shape
    left_padding = np.zeros((batch_size, 1))
    left_padded_reward = np.concatenate([left_padding, reward_np_array], axis=1)
    return np.diff(left_padded_reward, n=1, axis=1)


def phrase_reward_to_stepwise_reward(phrase_reward, eos_idx_mask):
    batch_size, seq_len = eos_idx_mask.size()
    stepwise_reward = np.zeros((batch_size, seq_len))
    for i in range(batch_size):
        pred_cnt = 0
        for j in range(seq_len):
            if eos_idx_mask[i, j].item() == 1:
                stepwise_reward[i, j] = phrase_reward[i, pred_cnt]
                pred_cnt += 1
            #elif j == seq_len:
            #    pass
    return stepwise_reward


def compute_pg_loss(log_likelihood, output_mask, q_val_sample):
    """
    :param log_likelihood: [batch_size, prediction_seq_len]
    :param input_mask: [batch_size, prediction_seq_len]
    :param q_val_sample: [batch_size, prediction_seq_len]
    :return:
    """
    log_likelihood = log_likelihood.view(-1)  # [batch_size * prediction_seq_len]
    output_mask = output_mask.view(-1)  # [batch_size * prediction_seq_len]
    q_val_sample = q_val_sample.view(-1)  # [batch_size * prediction_seq_len]
    objective = -log_likelihood * output_mask * q_val_sample
    objective = torch.sum(objective)/torch.sum(output_mask)
    return objective


if __name__ == "__main__":
    #reward = np.array([[1,3,5,6],[2,3,5,9]])
    #print(shape_reward(reward))

    #pred_str_list = [['multi', 'agent', 'system'], ['agent', 'warning'], ['multi', 'agent'], ['agent'], ['agent', 'system'], ['multi', 'system'], ['what', 'is']]
    #trg_str_list = [['multi', 'agent', 'system'], ['multi'], ['what', 'is']]
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='exact'))
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='sub'))
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='exact', dimension=2))
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='sub', dimension=2))

    #r = np.array([2, 1, 2, 0])
    #print(ndcg_at_k(r, 4, method=1))  # 0.96519546960144276

    r_2d = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0, 0, 1, 1, 0, 1, 0, 0]])
    k_list = [1,2,3]
    print(alpha_ndcg_at_ks(r_2d, k_list))
    r_2d = r_2d[:, np.array([0, 4, 6, 1, 5, 2, 7, 8, 9])]
    print(alpha_ndcg_at_ks(r_2d, k_list))

    '''
    r = np.array([0,1,1,0,1,0])
    k_list = [4, 6]
    print(average_precision_at_ks(r, k_list, num_trgs=5, num_predictions=6))
    '''
    pass
