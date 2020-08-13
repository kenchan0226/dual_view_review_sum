# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of https://github.com/memray/seq2seq-keyphrase-pytorch and https://github.com/ChenRocks/fast_abs_rl
"""
import inspect
import json
import re
import numpy as np
import os
import logging
from os.path import join
from cytoolz import curry, concat, compose
from os.path import basename


import gensim

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from toolz.sandbox import unzip

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3


class BertSeqClassifyDataset(TensorDataset):
    def __init__(self, split: str, path: str) -> None:
        self._data_pt_path = join(path, "{}.pt".format(split))
        features = torch.load(self._data_pt_path)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        super().__init__(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def get_all_labels_list(self):
        return self.tensors[3].view(-1).tolist()


def convert_to_bert_tensor(input_list):
    # input_list: a list of list of words, should not include the EOS word.
    input_list_lens = [len(l) for l in input_list]
    max_seq_len = max(input_list_lens)
    padded_batch = PAD * np.ones((len(input_list), max_seq_len))
    #segment_ids = PAD * np.ones((len(input_list), max_seq_len))

    for j in range(len(input_list)):
        current_len = input_list_lens[j]
        padded_batch[j][:current_len] = input_list[j]
        #segment_ids[j][:current_len] = 1

    padded_batch = torch.LongTensor(padded_batch)
    return padded_batch, input_list_lens


class JsonDataset(Dataset):
    def __init__(self, split: str, path: str) -> None:
        #assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        return js


class JsonDatasetFromIdx(Dataset):
    def __init__(self, split: str, path: str, start_idx: int) -> None:
        #assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = count_data(self._data_path) - start_idx
        self.start_idx = start_idx

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i + self.start_idx))) as f:
            js = json.loads(f.read())
        return js


class SummRating(JsonDataset):
    def __init__(self, split, path):
        super().__init__(split, path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        return js_data['reviewText'], js_data['summary'], js_data['overall'] - 1  # so that the rating starts from 0


class DecodeDataset(JsonDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, path):
        assert 'train' not in split
        super().__init__(split, path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['reviewText']
        #article_string = ' '.join(js_data['article'])
        return art_sents


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


@curry
def tokenize(input_str, max_len=-1):
    input_tokenized = input_str.strip().lower().split(' ')
    if max_len > 0:
        input_tokenized = input_tokenized[:max_len]
    return input_tokenized


@curry
def eval_coll_fn(batch, word2idx, src_max_len=-1):
    # batch: a list of art_str
    # tokenize article string
    source_list_tokenized = []
    f_sent_position_list, b_sent_position_list = [], []
    src_sent_nums = []
    for src_sent_list in batch:
        if src_sent_list:
            f_sent_position, b_sent_position = [], []
            trunc_src_token_list = []
            for src_sent_i in src_sent_list:
                src_sent_i_tokens = src_sent_i.strip().split(' ')
                f_posi = f_sent_position[-1] + len(src_sent_i_tokens) if len(f_sent_position) != 0 else (
                            len(src_sent_i_tokens) - 1)
                b_posi = f_posi - len(src_sent_i_tokens) + 1
                # filter out the sentences which make the src too long
                if (f_posi + 1) > src_max_len > 0:
                    break
                trunc_src_token_list = trunc_src_token_list + src_sent_i_tokens
                f_sent_position.append(f_posi)
                b_sent_position.append(b_posi)

            # Note: the src will add an EOS token as the ending token when converting the src token list into a tensor.
            # Therefore, we add one to the f_sent_position[-1] here
            f_sent_position[-1] += 1

            f_sent_position_list.append(f_sent_position)
            b_sent_position_list.append(b_sent_position)
            source_list_tokenized.append(trunc_src_token_list)
            src_sent_nums.append(len(f_sent_position))

    batch_size = len(source_list_tokenized)
    # convert to idx
    source_list_indiced = []
    source_oov_list_indiced = []
    oov_lists = []
    for src in source_list_tokenized:
        src_oov, oov_dict, oov_list = extend_vocab_oov(src, word2idx)
        src = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in src]
        source_list_indiced.append(src)
        source_oov_list_indiced.append(src_oov)
        oov_lists.append(oov_list)
    original_indices = list(range(batch_size))
    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list_indiced, source_oov_list_indiced, oov_lists, source_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list_indiced, source_oov_list_indiced, oov_lists, source_list_tokenized, original_indices = zip(*seq_pairs)
    """
    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list_indiced)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list_indiced)
    source_mask = create_padding_mask(source_tensor)

    # pad the f_sent_position_list and b_sent_position_list with 0s
    max_sent_num = max(src_sent_nums)
    f_sent_position_list = [sent_posi + [0] * (max_sent_num - len(sent_posi)) for sent_posi in f_sent_position_list]
    # [batch, max_sent_num]
    f_sent_position_tensor = torch.LongTensor(f_sent_position_list)

    b_sent_position_list = [sent_posi + [0] * (max_sent_num - len(sent_posi)) for sent_posi in b_sent_position_list]
    # [batch, max_sent_num]
    b_sent_position_tensor = torch.LongTensor(b_sent_position_list)

    # [batch, max_sent_num, 2]
    sent_position_tensor = torch.stack([f_sent_position_tensor, b_sent_position_tensor], dim=2)
    # [batch, max_sent_num]
    src_sent_mask = create_sequence_mask(src_sent_nums)

    # changed by wchen to a dictionary output
    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['src_sent_positions'] = sent_position_tensor
    batch_dict['src_sent_nums'] = src_sent_nums
    batch_dict['src_sent_mask'] = src_sent_mask
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['original_indices'] = original_indices

    return batch_dict


@curry
def summ_rating_flatten_coll_fn(batch, word2idx, src_max_len=-1, trg_max_len=-1):
    # batch: a list of (art_str, abs_str)
    # Remove empty data, tokenize, and truncate
    source_list_tokenized, target_list_tokenized = [], []
    f_sent_position_list, b_sent_position_list = [], []
    src_sent_nums = []
    target_sent_2d_list = []
    rating_list = []
    for src_sent_list, trg_sent_list, rating in batch:
        if src_sent_list and trg_sent_list and rating != "":
            f_sent_position, b_sent_position = [], []
            trunc_src_token_list = []
            for src_sent_i in src_sent_list:
                src_sent_i_tokens = src_sent_i.strip().split(' ')
                f_posi = f_sent_position[-1] + len(src_sent_i_tokens) if len(f_sent_position) != 0 else (len(src_sent_i_tokens) - 1)
                b_posi = f_posi - len(src_sent_i_tokens) + 1
                # filter out the sentences which make the src too long
                if (f_posi + 1) > src_max_len > 0:
                    break
                trunc_src_token_list = trunc_src_token_list + src_sent_i_tokens
                f_sent_position.append(f_posi)
                b_sent_position.append(b_posi)

            # Note: the src will add an EOS token as the ending token when converting the src token list into a tensor.
            # Therefore, we add one to the f_sent_position[-1] here
            # # for debugging
            # if len(f_sent_position) == 0:
            #     print(src_sent_list)
            #     print(trg_sent_list)

            # if there is no valid sentences in src, then skip this data sample
            if len(f_sent_position) == 0:
                continue

            rating_list.append(rating)
            f_sent_position[-1] += 1

            f_sent_position_list.append(f_sent_position)
            b_sent_position_list.append(b_sent_position)
            source_list_tokenized.append(trunc_src_token_list)
            src_sent_nums.append(len(f_sent_position))

            target_sent_2d_list.append(trg_sent_list)
            # concat sent list into one str
            trg = ' '.join(trg_sent_list)
            # # tokenize and truncate
            # source_list_tokenized.append(tokenize(src, src_max_len))
            target_list_tokenized.append(tokenize(trg, trg_max_len))

    batch_size = len(source_list_tokenized)

    # convert to idx
    source_list, target_list, source_oov_list, target_oov_list, oov_lists = convert_batch_to_idx(source_list_tokenized, target_list_tokenized, word2idx)
    original_indices = list(range(batch_size))

    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices = zip(*seq_pairs)
    """

    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list)
    target_tensor, target_lens = convert_to_tensor(target_list)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list)
    target_oov_tensor, _ = convert_to_tensor(target_oov_list)
    source_mask = create_padding_mask(source_tensor)
    target_mask = create_padding_mask(target_tensor)

    # convert rating to tensor
    rating_tensor = torch.LongTensor(rating_list)

    # pad the f_sent_position_list and b_sent_position_list with 0s
    max_sent_num = max(src_sent_nums)
    f_sent_position_list = [sent_posi + [0] * (max_sent_num - len(sent_posi)) for sent_posi in f_sent_position_list]
    # [batch, max_sent_num]
    f_sent_position_tensor = torch.LongTensor(f_sent_position_list)

    b_sent_position_list = [sent_posi + [0] * (max_sent_num - len(sent_posi)) for sent_posi in b_sent_position_list]
    # [batch, max_sent_num]
    b_sent_position_tensor = torch.LongTensor(b_sent_position_list)

    # [batch, max_sent_num, 2]
    sent_position_tensor = torch.stack([f_sent_position_tensor, b_sent_position_tensor], dim=2)
    # [batch, max_sent_num]
    src_sent_mask = create_sequence_mask(src_sent_nums)


    # changed by wchen to a dictionary output
    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['src_sent_positions'] = sent_position_tensor
    batch_dict['src_sent_nums'] = src_sent_nums
    batch_dict['src_sent_mask'] = src_sent_mask
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['tgt_sent_2d_list'] = target_sent_2d_list
    batch_dict['tgt_tensor'] = target_tensor
    batch_dict['tgt_oov_tensor'] = target_oov_tensor
    batch_dict['tgt_lens'] = target_lens
    batch_dict['tgt_mask'] = target_mask
    batch_dict['rating_tensor'] = rating_tensor
    batch_dict['original_indices'] = original_indices
    return batch_dict
    # return source_tensor, source_lens, source_mask, source_oov_tensor, oov_lists, source_list_tokenized, target_sent_2d_list, target_tensor, target_oov_tensor, target_lens, target_mask, rating_tensor, original_indices


@curry
def coll_fn(batch, word2idx, src_max_len=-1, trg_max_len=-1):
    # batch: a list of (art_str, abs_str)
    # Remove empty data, tokenize, and truncate
    source_list_tokenized, target_list_tokenized = [], []
    target_sent_2d_list = []
    for src_sent_list, trg_sent_list in batch:
        if src_sent_list and trg_sent_list:
            target_sent_2d_list.append(trg_sent_list)
            # concat each sent list into one str
            src = ' '.join(src_sent_list)
            trg = ' '.join(trg_sent_list)
            # tokenize and truncate
            source_list_tokenized.append(tokenize(src, src_max_len))
            target_list_tokenized.append(tokenize(trg, trg_max_len))

    batch_size = len(source_list_tokenized)

    # convert to idx
    source_list, target_list, source_oov_list, target_oov_list, oov_lists = convert_batch_to_idx(source_list_tokenized, target_list_tokenized, word2idx)
    original_indices = list(range(batch_size))

    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices = zip(*seq_pairs)
    """

    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list)
    target_tensor, target_lens = convert_to_tensor(target_list)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list)
    target_oov_tensor, _ = convert_to_tensor(target_oov_list)
    source_mask = create_padding_mask(source_tensor)
    target_mask = create_padding_mask(target_tensor)

    return source_tensor, source_lens, source_mask, source_oov_tensor, oov_lists, source_list_tokenized, target_sent_2d_list, target_tensor, target_oov_tensor, target_lens, target_mask, original_indices


def convert_batch_to_idx(source_list, target_list, word2idx):
    source_list_indiced, target_list_indiced, source_oov_list_indiced, target_oov_list_indiced = [], [], [], []
    oov_lists = []
    for src, trg in zip(source_list, target_list):
        src_oov, oov_dict, oov_list = extend_vocab_oov(src, word2idx)
        src = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in src]
        trg_oov = []
        for w in trg:
            if w in word2idx:
                trg_oov.append(word2idx[w])
            elif w in oov_dict:
                trg_oov.append(oov_dict[w])
            else:
                trg_oov.append(word2idx[UNK_WORD])
        trg = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in trg]
        source_list_indiced.append(src)
        target_list_indiced.append(trg)
        source_oov_list_indiced.append(src_oov)
        target_oov_list_indiced.append(trg_oov)
        oov_lists.append(oov_list)
    return source_list_indiced, target_list_indiced, source_oov_list_indiced, target_oov_list_indiced, oov_lists


def extend_vocab_oov(src_words, word2idx):
    oov_dict = {}
    src_oov = []
    for src_word in src_words:
        if src_word in word2idx:
            src_oov.append(word2idx[src_word])
        else:
            if src_word in oov_dict:
                idx = oov_dict[src_word]
            else:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                idx = len(word2idx) + len(oov_dict)
                oov_dict[src_word] = idx
            src_oov.append(idx)
    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list


def convert_to_tensor(input_list):
    # input_list: a list of list of words, should not include the EOS word.
    input_list = [l + [EOS] for l in input_list]  # append EOS at the end of each word list
    input_list_lens = [len(l) for l in input_list]
    max_seq_len = max(input_list_lens)
    padded_batch = PAD * np.ones((len(input_list), max_seq_len))

    for j in range(len(input_list)):
        current_len = input_list_lens[j]
        padded_batch[j][:current_len] = input_list[j]

    padded_batch = torch.LongTensor(padded_batch)
    return padded_batch, input_list_lens


def create_padding_mask(padded_batch):
    #pad_idx = word2idx[PAD_WORD]
    input_mask = torch.ne(padded_batch, PAD)
    input_mask = input_mask.type(torch.FloatTensor)
    return input_mask


def create_sequence_mask(seq_lens, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    seq_lens = torch.Tensor(seq_lens)
    batch_size = seq_lens.numel()
    max_len = max_len or seq_lens.max()
    return (torch.arange(0, max_len)
            .type_as(seq_lens)
            .repeat(batch_size, 1)
            .lt(seq_lens.unsqueeze(1))
            .float())


def make_vocab(wc, vocab_size):
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    return word2idx, idx2word


def make_embedding(idx2word, w2v_file):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(idx2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    # initialize embedding weight first
    init_range = 0.1
    embedding.data.uniform_(-init_range, init_range)

    #if initializer is not None:
    #    initializer(embedding)
    oovs = []
    with torch.no_grad():
        for i in range(len(idx2word)):
            # NOTE: idx2word can be list or dict
            if i == BOS:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == EOS:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif idx2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[idx2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def remove_old_ckpts(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward, Only keep the highest three checkpoints. """
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    score_list = [float(ckpt.split('-')[-1]) for ckpt in ckpts]
    ckpts_score_sorted = sorted(zip(score_list, ckpts), key=lambda p: p[0], reverse=reverse)
    _, ckpts_sorted = zip(*ckpts_score_sorted)
    for ckpt in ckpts_sorted[3:]:
        os.remove(join(model_dir, 'ckpt', ckpt))
    logging.info("Best model: {}".format(join(model_dir, 'ckpt', ckpts_sorted[0])))

