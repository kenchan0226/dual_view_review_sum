import torch
import config
import argparse
import pickle as pkl
from utils import io
from utils.io import DecodeDataset, eval_coll_fn, SummRating
from torch.utils.data import DataLoader
import os
from os.path import join
from model import hss_seq2seq
from model.hss_classifier import HSSClassifier
from sequence_generator import SequenceGenerator
from tqdm import tqdm
import json
from utils.string_helper import prediction_to_sentence
from utils.io import create_sequence_mask
import nltk
# import rreplace
import torch.nn as nn
import numpy as np
from model.hss_model import HSSModel
# from model.multi_task_basic_model import MultiTaskBasicModel
from model.multi_task_basic_classify_seq2seq import MultiTaskBasicClassifySeq2Seq
from model.attn_modulate_classify_seq2seq import AttnModulateClassifySeq2Seq
from model.hre_multi_task_basic_model import HirEncMultiTaskBasicModel
from model.external_feed_classify_seq2seq import ExternalFeedClassifySeq2Seq
from model.external_soft_feed_classify_seq2seq import ExternalSoftFeedClassifySeq2Seq
from model.multi_view_external_soft_feed_classify_seq2seq import MultiViewExternalSoftFeedClassifySeq2Seq
from model.multi_view_attn_modulate_classify_seq2seq import MultiViewAttnModulateClassifySeq2Seq
from model.multi_view_multi_task_basic_seq2seq import MultiViewMultiTaskBasicClassifySeq2Seq
from model.RnnEncSingleClassifier import RnnEncSingleClassifier
from model.seq2seq import Seq2SeqModel
from types import SimpleNamespace
from utils.ordinal_utilities import binary_results_to_rating_preds
from validation import evaluate_loss


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    # fill time into the name
    if opt.pred_path.find('%s') > 0:
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    # make directory
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
        os.makedirs(join(opt.pred_path, 'output'))

    # dump configuration
    torch.save(opt, open(join(opt.pred_path, 'decode.config'), 'wb'))
    json.dump(vars(opt), open(join(opt.pred_path, 'log.json'), 'w'))

    return opt


def init_pretrained_model(pretrained_model_path, opt, rating_tokens_tensor):
    if opt.model_type == 'hss':
        overall_model = HSSModel(opt)
    elif opt.model_type == 'multi_task_basic':
        overall_model = MultiTaskBasicClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "word_attn_modulate":
        overall_model = AttnModulateClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "hre_max":
        overall_model = HirEncMultiTaskBasicModel(opt)
    elif opt.model_type == 'external_feed':
        overall_model = ExternalFeedClassifySeq2Seq(opt)
    elif opt.model_type == "external_soft_feed":
        overall_model = ExternalSoftFeedClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "multi_view_ex_soft_feed":
        overall_model = MultiViewExternalSoftFeedClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "multi_view_attn_modulate":
        overall_model = MultiViewAttnModulateClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "multi_view_multi_task_basic":
        overall_model = MultiViewMultiTaskBasicClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "rnn_enc_single_classifier":
        overall_model = RnnEncSingleClassifier(opt)
    elif opt.model_type == "seq2seq":
        overall_model = Seq2SeqModel(opt)
    else:
        print(opt.model_type)
        raise ValueError("Invalid model type")
    overall_model.to(opt.device)
    overall_model.load_state_dict(torch.load(pretrained_model_path))
    overall_model.eval()
    return overall_model


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    dec_states = beam_search_result["dec_states"]
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, dec_state_n_best, oov, src_word_list in zip(predictions, scores, attention, dec_states, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            #sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_dict['dec_states'] = dec_state_n_best # a list of FloatTensor[output sequence length, memory_bank_size], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def preprocess_hss_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    sentiment_context = beam_search_result['sentiment_context']  # a list of list, len=(batch, n_best), tensor = [out_seq_len, memory_bank_size], seq_len including eos
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, senti_n_best, oov, src_word_list in zip(predictions, scores, attention, sentiment_context, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            #sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_dict['sentiment_context'] = senti_n_best  # a list of FloatTensor[output sequence length, memory_bank_size], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def predict(test_data_loader, overall_model, opt):

    if isinstance(overall_model, AttnModulateClassifySeq2Seq) or \
            isinstance(overall_model, ExternalFeedClassifySeq2Seq) or \
            isinstance(overall_model, ExternalSoftFeedClassifySeq2Seq) or \
            isinstance(overall_model, MultiViewExternalSoftFeedClassifySeq2Seq) or \
            isinstance(overall_model, MultiViewAttnModulateClassifySeq2Seq) or \
            isinstance(overall_model, MultiViewMultiTaskBasicClassifySeq2Seq) or \
            isinstance(overall_model, MultiTaskBasicClassifySeq2Seq) or \
            isinstance(overall_model, RnnEncSingleClassifier) or \
            isinstance(overall_model, Seq2SeqModel):
        seq2seq_model = overall_model
    else:
        seq2seq_model = overall_model.seq2seq_model
        classifier_model = overall_model.classifier_model

    if not isinstance(overall_model, RnnEncSingleClassifier):
        generator = SequenceGenerator(seq2seq_model,
                                      bos_idx=io.BOS,
                                      eos_idx=io.EOS,
                                      pad_idx=io.PAD,
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.pred_max_len,
                                      include_attn_dist=opt.include_attn_dist,
                                      length_penalty_factor=opt.length_penalty_factor,
                                      coverage_penalty_factor=opt.coverage_penalty_factor,
                                      length_penalty=opt.length_penalty,
                                      coverage_penalty=opt.coverage_penalty,
                                      cuda=opt.gpuid > -1,
                                      n_best=opt.n_best,
                                      block_ngram_repeat=opt.block_ngram_repeat,
                                      ignore_when_blocking=opt.ignore_when_blocking
                                      )
    else:
        generator = seq2seq_model
    enc_rating_preds = None
    dec_rating_preds = None
    merged_rating_preds = None
    dec_logit = None
    num_exported_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            # src, src_lens, src_mask, src_oov, oov_lists, src_str_list, original_idx_list = batch

            # changed by wchen to a dictionary batch
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src_mask = batch['src_mask']
            src_sent_positions = batch['src_sent_positions']
            src_sent_nums = batch['src_sent_nums']
            src_sent_mask = batch['src_sent_mask']
            src_oov = batch['src_oov_tensor']
            oov_lists = batch['oov_lists']
            src_str_list = batch['src_list_tokenized']
            original_idx_list = batch['original_indices']

            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            src_sent_positions = src_sent_positions.to(opt.device)
            src_sent_mask = src_sent_mask.to(opt.device)

            # Predict the summaries for the batch using beam search

            if isinstance(overall_model, HSSModel):
                beam_search_result, encoder_memory_bank = generator.hss_beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx, src_sent_positions, src_sent_nums, src_sent_mask)
                # a dictionary, with predictions, scores, attention and sentiment_context
                pred_list = preprocess_hss_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)
                # pred_list: list of {"sentences": [], "scores": [], "attention": [], "sentiment_context": []}

                # construct the sentiment_context_tensor for the classifier
                sentiment_context_list = [pred_result['sentiment_context'][0] for pred_result in pred_list]
                # get the output sequence lens with EOS
                out_lens = [sentiment_context.size(0) for sentiment_context in sentiment_context_list]

                # list of [output sequence length, memory_bank], len=batch
                # pad it
                sentiment_context_tensor = torch.nn.utils.rnn.pad_sequence(sentiment_context_list,
                                                                           batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                # Predict the ratings for the batch
                # forward classification model
                # 1. mask the memory bank vector of each padded src token as -inf
                # [batch, src_len, 1]
                expand_src_mask = src_mask.unsqueeze(-1)
                adding_src_mask = (1 - expand_src_mask).masked_fill((1 - expand_src_mask).byte(), -float('inf'))
                encoder_memory_bank = encoder_memory_bank * expand_src_mask + adding_src_mask

                # 2. mask the sentiment context vector of each padded trg token as -inf
                # [batch, trg_len, 1]
                out_mask = create_sequence_mask(out_lens)
                out_mask = out_mask.cuda()
                expand_out_mask = out_mask.unsqueeze(-1)
                adding_out_mask = (1 - expand_out_mask).masked_fill((1 - expand_out_mask).byte(), -float('inf'))
                sentiment_context_tensor = sentiment_context_tensor * expand_out_mask + adding_out_mask

                # sentiment_context_tensor, encoder_memory_bank
                enc_logit = classifier_model(encoder_memory_bank, sentiment_context_tensor)
            # elif isinstance(overall_model, MultiTaskBasicModel):
            #     beam_search_result, encoder_memory_bank = generator.beam_search(src, src_lens, src_oov, src_mask,
            #                                                                     oov_lists, opt.word2idx, src_sent_positions, src_sent_nums, src_sent_mask)
            #     # a dictionary, with predictions, scores, attention
            #     pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
            #                                               io.EOS, io.UNK, opt.replace_unk, src_str_list)
            #     # pred_list: list of {"sentences": [], "scores": [], "attention": []}
            #     # Predict the ratings for the batch
            #     # 1. mask the memory bank vector of each padded src token as -inf
            #     # # [batch, src_len, 1]
            #     # if overall_model.classifier_type == "max":
            #     #     expand_src_mask = src_mask.unsqueeze(-1)
            #     #     adding_src_mask = (1 - expand_src_mask).masked_fill((1 - expand_src_mask).byte(), -float('inf'))
            #     #     encoder_memory_bank = encoder_memory_bank * expand_src_mask + adding_src_mask
            #
            #     # encoder_memory_bank
            #     classifier_output = classifier_model(encoder_memory_bank, src_mask)
            #     if isinstance(classifier_output, tuple):
            #         enc_logit = classifier_output[0]
            #     else:
            #         enc_logit = classifier_output

            elif isinstance(overall_model, MultiTaskBasicClassifySeq2Seq):
                beam_search_result, enc_logit = generator.multi_task_basic_beam_search(src, src_lens, src_oov, src_mask,
                                                                                       oov_lists, opt.word2idx,
                                                                                       src_sent_positions, src_sent_nums,
                                                                                       src_sent_mask)
                pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                          io.EOS, io.UNK, opt.replace_unk, src_str_list)

            elif isinstance(overall_model, AttnModulateClassifySeq2Seq):
                beam_search_result, enc_logit = generator.word_attn_beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx, src_sent_positions, src_sent_nums, src_sent_mask)
                pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                          io.EOS, io.UNK, opt.replace_unk, src_str_list)
            elif isinstance(overall_model, HirEncMultiTaskBasicModel):
                beam_search_result, encoder_memory_bank = generator.hre_beam_search(src, src_lens, src_oov, src_mask,
                                                                                    oov_lists, opt.word2idx, src_sent_positions, src_sent_nums, src_sent_mask)
                # a dictionary, with predictions, scores, attention
                pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                          io.EOS, io.UNK, opt.replace_unk, src_str_list)
                # pred_list: list of {"sentences": [], "scores": [], "attention": []}
                # Predict the ratings for the batch
                # encoder_memory_bank
                enc_logit = classifier_model(encoder_memory_bank, src_mask)
            elif isinstance(overall_model, ExternalFeedClassifySeq2Seq):
                beam_search_result, batch_rating_tensor = generator.external_feed_beam_search(src, src_lens, src_oov, src_mask,
                                                                            oov_lists, opt.word2idx, src_sent_positions, src_sent_nums, src_sent_mask)
                pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                          io.EOS, io.UNK, opt.replace_unk, src_str_list)
            elif isinstance(overall_model, ExternalSoftFeedClassifySeq2Seq):
                beam_search_result, enc_logit = generator.external_soft_feed_beam_search(src, src_lens, src_oov, src_mask,
                                                                            oov_lists, opt.word2idx, src_sent_positions, src_sent_nums, src_sent_mask)
                pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                          io.EOS, io.UNK, opt.replace_unk, src_str_list)
            elif isinstance(overall_model, MultiViewExternalSoftFeedClassifySeq2Seq):
                beam_search_result, enc_logit = generator.external_soft_feed_beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx,
                                                                                         src_sent_positions, src_sent_nums, src_sent_mask)
                if overall_model.dec_classify_input_type == 'attn_vec':
                    # the input of the decoder classifier is the attentional vectors
                    pred_list = preprocess_hss_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size,
                                                              oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)

                    # construct the sentiment_context_tensor for the classifier
                    sentiment_context_list = [pred_result['sentiment_context'][0] for pred_result in pred_list]
                    # get the output sequence lens with EOS
                    out_lens = [sentiment_context.size(0) for sentiment_context in sentiment_context_list]

                    # list of [output sequence length, memory_bank], len=batch, padding
                    sentiment_context_tensor = torch.nn.utils.rnn.pad_sequence(sentiment_context_list, batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                    dec_classifier_input = sentiment_context_tensor
                else:
                    # the input of the decoder classifier is the hidden states of the decoder
                    assert overall_model.dec_classify_input_type == 'dec_state'
                    pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size,
                                                              oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)

                    # construct the dec_states_tensor for the classifier
                    dec_states_list = [pred_result['dec_states'][0] for pred_result in pred_list]
                    # get the output sequence lens with EOS
                    out_lens = [dec_state.size(0) for dec_state in dec_states_list]

                    # list of [output sequence length, memory_bank], len=batch, padding
                    dec_states_tensor = torch.nn.utils.rnn.pad_sequence(dec_states_list, batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                    dec_classifier_input = dec_states_tensor

                # [batch, trg_len, 1]
                out_mask = create_sequence_mask(out_lens)
                out_mask = out_mask.cuda()
                # sentiment_context_tensor, encoder_memory_bank
                dec_logit = overall_model.dec_classifier(dec_classifier_input, out_mask)
                if isinstance(dec_logit, tuple):
                    dec_logit = dec_logit[0]
            elif isinstance(overall_model, MultiViewAttnModulateClassifySeq2Seq):
                beam_search_result, enc_logit = generator.word_attn_beam_search(src, src_lens, src_oov, src_mask,
                                                                                oov_lists, opt.word2idx,
                                                                                src_sent_positions, src_sent_nums,
                                                                                src_sent_mask)
                if overall_model.dec_classify_input_type == 'attn_vec':
                    # the input of the decoder classifier is the attentional vectors
                    pred_list = preprocess_hss_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size,
                                                              oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)

                    # construct the sentiment_context_tensor for the classifier
                    sentiment_context_list = [pred_result['sentiment_context'][0] for pred_result in pred_list]
                    # get the output sequence lens with EOS
                    out_lens = [sentiment_context.size(0) for sentiment_context in sentiment_context_list]

                    # list of [output sequence length, memory_bank], len=batch, padding
                    sentiment_context_tensor = torch.nn.utils.rnn.pad_sequence(sentiment_context_list, batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                    dec_classifier_input = sentiment_context_tensor
                else:
                    # the input of the decoder classifier is the hidden states of the decoder
                    assert overall_model.dec_classify_input_type == 'dec_state'
                    pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size,
                                                              oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)

                    # construct the dec_states_tensor for the classifier
                    dec_states_list = [pred_result['dec_states'][0] for pred_result in pred_list]
                    # get the output sequence lens with EOS
                    out_lens = [dec_state.size(0) for dec_state in dec_states_list]

                    # list of [output sequence length, memory_bank], len=batch, padding
                    dec_states_tensor = torch.nn.utils.rnn.pad_sequence(dec_states_list, batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                    dec_classifier_input = dec_states_tensor

                # [batch, trg_len, 1]
                out_mask = create_sequence_mask(out_lens)
                out_mask = out_mask.cuda()
                # sentiment_context_tensor, encoder_memory_bank
                dec_logit = overall_model.dec_classifier(dec_classifier_input, out_mask)
                if isinstance(dec_logit, tuple):
                    dec_logit = dec_logit[0]
            elif isinstance(overall_model, MultiViewMultiTaskBasicClassifySeq2Seq):
                beam_search_result, enc_logit = generator.multi_task_basic_beam_search(src, src_lens, src_oov, src_mask,
                                                                                       oov_lists, opt.word2idx,
                                                                                       src_sent_positions, src_sent_nums,
                                                                                       src_sent_mask)
                if overall_model.dec_classify_input_type == 'attn_vec':
                    # the input of the decoder classifier is the attentional vectors
                    pred_list = preprocess_hss_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size,
                                                                  oov_lists, io.EOS, io.UNK, opt.replace_unk,
                                                                  src_str_list)

                    # construct the sentiment_context_tensor for the classifier
                    sentiment_context_list = [pred_result['sentiment_context'][0] for pred_result in pred_list]
                    # get the output sequence lens with EOS
                    out_lens = [sentiment_context.size(0) for sentiment_context in sentiment_context_list]

                    # list of [output sequence length, memory_bank], len=batch, padding
                    sentiment_context_tensor = torch.nn.utils.rnn.pad_sequence(sentiment_context_list,
                                                                               batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                    dec_classifier_input = sentiment_context_tensor
                else:
                    # the input of the decoder classifier is the hidden states of the decoder
                    assert overall_model.dec_classify_input_type == 'dec_state'
                    pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size,
                                                              oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)

                    # construct the dec_states_tensor for the classifier
                    dec_states_list = [pred_result['dec_states'][0] for pred_result in pred_list]
                    # get the output sequence lens with EOS
                    out_lens = [dec_state.size(0) for dec_state in dec_states_list]

                    # list of [output sequence length, memory_bank], len=batch, padding
                    dec_states_tensor = torch.nn.utils.rnn.pad_sequence(dec_states_list,
                                                                        batch_first=True)  # [batch, max_out_seq_len, memory_bank_size]
                    dec_classifier_input = dec_states_tensor

                # [batch, trg_len, 1]
                out_mask = create_sequence_mask(out_lens)
                out_mask = out_mask.cuda()
                # sentiment_context_tensor, encoder_memory_bank
                dec_logit = overall_model.dec_classifier(dec_classifier_input, out_mask)
                if isinstance(dec_logit, tuple):
                    dec_logit = dec_logit[0]
            elif isinstance(overall_model, RnnEncSingleClassifier):
                _, _, _, _, _, enc_logit, classifier_attention_dist = \
                    overall_model(src, src_lens, None, src_oov, None, src_mask, None, None, src_sent_positions, src_sent_nums, src_sent_mask)
                pred_list = None
            elif isinstance(overall_model, Seq2SeqModel):
                beam_search_result, _ = generator.beam_search(src, src_lens, src_oov, src_mask,
                                                              oov_lists, opt.word2idx,
                                                              src_sent_positions,
                                                              src_sent_nums,
                                                              src_sent_mask)
                pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                          io.EOS, io.UNK, opt.replace_unk, src_str_list)
                enc_logit = None
            else:
                raise ValueError("invalid model type")

            # append rating outputs
            if not isinstance(overall_model, ExternalFeedClassifySeq2Seq):
                if enc_rating_preds is None:
                    if opt.ordinal:
                        enc_rating_preds = binary_results_to_rating_preds(enc_logit.detach().cpu().numpy()) if enc_logit is not None else None
                        dec_rating_preds = binary_results_to_rating_preds(dec_logit.detach().cpu().numpy()) if dec_logit is not None else None
                        # add for merged rating preds
                        if dec_logit is not None:
                            # [batch, 5]
                            merged_logit = (enc_logit + dec_logit) / 2
                            merged_rating_preds = binary_results_to_rating_preds(merged_logit.detach().cpu().numpy())
                    else:
                        enc_rating_preds = enc_logit.detach().cpu().numpy() if enc_logit is not None else None
                        dec_rating_preds = dec_logit.detach().cpu().numpy() if dec_logit is not None else None
                        # add for merged rating preds
                        if dec_logit is not None:
                            # [batch, 5]
                            merged_logit = (enc_logit + dec_logit) / 2
                            merged_rating_preds = merged_logit.detach().cpu().numpy()
                else:
                    if opt.ordinal:
                        enc_rating_preds = np.append(enc_rating_preds, binary_results_to_rating_preds(
                            enc_logit.detach().cpu().numpy()), axis=0) if enc_rating_preds is not None else None
                        dec_rating_preds = np.append(dec_rating_preds, binary_results_to_rating_preds(
                            dec_logit.detach().cpu().numpy()), axis=0) if dec_rating_preds is not None else None
                        if dec_logit is not None:
                            # [batch, 5]
                            merged_logit = (enc_logit + dec_logit) / 2
                            merged_rating_preds = np.append(merged_rating_preds, binary_results_to_rating_preds(
                                merged_logit.detach().cpu().numpy()), axis=0)
                    else:
                        enc_rating_preds = np.append(enc_rating_preds, enc_logit.detach().cpu().numpy(), axis=0) if enc_rating_preds is not None else None
                        dec_rating_preds = np.append(dec_rating_preds, dec_logit.detach().cpu().numpy(), axis=0) if dec_rating_preds is not None else None
                        if dec_logit is not None:
                            # [batch, 5]
                            merged_logit = (enc_logit + dec_logit) / 2
                            merged_rating_preds = np.append(merged_rating_preds, merged_logit.detach().cpu().numpy(), axis=0)
            else:
                # append rating outputs
                if enc_rating_preds is None:
                    enc_rating_preds = batch_rating_tensor.detach().cpu().numpy()
                else:
                    enc_rating_preds = np.append(enc_rating_preds, batch_rating_tensor.detach().cpu().numpy(), axis=0)

            # For every input sample, export the predicted sentences to a .dec file
            if pred_list is not None:
                for src_str, pred, oov in zip(src_str_list, pred_list, oov_lists):
                    # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                    # pred_seq_list: a list of sequence objects, sorted by scores
                    # oov: a list of oov words
                    pred_str_list = pred['sentences']  # predicted sentences from a single src, a list of list of word, with len=[n_best, out_seq_len], does not include the final <EOS>
                    pred_score_list = pred['scores']
                    pred_attn_list = pred['attention']  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
                    # debug
                    #print(pred_str_list)

                    decode_out_str = ' '.join(pred_str_list[0])
                    decode_out_sent_list = nltk.tokenize.sent_tokenize(decode_out_str)

                    # output the predicted sentences to a file
                    with open(join(opt.pred_path, 'output/{}.dec'.format(num_exported_samples)), 'w') as f:
                        f.write(io.make_html_safe('\n'.join(decode_out_sent_list)))
                    num_exported_samples += 1

    if not isinstance(overall_model, ExternalFeedClassifySeq2Seq) and not opt.ordinal:
        enc_rating_preds = np.argmax(enc_rating_preds, axis=1) if enc_rating_preds is not None else None
        dec_rating_preds = np.argmax(dec_rating_preds, axis=1) if dec_rating_preds is not None else None
        merged_rating_preds = np.argmax(merged_rating_preds, axis=1) if merged_rating_preds is not None else None

    # dump word2idx
    with open(join(opt.pred_path, 'rating_output.pkl'), 'wb') as f:
        pkl.dump(enc_rating_preds, f, pkl.HIGHEST_PROTOCOL)

    if dec_rating_preds is not None:
        with open(join(opt.pred_path, 'dec_rating_output.pkl'), 'wb') as f:
            pkl.dump(dec_rating_preds, f, pkl.HIGHEST_PROTOCOL)

    if merged_rating_preds is not None:
        with open(join(opt.pred_path, 'merged_rating_output.pkl'), 'wb') as f:
            pkl.dump(merged_rating_preds, f, pkl.HIGHEST_PROTOCOL)

def main(opt):
    # load word2idx and idx2word
    model_dir_path = os.path.dirname(opt.pretrained_model)
    # model_dir_path = rreplace.rreplace(model_dir_path, 'ckpt', '', 1)
    # model_dir_path = model_dir_path.replace('ckpt', '', 1)
    model_dir_path = ''.join(model_dir_path.rsplit('ckpt', 1))
    with open(join(model_dir_path, 'vocab.pkl'), 'rb') as f:
        word2idx = pkl.load(f)
    # load rating_tokens_tensor
    if os.path.exists(os.path.join(model_dir_path, 'rating_tokens_tensor.pt')):
        rating_tokens_tensor = torch.load(os.path.join(model_dir_path, 'rating_tokens_tensor.pt'))
    else:
        rating_tokens_tensor = None

    idx2word = {i: w for w, i in word2idx.items()}
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab_size = len(word2idx)

    # load data
    if opt.teacher_force_evaluate:
        assert opt.split in ["val", "test"]
        # if opt.split == 'val':
        #     opt.trg_max_len = 100
        #     opt.src_max_len = 400
        # else:
        opt.trg_max_len = -1
        opt.src_max_len = -1
        coll_fn_customized = io.summ_rating_flatten_coll_fn(word2idx=word2idx, src_max_len=opt.src_max_len,
                                                            trg_max_len=opt.trg_max_len)
        test_loader = DataLoader(SummRating(opt.split, opt.data), collate_fn=coll_fn_customized, num_workers=opt.batch_workers,
                                  batch_size=opt.batch_size, pin_memory=True, shuffle=False)
    else:
        test_loader = DataLoader(DecodeDataset(opt.split, opt.data), collate_fn=eval_coll_fn(word2idx=word2idx, src_max_len=opt.src_max_len),
                                  num_workers=opt.batch_workers,
                                  batch_size=opt.batch_size, pin_memory=True, shuffle=False)

    # init the pretrained model
    old_opt_dict = json.load(open(join(model_dir_path, "initial.json")))
    old_opt = SimpleNamespace(**old_opt_dict)
    #old_opt = torch.load(join(model_dir_path, "initial.config"))
    old_opt.word2idx = word2idx
    old_opt.idx2word = idx2word
    old_opt.device = opt.device
    opt.ordinal = old_opt.ordinal
    #opt.model_type = old_opt.model_type
    overall_model = init_pretrained_model(opt.pretrained_model, old_opt, rating_tokens_tensor)

    # Print out predict path
    print("Prediction path: %s" % opt.pred_path)

    # output the summaries to opt.pred_path/output
    if opt.teacher_force_evaluate:
        val_classification_loss_func = nn.NLLLoss(reduction='mean')
        old_opt.inconsistency_loss_type = 'KL_div'
        evaluate_loss(test_loader, overall_model, val_classification_loss_func, old_opt, print_incon_stats=True)
    else:
        predict(test_loader, overall_model, opt)



if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.predict_opts(parser)
    opt = parser.parse_args()

    opt = process_opt(opt)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    main(opt)

