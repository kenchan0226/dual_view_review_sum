import torch
import argparse
import config
import logging
import os
from os.path import join
import json
from utils import io
from utils.io import SummRating
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle as pkl
from model.hss_model import HSSModel
# from model.multi_task_basic_model import MultiTaskBasicModel
from model.multi_task_basic_classify_seq2seq import MultiTaskBasicClassifySeq2Seq
from model.hre_multi_task_basic_model import HirEncMultiTaskBasicModel
from model.attn_modulate_classify_seq2seq import AttnModulateClassifySeq2Seq
from model.external_feed_classify_seq2seq import ExternalFeedClassifySeq2Seq
from model.external_soft_feed_classify_seq2seq import ExternalSoftFeedClassifySeq2Seq
from model.multi_view_external_soft_feed_classify_seq2seq import MultiViewExternalSoftFeedClassifySeq2Seq
from model.multi_view_attn_modulate_classify_seq2seq import MultiViewAttnModulateClassifySeq2Seq
from model.multi_view_multi_task_basic_seq2seq import MultiViewMultiTaskBasicClassifySeq2Seq
from model.RnnEncSingleClassifier import RnnEncSingleClassifier
from model.seq2seq import Seq2SeqModel
from utils.ordinal_loss import OrdinalMSELoss, OrdinalXELoss
import torch.nn as nn

import ml_pipeline

from utils.time_log import time_since
import datetime
import time
import numpy as np
import random


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    opt.exp += '.ml'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    if "ordinal" in opt.classifier_loss_type:
        opt.ordinal = True
    else:
        opt.ordinal = False

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
        os.makedirs(join(opt.model_path, 'ckpt'))

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(join(opt.model_path, 'initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(join(opt.model_path, 'initial.json'), 'w'))

    return opt


def init_model(opt, rating_tokens_tensor):
    logging.info('======================  Model Parameters  =========================')

    if opt.model_type == 'hss':
        overall_model = HSSModel(opt)
    elif opt.model_type == 'multi_task_basic':
        overall_model = MultiTaskBasicClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == 'word_attn_modulate':
        overall_model = AttnModulateClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == 'hre_max':
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
        raise ValueError("Invalid model type")
    overall_model.to(opt.device)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        # TODO: load the saved model and override the current one

    if opt.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = io.make_embedding(opt.idx2word, opt.w2v)
        overall_model.set_embedding(embedding)

    return overall_model


def build_loader(data_path, batch_size, word2idx, src_max_len, trg_max_len, num_workers, weighted_sampling):

    """
    train_class_count = torch.load(join(data_path, "train_class_count.pt"))  # a list of int
    # debug
    # logging.info("train class count: {}".format(train_class_count))
    weights = 1. / torch.tensor(train_class_count, dtype=torch.float)
    all_label_list = train_dataset.get_all_labels_list()
    # debug
    # logging.info("All label list: {}".format(all_label_list))
    # logging.info("weights: {}".format(weights))
    # logging.info("sum: {}".format(sum(all_label_list)))
    train_sample_weights = weights[all_label_list]
    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, sampler=train_sampler, shuffle=False)
    """
    rating_path = join(data_path, "ratings", "train")
    with open(join(rating_path, "gold_ratings.pkl"), 'rb') as f:
        all_rating = pkl.load(f)
    with open(join(rating_path, "rating_count.pkl"), 'rb') as f:
        rating_count = pkl.load(f)

    coll_fn_customized = io.summ_rating_flatten_coll_fn(word2idx=word2idx, src_max_len=src_max_len,
                                                        trg_max_len=trg_max_len)

    normalized_rating_count = rating_count / rating_count.sum()
    weights = 1. / normalized_rating_count
    weights = torch.from_numpy(weights).float().to(opt.device)

    if weighted_sampling:
        all_rating_label_list = all_rating.tolist()
        train_sample_weights = weights[all_rating_label_list]
        train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
        train_loader = DataLoader(SummRating('train', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                                  batch_size=batch_size, pin_memory=True, sampler=train_sampler, shuffle=False)
    else:
        train_loader = DataLoader(SummRating('train', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                                  batch_size=batch_size, pin_memory=True, shuffle=True)

    valid_loader = DataLoader(SummRating('val', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=False)
    return train_loader, valid_loader, weights


def main(opt):
    try:
        start_time = time.time()

        # construct vocab
        with open(join(opt.data, 'vocab_cnt.pkl'), 'rb') as f:
            wc = pkl.load(f)
        word2idx, idx2word = io.make_vocab(wc, opt.v_size)
        opt.word2idx = word2idx
        opt.idx2word = idx2word

        # construct
        if opt.rating_memory_pred:
            rating_tokens_tensor = []
            for i in range(1, 6):
                vocab_i = os.path.join(opt.data, 'rating_{}_vocab_counter_no_stop_word_and_punc.pkl'.format(i))
                vocab_i = pkl.load(open(vocab_i, 'rb'))
                vocab_i = vocab_i.most_common(opt.rating_v_size)
                vocab_tokens_i = [w[0] for w in vocab_i]
                # topk_rating_tokens.append(vocab_tokens_i)
                # we assume all the topk rating tokens are in the predefined vocabulary
                # otherwise, one error will happen
                # This condition also brings convenience for the final copy mechanism
                vocab_tokens_i_tensor = [word2idx[w] for w in vocab_tokens_i]
                vocab_tokens_i_tensor = torch.LongTensor(vocab_tokens_i_tensor)
                rating_tokens_tensor.append(vocab_tokens_i_tensor)
            # [5, rating_v_size]
            rating_tokens_tensor = torch.stack(rating_tokens_tensor, dim=0)
            # save rating_tokens_tensor
            torch.save(rating_tokens_tensor, os.path.join(opt.model_path, 'rating_tokens_tensor.pt'))
        else:
            rating_tokens_tensor = None

        # dump word2idx
        with open(join(opt.model_path, 'vocab.pkl'), 'wb') as f:
            pkl.dump(word2idx, f, pkl.HIGHEST_PROTOCOL)

        # construct loader
        load_data_time = time_since(start_time)
        train_data_loader, valid_data_loader, class_weights = build_loader(opt.data, opt.batch_size, word2idx, opt.src_max_len, opt.trg_max_len, opt.batch_workers, opt.weighted_sampling)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        # construct model
        start_time = time.time()
        overall_model = init_model(opt, rating_tokens_tensor)
        logging.info(overall_model)

        # construct optimizer
        optimizer_ml = torch.optim.Adam(params=filter(lambda p: p.requires_grad, overall_model.parameters()), lr=opt.learning_rate)

        # construct loss function
        #print(class_weights)
        #exit()
        if opt.classifier_loss_type == "ordinal_mse":
            train_classification_loss_func = OrdinalMSELoss(opt.num_classes, device=opt.device)
            val_classification_loss_func = train_classification_loss_func
        elif opt.classifier_loss_type == "ordinal_xe":
            train_classification_loss_func = OrdinalXELoss(opt.num_classes, device=opt.device)
            val_classification_loss_func = train_classification_loss_func
        else:
            if opt.weighted_classifier_loss:
                train_classification_loss_func = nn.NLLLoss(reduction='mean', weight=class_weights)
            else:
                train_classification_loss_func = nn.NLLLoss(reduction='mean')
            val_classification_loss_func = nn.NLLLoss(reduction='mean')

        # train the model
        ml_pipeline.train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt, train_classification_loss_func, val_classification_loss_func)

        training_time = time_since(start_time)
        logging.info('Model path: {}'.format(opt.model_path))
        logging.info('Time for training: {}'.format(datetime.timedelta(seconds=training_time)))

    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_ml.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.train_ml_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if opt.rating_memory_pred:
        assert opt.model_type in ["multi_task_basic", "word_attn_modulate", "external_soft_feed",
                                  "multi_view_ex_soft_feed", "multi_view_attn_modulate", "multi_view_multi_task_basic"],\
            "The {} is not implemented with rating memory".format(opt.model_type)
    if opt.encoder_type == "hre_brnn":
        assert opt.model_type in ["multi_task_basic", "word_attn_modulate", "external_soft_feed",
                                  "multi_view_ex_soft_feed", "multi_view_attn_modulate", "multi_view_multi_task_basic"],\
            "The {} is not implemented with hierarchical encoder for both classifying and generation".format(opt.model_type)

    if opt.encoder_type == "sep_layers_brnn" or opt.encoder_type == "sep_layers_brnn_reverse":
        assert opt.enc_layers == 2

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=opt.stdout)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
