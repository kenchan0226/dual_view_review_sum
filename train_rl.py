import torch
import argparse
import config
import logging
import os
from os.path import join
import json
from utils import io
from utils.io import Many2ManyDataset
from model.seq2seq import Seq2SeqModel
from model.lagrangian import Lagrangian
from torch.utils.data import DataLoader
import pickle as pkl
import rreplace
import rl_pipeline

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

    opt.exp += '.rl'

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
            open(join(opt.model_path, 'rl.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(join(opt.model_path, 'rl.config'), 'wb')
                   )
        json.dump(vars(opt), open(join(opt.model_path, 'rl.json'), 'w'))

    return opt


def init_pretrained_model(pretrained_model_path, opt):
    model = Seq2SeqModel(opt)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(opt.device)
    model.eval()
    return model


def build_loader(data_path, batch_size, word2idx, src_max_len, trg_max_len, num_workers):
    coll_fn_customized = io.coll_fn(word2idx=word2idx, src_max_len=src_max_len, trg_max_len=trg_max_len)
    train_loader = DataLoader(Many2ManyDataset('train', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(Many2ManyDataset('val', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=False)
    return train_loader, valid_loader


def main(opt):
    try:
        start_time = time.time()
        # load word2idx and idx2word
        pretrained_model_dir_path = os.path.dirname(opt.pretrained_model)
        pretrained_model_dir_path = rreplace.rreplace(pretrained_model_dir_path, 'ckpt', '', 1)
        with open(join(pretrained_model_dir_path, 'vocab.pkl'), 'rb') as f:
            word2idx = pkl.load(f)
        idx2word = {i: w for w, i in word2idx.items()}
        opt.word2idx = word2idx
        opt.idx2word = idx2word
        opt.vocab_size = len(word2idx)

        # dump word2idx
        with open(join(opt.model_path, 'vocab.pkl'), 'wb') as f:
            pkl.dump(word2idx, f, pkl.HIGHEST_PROTOCOL)

        # construct loader
        load_data_time = time_since(start_time)
        train_data_loader, valid_data_loader = build_loader(opt.data, opt.batch_size, word2idx, opt.src_max_len, opt.trg_max_len, opt.batch_workers)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        # load the config of pretrained model and dump the config to the rl model dir
        old_opt = torch.load(join(pretrained_model_dir_path, "initial.config"))
        json.dump(vars(old_opt), open(join(opt.model_path, 'initial.json'), 'w'))
        torch.save(old_opt, open(join(opt.model_path, 'initial.config'), 'wb'))

        # init the pretrained model
        old_opt.word2idx = word2idx
        old_opt.idx2word = idx2word
        old_opt.device = opt.device
        model = init_pretrained_model(opt.pretrained_model, old_opt)

        # construct optimizer
        optimizer_rl = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
        if opt.constrained_mdp:
            lagrangian_model = Lagrangian(len(opt.cost_types), opt.cost_thresholds, opt.lagrangian_init_val, opt.use_lagrangian_hinge_loss)
            lagrangian_model.to(opt.device)
            optimizer_lagrangian = torch.optim.Adam(params=filter(lambda p: p.requires_grad, lagrangian_model.parameters()), lr=opt.learning_rate_multiplier)
            lagrangian_params = (lagrangian_model, optimizer_lagrangian)
        else:
            lagrangian_params = None

        # train the model
        rl_pipeline.train_model(model, optimizer_rl, train_data_loader, valid_data_loader, opt, lagrangian_params)

        training_time = time_since(start_time)
        logging.info('Model path: {}'.format(opt.model_path))
        logging.info('Time for training: {}'.format(datetime.timedelta(seconds=training_time)))
    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_ml.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.train_rl_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    assert len(opt.cost_types) == len(opt.cost_thresholds)

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
