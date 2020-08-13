import torch.nn as nn
from utils.statistics import RewardStatistics, LagrangianStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_train_and_valid_reward, export_lagrangian_stats
import sys
import logging
from validation import evaluate_reward
from utils.reward import *
import math
from utils import io
import os
from utils.io import tokenize
import nltk
from cytoolz import concat
from utils.cost import compute_batch_cost
from utils.io import remove_old_ckpts

EPS = 1e-8


def train_model(model, optimizer_rl, train_data_loader, valid_data_loader, opt, lagrangian_params=None):
    total_batch = -1
    early_stop_flag = False

    report_train_reward_statistics = RewardStatistics()
    total_train_reward_statistics = RewardStatistics()
    report_train_reward = []
    report_valid_reward = []
    if opt.constrained_mdp:
        report_train_lagrangian_statistics = LagrangianStatistics()
        report_lagrangian_loss = []
        report_lagrangian_multipliers = []
        report_violate_amounts = []
        report_lagrangian_grad_norms = []
        lagrangian_model, optimizer_lagrangian = lagrangian_params
    best_valid_reward = float('-inf')
    num_stop_increasing = 0
    if opt.train_from:  # opt.train_from:
        raise ValueError("Not implemented the function of load from trained model")

    generator = SequenceGenerator(model,
                                  bos_idx=io.BOS,
                                  eos_idx=io.EOS,
                                  pad_idx=io.PAD,
                                  beam_size=1,
                                  max_sequence_length=opt.pred_max_len,
                                  cuda=opt.gpuid > -1,
                                  n_best=1
                                  )

    model.train()

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, lagrangian_params)
            if opt.constrained_mdp:
                batch_reward_stat, batch_lagrangian_stat = stat
            else:
                batch_reward_stat = stat

            report_train_reward_statistics.update(batch_reward_stat)
            total_train_reward_statistics.update(batch_reward_stat)
            if opt.constrained_mdp:
                report_train_lagrangian_statistics.update(batch_lagrangian_stat)

            if total_batch % opt.checkpoint_interval == 0:
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            """
            if total_batch % 20 == 0:
                print("lagrangian loss: {:.5f}; grad_norm: {:.5f}; violate_amount: {:.5f}".format(report_train_lagrangian_statistics.loss(), report_train_lagrangian_statistics.grad_norm(), report_train_lagrangian_statistics.violate_amt()))
                print("lagrangian value: {}".format(lagrangian_model.get_lagrangian_multiplier_array()))
                report_train_lagrangian_statistics.clear()
                print("threshold: {}".format(lagrangian_model.cost_threshold.cpu().numpy()))
            """

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
                    print("Enter check point!")
                    sys.stdout.flush()
                    # log training reward and pg loss
                    current_train_reward = report_train_reward_statistics.reward()
                    current_train_pg_loss = report_train_reward_statistics.loss()
                    report_train_reward.append(current_train_reward)
                    # Run validation and log valid reward
                    valid_reward_stat = evaluate_reward(valid_data_loader, generator, opt)
                    model.train()
                    current_valid_reward = valid_reward_stat.reward()
                    report_valid_reward.append(current_valid_reward)
                    # print out train and valid reward
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training reward: %.4f; avg training loss: %.4f; avg validation reward: %.4f; best validation reward: %.4f' % (
                            current_train_reward, current_train_pg_loss, current_valid_reward, best_valid_reward))
                    # log lagrangian training loss and last lagrangian value
                    if opt.constrained_mdp:
                        current_lagrangian_loss = report_train_lagrangian_statistics.loss()
                        current_lagrangian_grad_norm = report_train_lagrangian_statistics.grad_norm()
                        current_violate_amount = report_train_lagrangian_statistics.violate_amt()
                        report_lagrangian_loss.append(current_lagrangian_loss)
                        report_violate_amounts.append(current_violate_amount)
                        report_lagrangian_grad_norms.append(current_lagrangian_grad_norm)
                        lagrangian_multipliers_array = lagrangian_model.get_lagrangian_multiplier_array()
                        report_lagrangian_multipliers.append(lagrangian_multipliers_array)
                        logging.info("Lagrangian_loss: %.5f; grad_norm: %.5f; violate_amount: %.5f" % (current_lagrangian_loss, current_lagrangian_grad_norm, current_violate_amount))
                        logging.info("Value of lagrangian_multipliers: {}".format(lagrangian_multipliers_array))

                    if epoch >= opt.start_decay_and_early_stop_at:
                        if current_valid_reward > best_valid_reward: # update the best valid reward and save the model parameters
                            print("Valid reward increases")
                            sys.stdout.flush()
                            best_valid_reward = current_valid_reward
                            num_stop_increasing = 0

                            check_pt_model_path = os.path.join(opt.model_path, 'ckpt', '%s-epoch-%d-total_batch-%d-valid_reward-%.3f' % (
                                opt.exp, epoch, total_batch, current_valid_reward))
                            torch.save(  # save model parameters
                                model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            logging.info('Saving checkpoint to %s' % check_pt_model_path)
                        else:
                            print("Valid reward does not increase")
                            sys.stdout.flush()
                            num_stop_increasing += 1
                            # decay the learning rate by the factor specified by opt.learning_rate_decay
                            decay_learning_rate(optimizer_rl, opt.learning_rate_decay, opt.min_lr)

                        # decay the learning rate for lagrangian multiplier
                        if opt.constrained_mdp and opt.decay_multiplier_learning_rate:
                            logging.info("Decay learning rate of lagrangian multiplier....")
                            decay_learning_rate(optimizer_lagrangian, 0.5, 1e-8)

                        if not opt.disable_early_stop:
                            if num_stop_increasing >= opt.early_stop_tolerance:
                                logging.info('Have not increased for %d check points, early stop training' % num_stop_increasing)
                                early_stop_flag = True
                                break

                    report_train_reward_statistics.clear()
                    if opt.constrained_mdp:
                        report_train_lagrangian_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_reward(report_train_reward, report_valid_reward, opt.checkpoint_interval, train_valid_curve_path)
    if opt.constrained_mdp:
        export_lagrangian_stats(report_lagrangian_loss, report_lagrangian_multipliers, report_lagrangian_grad_norms, report_violate_amounts, opt.checkpoint_interval, opt.exp_path)

    # Only keep the highest three checkpoints
    remove_old_ckpts(opt.model_path, reverse=True)

def train_one_batch(batch, generator, optimizer, opt, lagrangian_params=None):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, _, _, _, _, _ = batch
    """
    src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    src_lens: a list containing the length of src sequences for each batch, with len=batch
    src_mask: a FloatTensor, [batch, src_seq_len]
    src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    oov_lists: a list of oov words for each src, 2dlist
    """

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)

    optimizer.zero_grad()

    batch_size = src.size(0)
    reward_type = opt.reward_type
    sent_level_reward = opt.sent_level_reward
    baseline = opt.baseline
    regularization_type = opt.regularization_type
    regularization_factor = opt.regularization_factor

    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False

    trg_sent_2d_list_tokenized = []  # each item is a list of target sentences (tokenized) for an input sample
    trg_str_list = []  # each item is the target output sequence (tokenized) for an input sample
    for trg_sent_list in trg_sent_2d_list:
        trg_sent_list = [trg_sent.strip().split(' ') for trg_sent in trg_sent_list]
        trg_sent_2d_list_tokenized.append(trg_sent_list)
        trg_str_list.append(list(concat(trg_sent_list)))

    trg_sent_2d_list = trg_sent_2d_list_tokenized  # each item is a list of target sentences (tokenized) for an input sample

    # if use self critical as baseline, greedily decode a sequence from the model
    if baseline == 'self':
        # sample greedy prediction
        generator.model.eval()
        with torch.no_grad():
            greedy_sample_list, _, _, greedy_eos_idx_mask, _, _ = generator.sample(src, src_lens, src_oov, src_mask,
                                                                                   oov_lists, greedy=True,
                                                                                   entropy_regularize=False)
            greedy_str_list = sample_list_to_str_list(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size,
                                                      io.EOS,
                                                      io.UNK, opt.replace_unk,
                                                      src_str_list)
            greedy_sent_2d_list = []
            for greedy_str in greedy_str_list:
                greedy_sent_list = nltk.tokenize.sent_tokenize(' '.join(greedy_str))
                greedy_sent_list = [greedy_sent.strip().split(' ') for greedy_sent in greedy_sent_list]
                greedy_sent_2d_list.append(greedy_sent_list)

            # compute reward of greedily decoded sequence, tensor with size [batch_size]
            baseline = compute_batch_reward(greedy_str_list, greedy_sent_2d_list, trg_str_list, trg_sent_2d_list,
                                            batch_size, reward_type=reward_type,
                                            regularization_factor=0.0,
                                            regularization_type=0, entropy=None, device=src.device)
        generator.model.train()

    # sample a sequence from the model
    # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, prediction is a list of 0 dim tensors
    # log_selected_token_dist: size: [batch, output_seq_len]

    # sample sequences for multiple times
    sample_batch_size = batch_size * opt.n_sample
    src = src.repeat(opt.n_sample, 1)
    src_lens = src_lens * opt.n_sample
    src_mask = src_mask.repeat(opt.n_sample, 1)
    src_oov = src_oov.repeat(opt.n_sample, 1)
    oov_lists = oov_lists * opt.n_sample
    src_str_list = src_str_list * opt.n_sample
    trg_sent_2d_list = trg_sent_2d_list * opt.n_sample
    trg_str_list = trg_str_list * opt.n_sample
    if opt.baseline != 'none':  # repeat the greedy rewards
        #baseline = np.tile(baseline, opt.n_sample)
        baseline = baseline.repeat(opt.n_sample)  # [sample_batch_size]

    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, greedy=False, entropy_regularize=entropy_regularize)
    pred_str_list = sample_list_to_str_list(sample_list, oov_lists, opt.idx2word, opt.vocab_size, io.EOS,
        io.UNK, opt.replace_unk, src_str_list)  # a list of word list, len(pred_word_2dlist)=sample_batch_size
    sample_time = time_since(start_time)
    max_pred_seq_len = log_selected_token_dist.size(1)

    pred_sent_2d_list = []  # each item is a list of predicted sentences (tokenized) for an input sample, used to compute summary level Rouge-l
    for pred_str in pred_str_list:
        pred_sent_list = nltk.tokenize.sent_tokenize(' '.join(pred_str))
        pred_sent_list = [pred_sent.strip().split(' ') for pred_sent in pred_sent_list]
        pred_sent_2d_list.append(pred_sent_list)

    if entropy_regularize:
        entropy_array = entropy.data.cpu().numpy()
    else:
        entropy_array = None

    # compute the reward
    with torch.no_grad():
        if sent_level_reward:
            raise ValueError("Not implemented.")
        else:  # neither using reward shaping
            # only receive reward at the end of whole sequence, tensor: [sample_batch_size]
            cumulative_reward = compute_batch_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, sample_batch_size, reward_type=reward_type,
                           regularization_factor=regularization_factor, regularization_type=regularization_type, entropy=entropy_array, device=src.device)
            # store the sum of cumulative reward (before baseline) for the experiment log
            cumulative_reward_sum = cumulative_reward.detach().sum(0).item()

            if opt.constrained_mdp:
                lagrangian_model, optimizer_lagrangian = lagrangian_params
                cumulative_cost = compute_batch_cost(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, sample_batch_size, opt.cost_types, src.device)  # [sample_batch_size, num_cost_types]
                #cumulative_cost = torch.from_numpy(cumulative_cost_array).type(torch.FloatTensor).to(src.device)

                # cumulative_cost: [sample_batch_size, len(cost_types)]
                # subtract the regularization term: \lambda \dot C_t
                constraint_regularization = lagrangian_model.compute_regularization(cumulative_cost)  # [sample_batch_size]
                cumulative_reward -= constraint_regularization

            # Subtract the cumulative reward by a baseline if needed
            if opt.baseline != 'none':
                cumulative_reward = cumulative_reward - baseline  # [sample_batch_size]
            # q value estimation for each time step equals to the (baselined) cumulative reward
            q_value_estimate = cumulative_reward.unsqueeze(1).repeat(1, max_pred_seq_len)  # [sample_batch_size, max_pred_seq_len]
            #q_value_estimate_array = np.tile(cumulative_reward.reshape([-1, 1]), [1, max_pred_seq_len])  # [batch, max_pred_seq_len]

    #shapped_baselined_reward = torch.gather(shapped_baselined_phrase_reward, dim=1, index=pred_phrase_idx_mask)

    # use the return as the estimation of q_value at each step

    #q_value_estimate = torch.from_numpy(q_value_estimate_array).type(torch.FloatTensor).to(src.device)
    q_value_estimate.requires_grad_(True)
    q_estimate_compute_time = time_since(start_time)

    # compute the policy gradient objective
    pg_loss = compute_pg_loss(log_selected_token_dist, output_mask, q_value_estimate)

    # back propagation to compute the gradient
    if opt.loss_normalization == "samples": # use number of target tokens to normalize the loss
        normalization = opt.n_sample
    elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
        normalization = sample_batch_size
    else:
        normalization = 1
    start_time = time.time()
    pg_loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(generator.model.parameters(), opt.max_grad_norm)

    # take a step of gradient descent
    optimizer.step()

    stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), sample_batch_size, sample_time, q_estimate_compute_time, backward_time)
    # (final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0)
    # reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0

    if opt.constrained_mdp:
        lagrangian_loss, lagrangian_grad_norm, violate_amount = train_lagrangian_multiplier(lagrangian_model, cumulative_cost, optimizer_lagrangian, normalization, opt.max_grad_norm)
        lagrangian_stat = LagrangianStatistics(lagrangian_loss=lagrangian_loss, n_batch=sample_batch_size, lagrangian_grad_norm=lagrangian_grad_norm, violate_amount=violate_amount)
        stat = (stat, lagrangian_stat)

    return stat, log_selected_token_dist.detach()


def train_lagrangian_multiplier(lagrangian_model, cumulative_cost, optimizer, normalization, max_grad_norm):
    """
    :param lagrangian_multiplier: [batch, len(cost_types)]
    :param cumulative_cost: [batch, len(cost_types)]
    :param cost_threshold: [len(cost_types)]
    :param optimizer:
    :param normalization
    :return:
    """
    optimizer.zero_grad()
    lagrangian_loss, violate_amount = lagrangian_model(cumulative_cost)
    lagrangian_loss.div(normalization).backward()
    grad_norm = lagrangian_model.lagrangian_multiplier.grad.detach().sum().item()
    #grad_norm = lagrangian_model.lagrangian_multiplier.grad.detach().norm(2).item()
    #grad_norm_before_clipping = nn.utils.clip_grad_norm_(lagrangian_model.parameters(), max_grad_norm)
    optimizer.step()
    lagrangian_model.clamp_lagrangian_multiplier()
    return lagrangian_loss.item(), grad_norm, violate_amount

def decay_learning_rate(optimizer, decay_factor, min_lr):
    # decay the learning rate by the factor specified by opt.learning_rate_decay
    if decay_factor < 1:
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * decay_factor
            if new_lr < min_lr:
                new_lr = min_lr
            if old_lr - new_lr > EPS:
                param_group['lr'] = new_lr
        logging.info('Learning rate drops to {}'.format(new_lr))

