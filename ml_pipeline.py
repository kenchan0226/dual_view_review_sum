import torch.nn as nn
from utils.masked_loss import masked_cross_entropy
from utils.statistics import JointLossStatistics
from utils.time_log import time_since
from validation import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
from utils.report import export_train_and_valid_loss
from utils.io import remove_old_ckpts
from utils.inconsistency_loss import inconsistency_loss_func

EPS = 1e-8


def train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt, train_classification_loss_func, val_classification_loss_func):
    exp = opt.exp.split('.')[0]

    # make the code compatible when tensorboardX is not available
    try:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(comment='_' + exp)
    except ModuleNotFoundError:
        tb_writer = None

    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    #total_train_loss_statistics = JointLossStatistics()
    report_train_loss_statistics = JointLossStatistics()
    #report_train_ppl = []
    #report_valid_ppl = []
    #report_train_loss = []
    #report_valid_loss = []
    best_valid_joint_loss = float('inf')
    num_stop_dropping = 0

    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    overall_model.train()

    final_best_valid_joint_loss = float('inf')
    final_correspond_valid_ppl = float('inf')
    final_correspond_enc_class_loss = 0.0
    final_correspond_enc_class_f1 = 0.0
    final_correspond_dec_class_loss = 0.0
    final_correspond_dec_class_f1 = 0.0
    final_inconsist_loss = 0.0

    previous_valid_joint_loss = float('inf')
    previous_valid_ppl = float('inf')

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            # Training
            batch_loss_stat, decoder_dist = train_one_batch(batch, overall_model, optimizer_ml, opt, total_batch, train_classification_loss_func, tb_writer)
            report_train_loss_statistics.update(batch_loss_stat)
            #total_train_loss_statistics.update(batch_loss_stat)

            #logging.info("one_batch")
            #report_loss.append(('train_ml_loss', loss_ml))
            #report_loss.append(('PPL', loss_ml))

            if total_batch % opt.checkpoint_interval == 0:
                logging.info("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
                    logging.info("Enter check point!")
                    sys.stdout.flush()
                    # log training loss and training ppl
                    train_ppl = report_train_loss_statistics.ppl()
                    train_gen_loss = report_train_loss_statistics.generation_loss()
                    train_class_loss = report_train_loss_statistics.classification_loss()
                    train_enc_class_loss = report_train_loss_statistics.enc_classification_loss()
                    train_dec_class_loss = report_train_loss_statistics.dec_classification_loss()
                    train_inconsist_loss = report_train_loss_statistics.inconsist_loss()
                    train_joint_loss = report_train_loss_statistics.joint_loss()
                    # Run validation and log valid loss and ppl
                    valid_loss_stat, valid_class_result = evaluate_loss(valid_data_loader, overall_model, val_classification_loss_func, opt)
                    overall_model.train()
                    valid_ppl = valid_loss_stat.ppl()
                    #valid_gen_loss = valid_loss_stat.generation_loss()
                    valid_class_loss = valid_loss_stat.classification_loss()
                    valid_enc_class_loss = valid_loss_stat.enc_classification_loss()
                    valid_dec_class_loss = valid_loss_stat.dec_classification_loss()
                    valid_inconsist_loss = valid_loss_stat.inconsist_loss()
                    valid_joint_loss = valid_loss_stat.joint_loss()
                    # valid_f1 = valid_class_result['f1']
                    valid_enc_f1 = valid_class_result[0]['f1']
                    valid_f1 = valid_enc_f1
                    valid_dec_f1 = valid_class_result[1]['f1'] if valid_class_result[1] is not None else 0.0

                    #valid_acc = valid_class_result['acc']
                    # debug
                    if math.isnan(valid_joint_loss) or math.isnan(train_joint_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                        exit()
                    # print out train and valid loss
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'training gen loss: %.3f; training ppl: %.3f; validation ppl: %.3f' % (
                            train_gen_loss, train_ppl, valid_ppl))
                    logging.info(
                        'training class loss: %.3f; valid class loss: %.3f; valid class f1: %.3f' % (
                            train_class_loss, valid_class_loss, valid_f1))
                    logging.info(
                        'training enc class loss: %.3f; valid enc class loss: %.3f; valid enc class f1: %.3f' % (
                            train_enc_class_loss, valid_enc_class_loss, valid_enc_f1))
                    logging.info(
                        'training dec class loss: %.3f; valid dec class loss: %.3f; valid dec class f1: %.3f' % (
                            train_dec_class_loss, valid_dec_class_loss, valid_dec_f1))
                    logging.info(
                        'training inconsistency loss: %.5f; valid inconsistency loss: %.5f' % (
                            train_inconsist_loss, valid_inconsist_loss))
                    logging.info(
                        'training joint loss: %.3f; valid joint loss: %.3f; best valid joint loss: %.3f' % (
                            train_joint_loss, valid_joint_loss, best_valid_joint_loss))

                    if opt.early_stop_loss == "joint":
                        previous_valid_loss = previous_valid_joint_loss
                        current_valid_loss = valid_joint_loss
                    elif opt.early_stop_loss == "ppl":
                        previous_valid_loss = previous_valid_ppl
                        current_valid_loss = valid_ppl
                    else:
                        raise ValueError

                    if epoch >= opt.start_decay_and_early_stop_at:
                        if current_valid_loss < previous_valid_loss: # update the best valid loss and save the model parameters
                            logging.info("Valid loss drops")
                            sys.stdout.flush()
                            num_stop_dropping = 0
                            check_pt_model_path = os.path.join(opt.model_path, 'ckpt', '%s-epoch-%d-total_batch-%d-%s-%.3f' % (
                                opt.exp, epoch, total_batch, opt.early_stop_loss, current_valid_loss))
                            torch.save(  # save model parameters
                                overall_model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            logging.info('Saving checkpoint to %s' % check_pt_model_path)
                        else:
                            logging.info("Valid loss does not drop")
                            sys.stdout.flush()
                            num_stop_dropping += 1
                            # decay the learning rate by a factor
                            if opt.learning_rate_decay < 1:
                                for i, param_group in enumerate(optimizer_ml.param_groups):
                                    old_lr = float(param_group['lr'])
                                    new_lr = old_lr * opt.learning_rate_decay
                                    if new_lr < opt.min_lr:
                                        new_lr = opt.min_lr
                                    if old_lr - new_lr > EPS:
                                        param_group['lr'] = new_lr
                                logging.info('Learning rate drops to {}'.format(new_lr))

                        previous_valid_joint_loss = valid_joint_loss
                        previous_valid_ppl = valid_ppl

                        best_condition1 = opt.early_stop_loss == 'joint' and valid_joint_loss < final_best_valid_joint_loss
                        best_condition2 = opt.early_stop_loss == 'ppl' and valid_ppl < final_correspond_valid_ppl
                        if best_condition1 or best_condition2:
                            best_valid_joint_loss = valid_joint_loss
                            # store other information of the checkpoint with best_valid_joint_loss
                            final_best_valid_joint_loss = valid_joint_loss
                            final_correspond_valid_ppl = valid_ppl
                            final_correspond_enc_class_loss = valid_enc_class_loss
                            final_correspond_enc_class_f1 = valid_enc_f1
                            final_correspond_dec_class_loss = valid_dec_class_loss
                            final_correspond_dec_class_f1 = valid_dec_f1
                            final_inconsist_loss = valid_inconsist_loss


                        if not opt.disable_early_stop:
                            if num_stop_dropping >= opt.early_stop_tolerance:
                                logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)
                                early_stop_flag = True
                                break

                    report_train_loss_statistics.clear()

    # logging the final validation statistics
    logging.info("final_best_valid_joint_loss: %.3f" % final_best_valid_joint_loss)
    logging.info("final_correspond_valid_ppl: %.3f" % final_correspond_valid_ppl)
    logging.info("final_correspond_enc_class_loss: %.3f" % final_correspond_enc_class_loss)
    logging.info("final_correspond_enc_class_f1: %.3f" % final_correspond_enc_class_f1)
    logging.info("final_correspond_dec_class_loss: %.3f" % final_correspond_dec_class_loss)
    logging.info("final_correspond_dec_class_f1: %.3f" % final_correspond_dec_class_f1)
    logging.info("final_inconsist_loss: %.3f" % final_inconsist_loss)

    # export the training curve
    #train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    #export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl, opt.checkpoint_interval, train_valid_curve_path)

    # Only keep the best three checkpoints
    remove_old_ckpts(opt.model_path, reverse=False)


def train_one_batch(batch, overall_model, optimizer, opt, global_step, classification_loss_func, tb_writer):
    # src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, rating, _ = batch

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
    trg_sent_2d_list = batch['tgt_sent_2d_list']
    trg = batch['tgt_tensor']
    trg_oov = batch['tgt_oov_tensor']
    trg_lens = batch['tgt_lens']
    trg_mask = batch['tgt_mask']
    rating = batch['rating_tensor']
    indices = batch['original_indices']

    """
    trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[io.SEP_WORD]
                 if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eos>
    trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
    """
    #seq2seq_model = overall_model['generator']
    #classifier_model = overall_model['classifier']
    batch_size = src.size(0)
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)

    src_sent_positions = src_sent_positions.to(opt.device)
    src_sent_mask = src_sent_mask.to(opt.device)

    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)
    rating = rating.to(opt.device)

    optimizer.zero_grad()

    start_time = time.time()

    # forward
    if overall_model.model_type == 'hre_max':
        decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, classifier_logit, classifier_attention_dist = \
            overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums, src_sent_mask)
    else:
        decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, classifier_logit, classifier_attention_dist = \
            overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, rating, src_sent_positions, src_sent_nums, src_sent_mask)

    forward_time = time_since(start_time)

    start_time = time.time()
    # compute loss for generation
    if decoder_dist is not None:
        if opt.copy_attention:  # Compute the loss using target with oov words
            generation_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                             opt.coverage_attn, coverage, seq2seq_attention_dist, opt.lambda_coverage, opt.coverage_loss)
        else:  # Compute the loss using target without oov words
            generation_loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                        opt.coverage_attn, coverage, seq2seq_attention_dist, opt.lambda_coverage, opt.coverage_loss)
    else:
        # RnnEncSingleClassifier model
        assert opt.class_loss_internal_enc_weight == 1.0
        assert opt.class_loss_weight == 1.0
        generation_loss = torch.Tensor([0.0]).to(opt.device)

    if math.isnan(generation_loss.item()):
        logging.info("global_step: %d" % global_step)
        logging.info("src")
        logging.info(src)
        logging.info(src_oov)
        logging.info(src_str_list)
        logging.info(src_lens)
        logging.info(src_mask)
        logging.info("trg")
        logging.info(trg)
        logging.info(trg_oov)
        logging.info(trg_sent_2d_list)
        logging.info(trg_lens)
        logging.info(trg_mask)
        logging.info("oov list")
        logging.info(oov_lists)
        logging.info("Decoder")
        logging.info(decoder_dist)
        logging.info(h_t)
        logging.info(seq2seq_attention_dist)
        raise ValueError("Generation loss is NaN")

    # normalize generation loss
    total_trg_tokens = sum(trg_lens)
    if opt.loss_normalization == "tokens": # use number of target tokens to normalize the loss
        generation_loss_normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
        generation_loss_normalization = batch_size
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert generation_loss_normalization > 0, 'normalization should be a positive number'
    normalized_generation_loss = generation_loss.div(generation_loss_normalization)

    # compute loss of classification
    if classifier_logit is not None:
        if isinstance(classifier_logit, tuple):
            # from multi_view_model
            enc_normalized_classification_loss = classification_loss_func(classifier_logit[0], rating)  # normalized by batch size already
            dec_normalized_classification_loss = classification_loss_func(classifier_logit[1], rating)  # normalized by batch size already
            # compute loss of inconsistency for the multi view model
            if opt.inconsistency_loss_type != "None":
                inconsistency_loss = inconsistency_loss_func(classifier_logit[0], classifier_logit[1], opt.inconsistency_loss_type, opt.detach_dec_incosist_loss)
            else:
                inconsistency_loss = torch.Tensor([0.0]).to(opt.device)
        else:
            enc_normalized_classification_loss = classification_loss_func(classifier_logit, rating)  # normalized by batch size already
            dec_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)
            inconsistency_loss = torch.Tensor([0.0]).to(opt.device)
    else:
        enc_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)
        dec_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)
        inconsistency_loss = torch.Tensor([0.0]).to(opt.device)

    total_normalized_classification_loss = opt.class_loss_internal_enc_weight * enc_normalized_classification_loss + \
                                           opt.class_loss_internal_dec_weight * dec_normalized_classification_loss

    joint_loss = opt.gen_loss_weight * normalized_generation_loss + opt.class_loss_weight * total_normalized_classification_loss + opt.inconsistency_loss_weight * inconsistency_loss

    loss_compute_time = time_since(start_time)

    start_time = time.time()
    # back propagation on the joint loss
    joint_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(overall_model.parameters(), opt.max_grad_norm)
        # grad_norm_after_clipping = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        # logging.info('clip grad (%f -> %f)' % (grad_norm_before_clipping, grad_norm_after_clipping))

    optimizer.step()

    # log each loss to tensorboard
    if tb_writer is not None:
        tb_writer.add_scalar('enc_classification_loss', enc_normalized_classification_loss.item(), global_step)
        tb_writer.add_scalar('dec_classification_loss', dec_normalized_classification_loss.item(), global_step)
        tb_writer.add_scalar('inconsistency_loss', inconsistency_loss.item(), global_step)
        tb_writer.add_scalar('total_classification_loss', total_normalized_classification_loss.item(), global_step)
        tb_writer.add_scalar('generation_loss', normalized_generation_loss.item(), global_step)
        tb_writer.add_scalar('joint_loss', joint_loss.item(), global_step)

    # construct a statistic object for the loss
    stat = JointLossStatistics(joint_loss.item(), generation_loss.item(),
                               enc_normalized_classification_loss.item(), dec_normalized_classification_loss.item(), inconsistency_loss.item(),
                               n_iterations=1, n_tokens=total_trg_tokens, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    decoder_dist_out = decoder_dist.detach() if decoder_dist is not None else None
    return stat, decoder_dist_out,