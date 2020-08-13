import torch
from utils.masked_loss import masked_cross_entropy
from utils.statistics import RewardStatistics, JointLossStatistics
import time
from utils.time_log import time_since
#from utils.reward import sample_list_to_str_list, compute_batch_reward
from utils import io
import nltk
from cytoolz import concat
from torch.nn import CrossEntropyLoss
from utils.utils_glue import acc_and_f1, acc_and_macro_f1
import numpy as np
from utils.ordinal_utilities import binary_results_to_rating_preds
from utils.inconsistency_loss import inconsistency_loss_func


def evaluate_loss(data_loader, overall_model, classification_loss_func, opt, print_incon_stats=False):
    overall_model.eval()
    generation_loss_sum = 0.0
    joint_loss_sum = 0.0
    classification_loss_sum = 0.0
    enc_classification_loss_sum = 0.0
    dec_classification_loss_sum = 0.0
    inconsist_loss_sum = 0.0
    total_trg_tokens = 0
    total_num_iterations = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    enc_rating_preds = None
    dec_rating_preds = None
    incon_loss_preds = None

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
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

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            total_num_iterations += 1
            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_sent_positions = src_sent_positions.to(opt.device)
            src_sent_mask = src_sent_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)
            rating = rating.to(opt.device)

            start_time = time.time()

            # forward
            if overall_model.model_type == 'hre_max':
                decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, classifier_logit, classifier_attention_dist = \
                    overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums, src_sent_mask)
            else:
                decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage, classifier_logit, classifier_attention_dist = overall_model(
                    src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, rating, src_sent_positions, src_sent_nums, src_sent_mask)

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if decoder_dist is not None:
                if opt.copy_attention:  # Compute the loss using target with oov words
                    generation_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                                opt.coverage_attn, coverage, seq2seq_attention_dist, opt.lambda_coverage, coverage_loss=False)
                else:  # Compute the loss using target without oov words
                    generation_loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                                opt.coverage_attn, coverage, seq2seq_attention_dist, opt.lambda_coverage, coverage_loss=False)
            else:
                generation_loss = torch.Tensor([0.0]).to(opt.device)

            # normalize generation loss
            num_trg_tokens = sum(trg_lens)
            normalized_generation_loss = generation_loss.div(num_trg_tokens)

            # compute loss of classification
            if print_incon_stats:
                assert isinstance(classifier_logit, tuple)

            if classifier_logit is not None:
                if isinstance(classifier_logit, tuple):
                    # from multi_view_model
                    enc_classifier_logit = classifier_logit[0]
                    dec_classifier_logit = classifier_logit[1]
                    enc_normalized_classification_loss = classification_loss_func(classifier_logit[0], rating)  # normalized by batch size already
                    dec_normalized_classification_loss = classification_loss_func(classifier_logit[1], rating)  # normalized by batch size already
                    # compute loss of inconsistency for the multi view model
                    if opt.inconsistency_loss_type != "None" or print_incon_stats:
                        inconsistency_loss = inconsistency_loss_func(classifier_logit[0], classifier_logit[1],
                                                                     opt.inconsistency_loss_type, opt.detach_dec_incosist_loss)
                    else:
                        inconsistency_loss = torch.Tensor([0.0]).to(opt.device)
                else:
                    enc_classifier_logit = classifier_logit
                    dec_classifier_logit = None
                    enc_normalized_classification_loss = classification_loss_func(classifier_logit, rating)  # normalized by batch size already
                    dec_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)
                    inconsistency_loss = torch.Tensor([0.0]).to(opt.device)
            else:
                enc_classifier_logit = None
                dec_classifier_logit = None
                enc_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)
                dec_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)
                inconsistency_loss = torch.Tensor([0.0]).to(opt.device)

            total_normalized_classification_loss = opt.class_loss_internal_enc_weight * enc_normalized_classification_loss + \
                                                   opt.class_loss_internal_dec_weight * dec_normalized_classification_loss

            # compute validation performance
            if enc_rating_preds is None and dec_rating_preds is None:
                if opt.ordinal:
                    enc_rating_preds = binary_results_to_rating_preds(enc_classifier_logit.detach().cpu().numpy()) if enc_classifier_logit is not None else None
                    dec_rating_preds = binary_results_to_rating_preds(dec_classifier_logit.detach().cpu().numpy()) if dec_classifier_logit is not None else None
                else:
                    enc_rating_preds = enc_classifier_logit.detach().cpu().numpy() if enc_classifier_logit is not None else None
                    dec_rating_preds = dec_classifier_logit.detach().cpu().numpy() if dec_classifier_logit is not None else None
                    # if print_incon_stats:
                    #     incon_loss_preds = inconsistency_loss.detach().cpu().numpy()
                out_label_ids = rating.detach().cpu().numpy()
            else:
                if opt.ordinal:
                    enc_rating_preds = np.append(enc_rating_preds, binary_results_to_rating_preds(enc_classifier_logit.detach().cpu().numpy()), axis=0) if enc_classifier_logit is not None else None
                    dec_rating_preds = np.append(dec_rating_preds, binary_results_to_rating_preds(dec_classifier_logit.detach().cpu().numpy()), axis=0) if dec_classifier_logit is not None else None
                else:
                    enc_rating_preds = np.append(enc_rating_preds, enc_classifier_logit.detach().cpu().numpy(), axis=0) if enc_classifier_logit is not None else None
                    dec_rating_preds = np.append(dec_rating_preds, dec_classifier_logit.detach().cpu().numpy(), axis=0) if dec_classifier_logit is not None else None
                    # if print_incon_stats:
                    #     incon_loss_preds = np.append(incon_loss_preds, inconsistency_loss.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, rating.detach().cpu().numpy(), axis=0)
            # joint loss
            joint_loss = opt.gen_loss_weight * normalized_generation_loss + opt.class_loss_weight * total_normalized_classification_loss + opt.inconsistency_loss_weight * inconsistency_loss

            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            classification_loss_sum += total_normalized_classification_loss
            enc_classification_loss_sum += enc_normalized_classification_loss
            dec_classification_loss_sum += dec_normalized_classification_loss
            inconsist_loss_sum += inconsistency_loss
            joint_loss_sum += joint_loss.item()
            generation_loss_sum += generation_loss.item()
            total_trg_tokens += num_trg_tokens

    if not opt.ordinal:
        # merged preds
        if enc_rating_preds is not None and dec_rating_preds is not None:
            merged_rating_preds = (enc_rating_preds + dec_rating_preds) / 2
            merged_rating_preds = np.argmax(merged_rating_preds, axis=1)
        else:
            merged_rating_preds = None
        enc_rating_preds = np.argmax(enc_rating_preds, axis=1) if enc_rating_preds is not None else None
        dec_rating_preds = np.argmax(dec_rating_preds, axis=1) if dec_rating_preds is not None else None

    if print_incon_stats:
        inconsistency_statistics(out_label_ids, enc_rating_preds, dec_rating_preds, merged_rating_preds)

    enc_classification_result = acc_and_macro_f1(enc_rating_preds, out_label_ids) if enc_rating_preds is not None else {"acc": 0.0, "f1": 0.0, "acc_and_f1": 0.0}
    dec_classification_result = acc_and_macro_f1(dec_rating_preds, out_label_ids) if dec_rating_preds is not None else None
    loss_stat = JointLossStatistics(joint_loss_sum, generation_loss_sum, enc_classification_loss_sum, dec_classification_loss_sum, inconsist_loss_sum, total_num_iterations, total_trg_tokens, forward_time=forward_time_total, loss_compute_time=loss_compute_time_total)
    # joint_loss=0.0, generation_loss=0.0, classification_loss=0.0, n_iterations=0, n_tokens=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0

    return loss_stat, (enc_classification_result, dec_classification_result)


def inconsistency_statistics(out_label_ids, enc_rating_preds, dec_rating_preds, merged_rating_preds):
    """
    Calculate inconsistency statistics between the source view and summary view classifiers.
    :param out_label_ids: [data_num], int
    :param enc_rating_preds: [data_num], int
    :param dec_rating_preds: [data_num], int
    :param merged_rating_preds: [data_num], int
    :return: print out the computed inconsistency stats
    """
    data_num = out_label_ids.shape[0]
    assert out_label_ids.ndim == 1
    assert enc_rating_preds.ndim == 1 and enc_rating_preds.shape[0] == data_num
    assert dec_rating_preds.ndim == 1 and dec_rating_preds.shape[0] == data_num
    # assert incon_loss_preds.ndim == 1 and incon_loss_preds.shape[0] == data_num

    total_incon_cnt = 0
    correct_incon_cnt = 0
    enc_correct_incon_cnt = 0
    dec_correct_incon_cnt = 0
    for i in range(data_num):
        if enc_rating_preds[i] != dec_rating_preds[i]:
            total_incon_cnt += 1
            if enc_rating_preds[i] == out_label_ids[i]:
                enc_correct_incon_cnt += 1
                correct_incon_cnt += 1
            elif dec_rating_preds[i] == out_label_ids[i]:
                dec_correct_incon_cnt += 1
                correct_incon_cnt += 1

    total_incon_ratio = round(total_incon_cnt / data_num, 5)
    correct_incon_ratio = round(correct_incon_cnt / data_num, 5)
    enc_correct_incon_ratio = round(enc_correct_incon_cnt / data_num, 5)
    dec_correct_incon_ratio = round(dec_correct_incon_cnt / data_num, 5)

    print("Total data num: {}".format(data_num))
    print("Total inconsistent num: {}, ratio: {}".format(total_incon_cnt, total_incon_ratio))
    print("Correct inconsistent num: {}, ratio: {}".format(correct_incon_cnt, correct_incon_ratio))
    print("Enc correct inconsistent num: {}, ratio: {}".format(enc_correct_incon_cnt, enc_correct_incon_ratio))
    print("Dec correct inconsistent num: {}, ratio: {}".format(dec_correct_incon_cnt, dec_correct_incon_ratio))

    enc_results = acc_and_macro_f1(enc_rating_preds, out_label_ids)
    dec_results = acc_and_macro_f1(dec_rating_preds, out_label_ids)
    merged_results = acc_and_macro_f1(merged_rating_preds, out_label_ids)
    print("\nEnc Macro F1: {:.5f}".format(enc_results['f1']))
    print("Dec Macro F1: {:.5f}".format(dec_results['f1']))
    print("Merged Macro F1: {:.5f}".format(merged_results['f1']))


'''
def evaluate_reward(data_loader, generator, opt):
    """Return the avg. reward in the validation dataset"""
    generator.model.eval()
    final_reward_sum = 0.0
    n_batch = 0
    sample_time_total = 0.0
    reward_type = opt.reward_type

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            # load one2many dataset
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, _ = batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            start_time = time.time()
            # sample a sequence
            # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
            sample_list, log_selected_token_dist, output_mask, _, _, _ = generator.sample(
                src, src_lens, src_oov, src_mask, oov_lists, greedy=True, entropy_regularize=False)

            pred_str_list = sample_list_to_str_list(sample_list, oov_lists, opt.idx2word, opt.vocab_size, io.EOS,
                                                        io.UNK, opt.replace_unk,
                                                        src_str_list)
            sample_time = time_since(start_time)
            sample_time_total += sample_time

            pred_sent_2d_list = []  # each item is a list of predicted sentences (tokenized) for an input sample, used to compute summary level Rouge-l
            trg_sent_2d_list_tokenized = []  # each item is a list of target sentences (tokenized) for an input sample
            trg_str_list = []  # each item is the target output sequence (tokenized) for an input sample
            for pred_str, trg_sent_list in zip(pred_str_list, trg_sent_2d_list):
                pred_sent_list = nltk.tokenize.sent_tokenize(' '.join(pred_str))
                pred_sent_list = [pred_sent.strip().split(' ') for pred_sent in pred_sent_list]
                pred_sent_2d_list.append(pred_sent_list)

                trg_sent_list = [trg_sent.strip().split(' ') for trg_sent in trg_sent_list]
                trg_sent_2d_list_tokenized.append(trg_sent_list)
                trg_str_list.append(list(concat(trg_sent_list)))

            trg_sent_2d_list = trg_sent_2d_list_tokenized

            final_reward = compute_batch_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, reward_type, regularization_type = 0)  # np.array, [batch_size]

            final_reward_sum += final_reward.detach().sum(0).item()

    eval_reward_stat = RewardStatistics(final_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)

    return eval_reward_stat
'''
