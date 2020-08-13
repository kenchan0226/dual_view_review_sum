import logging
import os
import sys
import time


def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger

def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=128,
                        help='Word embedding for both.')

    #parser.add_argument('-position_encoding', action='store_true',
    #                    help='Use a sin to mark relative words positions.')
    parser.add_argument('-share_embeddings', default=True, action='store_true',
                        help="""Share the word embeddings between encoder
                         and decoder.""")

    parser.add_argument('-v_size', type=int, default=50000,
                        help="Size of the vocabulary (excluding the special tokens)")
    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='rnn',
                        choices=['rnn', 'hre_brnn', 'sep_layers_brnn', 'sep_layers_brnn_reverse'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-encoder_size', type=int, default=256,
                        help='Size of encoder hidden states')
    parser.add_argument('-decoder_size', type=int, default=512,
                        help='Size of decoder hidden states')
    parser.add_argument('-dropout', type=float, default=0.1,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('-classifier_dropout', type=float, default=0.1,
                        help="Dropout probability applied in classifier.")
    parser.add_argument('-residual', action='store_true', default=False,
                        help="Use residual connection in encoder.")
    # parser.add_argument('-input_feed', type=int, default=1,
    #                     help="""Feed the context vector at each time step as
    #                     additional input (via concatenation with the word
    #                     embeddings) to the decoder.""")

    #parser.add_argument('-rnn_type', type=str, default='GRU',
    #                    choices=['LSTM', 'GRU'],
    #                    help="""The gate type to use in the RNNs""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    #parser.add_argument('-input_feeding', action="store_true",
    #                    help="Apply input feeding or not. Feed the updated hidden vector (after attention)"
    #                         "as new hidden vector to the decoder (Luong et al. 2015). "
    #                         "Feed the context vector at each time step  after normal attention"
    #                         "as additional input (via concatenation with the word"
    #                         "embeddings) to the decoder.")

    parser.add_argument('-bidirectional', default=True,
                        action = "store_true",
                        help="whether the encoder is bidirectional")

    parser.add_argument('-bridge', type=str, default='copy',
                        choices=['copy', 'dense', 'dense_nonlinear', 'none'],
                        help="An additional layer between the encoder and the decoder")

    # Attention options
    parser.add_argument('-attn_mode', type=str, default='concat',
                       choices=['general', 'concat'],
                       help="""The attention type to use:
                       dot or general (Luong) or concat (Bahdanau)""")
    #parser.add_argument('-attention_mode', type=str, default='concat',
    #                    choices=['dot', 'general', 'concat'],
    #                    help="""The attention type to use:
    #                    dot or general (Luong) or concat (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('-copy_attention', action="store_true",
                        help='Train a copy model.')

    #parser.add_argument('-copy_mode', type=str, default='concat',
    #                    choices=['dot', 'general', 'concat'],
    #                    help="""The attention type to use: dot or general (Luong) or concat (Bahdanau)""")

    #parser.add_argument('-copy_input_feeding', action="store_true",
    #                    help="Feed the context vector at each time step after copy attention"
    #                         "as additional input (via concatenation with the word"
    #                         "embeddings) to the decoder.")

    #parser.add_argument('-reuse_copy_attn', action="store_true",
    #                   help="Reuse standard attention for copy (see See et al.)")

    #parser.add_argument('-copy_gate', action="store_true",
    #                    help="A gate controling the flow from generative model and copy model (see See et al.)")

    parser.add_argument('-coverage_attn', action="store_true",
                        help='Train a coverage attention layer.')
    parser.add_argument('-review_attn', action="store_true",
                        help='Train a review attention layer')

    parser.add_argument('-lambda_coverage', type=float, default=1.0,
                        help='Lambda value for coverage by See et al.')
    parser.add_argument('-coverage_loss', action="store_true", default=False,
                        help='whether to include coverage loss')
    parser.add_argument('-orthogonal_loss', action="store_true", default=False,
                        help='whether to include orthogonal loss')
    parser.add_argument('-lambda_orthogonal', type=float, default=0.03,
                        help='Lambda value for the orthogonal loss by Yuan et al.')
    parser.add_argument('-model_type', type=str, default="", choices=["hss", "multi_task_basic", "hre_max", "word_attn_modulate", "external_feed", "external_soft_feed",
                                                                         "multi_view_ex_soft_feed", "multi_view_attn_modulate", "multi_view_multi_task_basic", "rnn_enc_single_classifier", "seq2seq"],
                        help='Type of model')
    parser.add_argument('-classifier_type', type=str, default="max", choices=["max", "word_attn", "word_attn_no_query", "word_multi_hop_attn"],
                        help='Type of model')
    parser.add_argument('-dec_classifier_type', type=str, default="max",
                        choices=["max", "word_attn", "word_attn_no_query", "word_multi_hop_attn"],
                        help='Type of the classifier on the decoder side')
    parser.add_argument('-dec_classify_input_type', type=str, default='dec_state', choices=['attn_vec', 'dec_state'])

    # rating topk vocab options
    parser.add_argument('-rating_v_size', type=int, default=200,
                        help="Size of the rating vocabulary")
    parser.add_argument('-rating_memory_pred', action='store_true',
                        help="""Add the attended context vector from rating memory bank 
                            to the final softmax layer when predicting next token""")
    parser.add_argument('-rating_memory_type', type=str, default='gold',
                        choices=['gold', 'pred'],
                        help="""Type of rating memory to use.""")
    parser.add_argument('-rating_bridge_type', type=str, default="None",
                        choices=["None", "relu_one_layer", "tanh_one_layer"],
                        help='Type of rating bridge')

    parser.add_argument('-query_hidden_size', type=int, default=512,
                        help='Size of encoder hidden states')
    parser.add_argument('-detach_enc_logit_for_soft_feed', action="store_true",
                        help='Detach the enc_logit when doing external soft feeding.')
    #parser.add_argument('-manager_mode', type=int, default=1, choices=[1],
    #                    help='Only effective in separate_present_absent. 1: two trainable vectors as the goal vectors;')
    #parser.add_argument('-goal_vector_size', type=int, default=16,
    #                    help='size of goal vector')
    #parser.add_argument('-goal_vector_mode', type=int, default=0, choices=[0, 1, 2],
    #                    help='Only effective in separate_present_absent. 0: no goal vector; 1: goal vector act as an extra input to the decoder; 2: goal vector act as an extra input to p_gen')


def train_ml_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the train and val folder""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    #parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
    #                    help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-delimiter', type=str, default='.',
                        help='Delimiter for orthogonal regularization')
    parser.add_argument('-src_max_len', type=int, default=400, help='The maximum number of words in source.')
    parser.add_argument('-trg_max_len', type=int, default=100, help='The maximum number of words in target.')

    # Init options
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')

    # Pretrained word vectors
    parser.add_argument('-w2v', type=str, help="""use pretrained word2vec word embedding""")
    # Fixed word vectors
    """
    parser.add_argument('-fix_word_vecs_enc',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-fix_word_vecs_dec',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    """
    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')
    #parser.add_argument('-optim', default='adam',
    #                    choices=['sgd', 'adagrad', 'adadelta', 'adam'],
    #                    help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=2,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-loss_normalization', default="tokens", choices=['tokens', 'batches'],
                        help="Normalize the cross-entropy loss by the number of tokens or batch size")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-min_lr', type=float, default=1e-5,
                        help="""minimum learning rate""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_checkpoint_at', type=int, default=2,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-start_decay_and_early_stop_at', type=int, default=2,
                        help="""Start learning rate decay and check for early stopping after 
                        and including this epoch""")
    parser.add_argument('-checkpoint_interval', type=int, default=1000,
                        help='Run validation and save model parameters at this interval.')
    parser.add_argument('-disable_early_stop', action="store_true", default=False,
                        help="A flag to disable early stopping in rl training.")
    parser.add_argument('-early_stop_tolerance', type=int, default=4,
                        help="Stop training if it doesn't improve any more for several rounds of validation")
    # export options
    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")
    parser.add_argument('-exp', type=str, default="cnn-dm",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")
    parser.add_argument('-model_path', type=str, default="/research/king3/ik_grp/wchen/senti_summ_models/saved_model/%s.%s",
                        help="Path to save model checkpoints.")
    # classification options
    parser.add_argument('-num_classes', type=int, default=5,
                        help='Number of classes of rating')
    parser.add_argument('-inconsistency_loss_type', type=str, default="None",
                        choices=["None", "cross_entropy", "KL_div"],
                        help='Type of the inconsistency loss between the enc and dec classifiers')
    parser.add_argument('-detach_dec_incosist_loss', action="store_true",
                        help='Detach the dec_logit when calculating the inconsistency loss.')
    parser.add_argument('-detach_classify_dec_states', action="store_true",
                        help='Detach the decoder states when calculating the dec classification loss.')
    parser.add_argument('-inconsistency_loss_weight', type=float, default=1.0,
                        help='Coefficient for the inconsistency loss for multi view model')
    parser.add_argument('-gen_loss_weight', type=float, default=1.0,
                        help='Coefficient for the cross entropy loss for generative model')
    parser.add_argument('-class_loss_weight', type=float, default=1.0,
                        help='Coefficient for the cross entropy loss for total classification loss')
    parser.add_argument('-class_loss_internal_enc_weight', type=float, default=1.0,
                        help='Coefficient for the cross entropy loss for encoder classification model')
    parser.add_argument('-class_loss_internal_dec_weight', type=float, default=1.0,
                        help='Coefficient for the cross entropy loss for decoder classification model')
    parser.add_argument('-weighted_sampling', action="store_true",
                        help='Sampling data according to class distribution to address imbalance learning problem.')
    parser.add_argument('-weighted_classifier_loss', action="store_true",
                        help='Apply weights to classification loss according to class distribution to address imbalance learning problem.')
    parser.add_argument('-classifier_loss_type', type=str, default="xe", choices=['xe', 'ordinal_xe', 'ordinal_mse'],
                        help="Loss for classifier. xe: cross-entropy. ")
    parser.add_argument('-early_stop_loss', type=str, default="joint", choices=['joint', 'ppl'],
                        help="Loss for early stoppping. ")
    # logging options
    parser.add_argument('-stdout', action="store_true",
                        help='Whether to output the logs to stdout')


def train_rl_opts(parser):
    # Model loading/saving options
    parser.add_argument('-pretrained_model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the train and val folder""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                            path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    # parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
    #                    help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                            reproducibility.""")
    parser.add_argument('-delimiter', type=str, default='.',
                        help='Delimiter for orthogonal regularization')
    parser.add_argument('-src_max_len', type=int, default=400, help='The maximum number of words in source.')
    parser.add_argument('-trg_max_len', type=int, default=-1, help='The maximum number of words in target. Not truncate target by default')

    # Init options
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')
    # parser.add_argument('-optim', default='adam',
    #                    choices=['sgd', 'adagrad', 'adadelta', 'adam'],
    #                    help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=2,
                        help="""If the norm of the gradient vector exceeds this,
                            renormalize it to have the norm equal to
                            max_grad_norm""")

    # Reinforcement Learning options
    parser.add_argument('-regularization_type', type=int, default=0, choices=[0, 1, 2],
                        help='0: no regularization, 1: percentage of unique keyphrases, 2: entropy')
    parser.add_argument('-regularization_factor', type=float, default=0.0,
                        help="Factor of regularization")
    parser.add_argument('-replace_unk', action="store_true",
                        help='Replace the unk token with the token of highest attention score.')
    parser.add_argument('-max_sample_length', default=6, type=int,
                        help="The max length of sequence that can be sampled by the model")
    parser.add_argument('-max_greedy_length', default=6, type=int,
                        help="The max length of sequence that can be sampled by the model")
    parser.add_argument('-pred_max_len', type=int, default=120, help='Maximum prediction length.')
    parser.add_argument('-reward_type', default='0', type=int,
                        choices=[0, 1],
                        help="""Type of reward. 0: weighted sum of ROUGE-1, ROUGE-2, and ROUGE-L. 1: ROUGE-L """)
    parser.add_argument('-baseline', default="self", choices=["none", "self"],
                        help="The baseline in RL training. none: no baseline; self: use greedy decoding as baseline")
    #parser.add_argument('-mc_rollouts', action="store_true", default=False,
    #                    help="Use Monte Carlo rollouts to estimate q value. Not support yet.")
    #parser.add_argument('-num_rollouts', type=int, default=3,
    #                    help="The number of Monte Carlo rollouts. Only effective when mc_rollouts is True. Not supported yet")
    parser.add_argument('-n_sample', type=int, default=1,
                        help="The number of samples to draw for each input sequence. ")
    parser.add_argument('-loss_normalization', default="none", choices=['none', 'batches', 'samples'],
                        help="Normalization of the policy gradient loss.")
    parser.add_argument('-sent_level_reward', action="store_true",
                        help="Use sentence level reward")
    parser.add_argument('-constrained_mdp', action="store_true",
                        help="Use constrained mdp")
    parser.add_argument('-cost_types', nargs='+', default=[], type=int,
                        help=""" Specify a list of cost function. 
                        Type of cost function. 0: number of 3-gram repeat. Only effective when using constrained mdp.""")
    parser.add_argument('-cost_thresholds', nargs='+', default=[], type=float,
                        help=""" Specify a list of thresholds. Only effective when using constrained mdp.""")
    parser.add_argument('-lagrangian_init_val', type=float, default=0.0,
                        help="The initial value of the lagrangian multiplier. ")
    parser.add_argument('-use_lagrangian_hinge_loss', action="store_true",
                        help="Use hinge loss in lagrangian. ")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.00005,
                        help="""Starting learning rate of policy.""")
    parser.add_argument('-learning_rate_multiplier', type=float, default=0.00001,
                        help="""Starting learning rate of lagrangian multiplier.""")
    parser.add_argument('-min_lr', type=float, default=1e-5,
                        help="""minimum learning rate""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                            this much if (i) perplexity does not decrease on the
                            validation set or (ii) epoch has gone past
                            start_decay_at""")
    parser.add_argument('-start_checkpoint_at', type=int, default=2,
                        help="""Start checkpointing every epoch after and including
                            this epoch""")
    parser.add_argument('-start_decay_and_early_stop_at', type=int, default=2,
                        help="""Start learning rate decay and check for early stopping after 
                            and including this epoch""")
    parser.add_argument('-checkpoint_interval', type=int, default=4000,
                        help='Run validation and save model parameters at this interval.')
    parser.add_argument('-disable_early_stop', action="store_true", default=False,
                        help="A flag to disable early stopping in rl training.")
    parser.add_argument('-early_stop_tolerance', type=int, default=4,
                        help="Stop training if it doesn't improve any more for several rounds of validation")
    parser.add_argument('-decay_multiplier_learning_rate', action="store_true",
                        help="Decay the learning rate of lagrangian multiplier. ")

    # export options
    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")
    parser.add_argument('-model_path', type=str, default="saved_model/%s.%s",
                        help="Path to save model checkpoints.")


def predict_opts(parser):
    parser.add_argument('-pretrained_model', required=True,
                       help='Path to model .pt file')
    parser.add_argument('-teacher_force_evaluate', action="store_true", help="Do teacher forced evaluation on the validation dataset")
    parser.add_argument('-attn_debug', action="store_true", help="Whether to print attn for each word")
    #parser.add_argument('-src_file', required=True,
    #                    help="""Path to source file""")
    #parser.add_argument('-trg_file', required=True,
    #                    help="""Path to target file""")
    parser.add_argument('-data', required=True,
                        help="""Path prefix to test folder""")
    parser.add_argument('-split', default='test',
                        help="""Which split to decode""")
    parser.add_argument('-beam_size', type=int, default=5,
                       help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help='Pick the top n_best sequences from beam_search, if n_best < 0, then n_best=beam_size')
    parser.add_argument('-src_max_len', type=int, default=-1, help='The maximum number of words in source.')
    parser.add_argument('-pred_max_len', type=int, default=120,
                       help='Maximum prediction length.')
    parser.add_argument('-length_penalty_factor', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
    parser.add_argument('-coverage_penalty_factor', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    parser.add_argument('-length_penalty', default='none', choices=['none', 'wu', 'avg'],
    help="""Length Penalty to use.""")
    parser.add_argument('-coverage_penalty', default='none', choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                            reproducibility.""")
    parser.add_argument('-batch_size', type=int, default=8,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")

    parser.add_argument('-include_attn_dist', action="store_true",
                        help="Whether to return the attention distribution, for the visualization of the attention weights, haven't implemented")

    parser.add_argument('-pred_path', type=str, required=True,
                        help="Path of outputs of predictions.")
    parser.add_argument('-pred_file_prefix', type=str, default="",
                        help="Prefix of prediction file.")
    parser.add_argument('-exp', type=str, default="cnn-dm",
                        help="Name of the experiment for logging.")
    parser.add_argument('-delimiter', type=str, default='.',
                        help='Delimiter for orthogonal regularization')
    parser.add_argument('-max_eos_per_output_seq', type=int, default=1,  # max_eos_per_seq
                        help='Specify the max number of eos in one output sequences to control the number of keyphrases in one output sequence. Only effective when one2many_mode=3 or one2many_mode=2.')
    parser.add_argument('-sampling', action="store_true",
                        help='Use sampling instead of beam search to generate the predictions.')
    parser.add_argument('-replace_unk', action="store_true",
                            help='Replace the unk token with the token of highest attention score.')
    parser.add_argument('-block_ngram_repeat', type=int, default=0,
                        help='Block repeat of n-gram')
    parser.add_argument('-ignore_when_blocking', nargs='+', type=str,
                       default=[], help="""Ignore these strings when blocking repeats. You want to block sentence delimiters.""")

