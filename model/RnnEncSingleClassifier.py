from model.rnn_encoder import *
from utils import io
from model.word_attn_classifier import WordAttnClassifier
from model.pooling_classifier import MaxPoolClassifier
from model.word_attn_no_query_classifier import WordAttnNoQueryClassifier
from model.word_multi_hop_attn_classifier import WordMultiHopAttnClassifier


class RnnEncSingleClassifier(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt, rating_tokens_tensor=None):
        """Initialize model.
        :param rating_tokens_tensor: a LongTensor, [5, rating_v_size], stores the top rating_v_size tokens' indexs of each rating score
        """
        super(RnnEncSingleClassifier, self).__init__()

        self.vocab_size = len(opt.word2idx)
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.memory_bank_size = self.num_directions * self.encoder_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dropout = opt.dropout
        self.model_type = opt.model_type

        self.pad_idx_src = io.PAD
        # self.pad_idx_trg = io.PAD
        # self.bos_idx = io.BOS
        # self.eos_idx = io.EOS
        self.unk_idx = io.UNK
        self.hr_enc = opt.encoder_type == "hre_brnn"
        if opt.encoder_type == 'sep_layers_brnn':
            self.separate_mode = 1
            self.separate_layer_enc = True
        elif opt.encoder_type == 'sep_layers_brnn_reverse':
            self.separate_mode = 2
            self.separate_layer_enc = True
        elif opt.encoder_type == 'rnn' and opt.enc_layers == 2 and opt.residual:
            self.separate_mode = 0
            self.separate_layer_enc = False
        elif opt.residual and opt.enc_layers != 2:
            raise ValueError
        else:
            self.separate_mode = -1
            self.separate_layer_enc = False
        #self.separate_layer_enc_reverse = opt.encoder_type == 'sep_layers_brnn_reverse'

        self.num_classes = opt.num_classes
        self.classifier_type = opt.classifier_type
        if opt.classifier_type == "word_attn":
            self.enc_classifier = WordAttnClassifier(opt.query_hidden_size, self.memory_bank_size, opt.num_classes, opt.attn_mode, opt.classifier_dropout, opt.ordinal, self.hr_enc)
        elif opt.classifier_type == "max":
            self.enc_classifier = MaxPoolClassifier(self.memory_bank_size, opt.num_classes, opt.classifier_dropout, opt.ordinal, self.hr_enc)
        elif opt.classifier_type == "word_attn_no_query":
            self.enc_classifier = WordAttnNoQueryClassifier(self.memory_bank_size, opt.num_classes, opt.classifier_dropout, opt.ordinal, self.hr_enc)
        elif opt.classifier_type == "word_multi_hop_attn":
            self.enc_classifier = WordMultiHopAttnClassifier(opt.query_hidden_size, self.memory_bank_size, opt.num_classes, opt.attn_mode, opt.classifier_dropout, opt.ordinal, self.hr_enc)
        else:
            raise ValueError

        if self.hr_enc:
            self.encoder = CatHirRNNEncoder(
                vocab_size=self.vocab_size,
                embed_size=self.emb_dim,
                hidden_size=self.encoder_size,
                num_layers=self.enc_layers,
                bidirectional=self.bidirectional,
                pad_token=self.pad_idx_src,
                dropout=self.dropout
            )
        elif self.separate_mode >= 0:
            self.encoder = TwoLayerRNNEncoder(
                vocab_size=self.vocab_size,
                embed_size=self.emb_dim,
                hidden_size=self.encoder_size,
                num_layers=self.enc_layers,
                bidirectional=self.bidirectional,
                pad_token=self.pad_idx_src,
                dropout=self.dropout,
                separate_mode=self.separate_mode,
                residual=opt.residual
            )
        else:
            self.encoder = RNNEncoderBasic(
                vocab_size=self.vocab_size,
                embed_size=self.emb_dim,
                hidden_size=self.encoder_size,
                num_layers=self.enc_layers,
                bidirectional=self.bidirectional,
                pad_token=self.pad_idx_src,
                dropout=self.dropout
            )

        self.init_embedding_weights()

    def init_embedding_weights(self):
        """Initialize weights."""
        init_range = 0.1
        self.encoder.embedding.weight.data.uniform_(-init_range, init_range)

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self.encoder.embedding.weight.size() == embedding.size()
        self.encoder.embedding.weight.data.copy_(embedding)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, rating, src_sent_positions, src_sent_nums, src_sent_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param rating: a LongTensor, [batch]
        :param src_sent_positions: a LongTensor containing the forward and backward ending positions of src sentences, [batch, max_sent_num, 2]
        :param src_sent_nums: a list containing the sentence number of each src, [batch]
        :param src_sent_mask: a FloatTensor, [batch, max_sent_num]
        :return:
        """
        batch_size, max_src_len = list(src.size())

        # Encoding
        memory_banks, encoder_final_states = self.encoder(src, src_lens, src_mask, src_sent_positions, src_sent_nums)
        word_memory_bank, sent_memory_bank = memory_banks
        word_encoder_final_state, sent_encoder_final_state = encoder_final_states
        src_masks = (src_mask, src_sent_mask)

        assert word_memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert word_encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # classification
        if self.separate_layer_enc:
            classifier_memory_bank = sent_memory_bank
        else:
            classifier_memory_bank = word_memory_bank

        enc_classifier_output = self.enc_classifier(classifier_memory_bank, src_mask, sent_memory_bank, src_sent_mask)
        if isinstance(enc_classifier_output, tuple):
            enc_logit = enc_classifier_output[0]
            enc_classify_attn_dist = enc_classifier_output[1][0]
            sent_enc_classify_attn_dist = enc_classifier_output[1][1]
        else:
            enc_logit = enc_classifier_output
            enc_classify_attn_dist = None
            sent_enc_classify_attn_dist = None

        logit = enc_logit
        classifier_attention_dist = ((enc_classify_attn_dist, None), sent_enc_classify_attn_dist)
        return None, None, None, word_encoder_final_state, None, logit, classifier_attention_dist

    def init_context(self, memory_bank):
        # Init by max pooling, may support other initialization later
        context, _ = memory_bank.max(dim=1)
        return context
