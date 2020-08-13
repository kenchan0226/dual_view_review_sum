from model.rnn_encoder import *
#from model.rnn_decoder import RNNDecoder
from model.hss_decoder import HssDecoder
from utils import io


class HSSSeq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(HSSSeq2SeqModel, self).__init__()

        self.vocab_size = len(opt.word2idx)
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size
        #self.ctx_hidden_dim = opt.rnn_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout

        self.bridge = opt.bridge

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = io.PAD
        self.pad_idx_trg = io.PAD
        self.bos_idx = io.BOS
        self.eos_idx = io.EOS
        self.unk_idx = io.UNK
        self.sep_idx = None
        # self.sep_idx = opt.word2idx['.']
        self.orthogonal_loss = opt.orthogonal_loss

        if self.orthogonal_loss:
            assert self.sep_idx is not None

        self.share_embeddings = opt.share_embeddings
        self.review_attn = opt.review_attn

        self.attn_mode = opt.attn_mode

        #self.goal_vector_mode = opt.goal_vector_mode
        #self.goal_vector_size = opt.goal_vector_size
        #self.manager_mode = opt.manager_mode

        '''
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            self.pad_idx_src
        )
        '''

        self.hr_enc = opt.encoder_type == 'hre_brnn'

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
        # new
        self.decoder = HssDecoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            coverage_attn=self.coverage_attn,
            copy_attn=self.copy_attn,
            review_attn=self.review_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            dropout=self.dropout,
            hr_enc=self.hr_enc
        )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = nn.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        """
        if self.separate_present_absent and self.goal_vector_mode > 0:
            if self.manager_mode == 2:  # use GRU as a manager
                self.manager = nn.GRU(input_size=self.decoder_size, hidden_size=self.goal_vector_size, num_layers=1, bidirectional=False, batch_first=False, dropout=self.dropout)
                self.bridge_manager = opt.bridge_manager
                if self.bridge_manager:
                    self.manager_bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.goal_vector_size)
                else:
                    self.manager_bridge_layer = None
            elif self.manager_mode == 1:  # use two trainable vectors only
                self.manager = ManagerBasic(self.goal_vector_size)
        """

        if self.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self.init_embedding_weights()

    def init_embedding_weights(self):
        """Initialize weights."""
        init_range = 0.1
        self.encoder.embedding.weight.data.uniform_(-init_range, init_range)
        if not self.share_embeddings:
            self.decoder.embedding.weight.data.uniform_(-init_range, init_range)

        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        #self.encoder2decoder_hidden.bias.data.fill_(0)
        #self.encoder2decoder_cell.bias.data.fill_(0)
        #self.decoder2vocab.bias.data.fill_(0)

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self.share_embeddings
        #print("encoder embedding: {}".format(self.encoder.embedding.weight.size()))
        #print("pretrained embedding: {}".format(embedding.size()))
        assert self.encoder.embedding.weight.size() == embedding.size()
        self.encoder.embedding.weight.data.copy_(embedding)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, src_sent_positions, src_sent_nums, src_sent_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param num_trgs: only effective in one2many mode 2, a list of num of targets in each batch, with len=batch_size
        :param sampled_source_representation_2dlist: only effective when using target encoder, a 2dlist of tensor with dim=[memory_bank_size]
        :param source_representation_target_list: a list that store the index of ground truth source representation for each batch, dim=[batch_size]
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

        # Decoding
        h_t_init = self.init_decoder_state(word_encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)
        #context = self.init_context(memory_bank)  # [batch, memory_bank_size]

        decoder_dist_all = []
        attention_dist_all = []
        # new
        sentiment_context_all = []

        if self.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float).requires_grad_()  # [batch, max_src_seq]
            #coverage_all = coverage.new_zeros((max_target_length, batch_size, max_src_len), dtype=torch.float)  # [max_trg_len, batch_size, max_src_len]
            coverage_all = []
        else:
            coverage = None
            coverage_all = None

        if self.review_attn:
            decoder_memory_bank = h_t_init[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            assert decoder_memory_bank.size() == torch.Size([batch_size, 1, self.decoder_size])
        else:
            decoder_memory_bank = None

        if self.orthogonal_loss:  # create a list of batch_size empty list
            delimiter_decoder_states_2dlist = [[] for i in range(batch_size)]

        # init y_t to be BOS token
        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]

        for t in range(max_target_length):
            # determine the hidden state that will be feed into the next step
            # according to the time step or the target input
            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next

            if self.review_attn and t > 0:
                decoder_memory_bank = torch.cat([decoder_memory_bank, h_t[-1, :, :].unsqueeze(1)], dim=1)  # [batch, t+1, decoder_size]

            # new, attn_dist: word level, coverage: word level
            decoder_dist, h_t_next, _, sentiment_context, attn_dist, p_gen, coverage = \
                self.decoder(y_t, h_t, memory_banks, src_masks, max_num_oov, src_oov, coverage, decoder_memory_bank)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            # new
            sentiment_context_all.append(sentiment_context)  # [batch_size, memory_bank_size]
            y_t_next = trg[:, t]  # [batch]

            # if this hidden state corresponds to the delimiter, stack it
            if self.orthogonal_loss:
                for i in range(batch_size):
                    if y_t_next[i].item() == self.sep_idx:
                        delimiter_decoder_states_2dlist[i].append(h_t_next[-1, i, :])  # [decoder_size]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
        # new
        sentiment_context_all = torch.stack(sentiment_context_all, dim=1)  # [batch_size, trg_len, memory_bank_size]
        assert sentiment_context_all.size() == torch.Size((batch_size, max_target_length, self.num_directions * self.encoder_size))

        if self.coverage_attn:
            coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
            assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        # Pad delimiter_decoder_states_2dlist with zero vectors
        if self.orthogonal_loss:
            assert len(delimiter_decoder_states_2dlist) == batch_size
            delimiter_decoder_states_lens = [len(delimiter_decoder_states_2dlist[i]) for i in range(batch_size)]
            # [batch_size, decoder_size, max_num_delimiters]
            delimiter_decoder_states = self.tensor_2dlist_to_tensor(delimiter_decoder_states_2dlist, batch_size, self.decoder_size, delimiter_decoder_states_lens)
        else:
            delimiter_decoder_states_lens = None
            delimiter_decoder_states = None

        # new
        return decoder_dist_all, h_t_next, attention_dist_all, word_encoder_final_state, coverage_all, sentiment_context_all, word_memory_bank

    def tensor_2dlist_to_tensor(self, tensor_2d_list, batch_size, hidden_size, seq_lens):
        """
        :param tensor_2d_list: a 2d list of tensor with size=[hidden_size], len(tensor_2d_list)=batch_size, len(tensor_2d_list[i])=seq_len[i]
        :param batch_size:
        :param hidden_size:
        :param seq_lens: a list that store the seq len of each batch, with len=batch_size
        :return: [batch_size, hidden_size, max_seq_len]
        """
        # assert tensor_2d_list[0][0].size() == torch.Size([hidden_size])
        max_seq_len = max(seq_lens)
        for i in range(batch_size):
            for j in range(max_seq_len - seq_lens[i]):
                tensor_2d_list[i].append( torch.ones(hidden_size).to(self.device) * self.pad_idx_trg )  # [hidden_size]
            tensor_2d_list[i] = torch.stack(tensor_2d_list[i], dim=1)  # [hidden_size, max_seq_len]
        tensor_3d = torch.stack(tensor_2d_list, dim=0)  # [batch_size, hidden_size, max_seq_len]
        return tensor_3d

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state

    def init_context(self, memory_bank):
        # Init by max pooling, may support other initialization later
        context, _ = memory_bank.max(dim=1)
        return context
