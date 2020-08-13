import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
    Base class for rnn encoder
    """
    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        raise NotImplementedError


class RNNEncoderBasic(RNNEncoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoderBasic, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        #self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        # Debug
        #if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        src_embed = self.embedding(src) # [batch, src_len, embed_size]
        #src_embed_np = src_embed.detach().cpu().numpy()[:, 0, :]

        # sort src_embed according to its length
        batch_size = src.size(0)
        assert len(src_lens) == batch_size
        sort_ind = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens_sorted = [src_lens[i] for i in sort_ind]
        src_embed = reorder_sequence(src_embed, sort_ind, batch_first=True)
        #src_embed_sorted_np = src_embed.detach().cpu().numpy()[:, 0, :]

        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens_sorted, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # restore the order of memory_bank
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        memory_bank = reorder_sequence(memory_bank, reorder_ind, batch_first=True)
        encoder_final_state = reorder_gru_states(encoder_final_state, reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1,:,:], encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :] # [batch, hidden_size]

        return (memory_bank.contiguous(), None), (encoder_last_layer_final_state, None)


class TwoLayerRNNEncoder(RNNEncoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0, separate_mode=1, residual=False):
        super(TwoLayerRNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        #self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.memory_bank_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.first_layer_rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=1,
            bidirectional=bidirectional, batch_first=True)
        self.second_layer_rnn = nn.GRU(input_size=self.memory_bank_size, hidden_size=hidden_size, num_layers=1,
            bidirectional=bidirectional, batch_first=True)
        #self.reverse_output = reverse
        self.separate_mode = separate_mode  # [0: no separate, 1: top for gen, low for class, 2: top for class, low for gen]
        self.residual = residual

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        # Debug
        #if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        src_embed = self.embedding(src) # [batch, src_len, embed_size]
        #src_embed_np = src_embed.detach().cpu().numpy()[:, 0, :]

        # sort src_embed according to its length
        batch_size = src.size(0)
        assert len(src_lens) == batch_size
        sort_ind = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens_sorted = [src_lens[i] for i in sort_ind]
        src_embed = reorder_sequence(src_embed, sort_ind, batch_first=True)
        #src_embed_sorted_np = src_embed.detach().cpu().numpy()[:, 0, :]

        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens_sorted, batch_first=True)
        """        
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)
        """
        first_layer_memory_bank, first_layer_encoder_final_state = self.first_layer_rnn(packed_input_src)
        first_layer_memory_bank, _ = nn.utils.rnn.pad_packed_sequence(first_layer_memory_bank, batch_first=True)
        dropped_first_layer_memory_bank = self.dropout_layer(first_layer_memory_bank)
        dropped_first_layer_memory_bank = nn.utils.rnn.pack_padded_sequence(dropped_first_layer_memory_bank, src_lens_sorted, batch_first=True)
        second_layer_memory_bank, second_layer_encoder_final_state = self.second_layer_rnn(dropped_first_layer_memory_bank)
        second_layer_memory_bank, _ = nn.utils.rnn.pad_packed_sequence(second_layer_memory_bank, batch_first=True)

        # residual connection
        if self.residual:
            second_layer_memory_bank = 0.5 * second_layer_memory_bank + 0.5 * first_layer_memory_bank
            second_layer_encoder_final_state = 0.5 * second_layer_encoder_final_state + 0.5 * first_layer_encoder_final_state

        # restore the order of memory_bank
        """
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        memory_bank = reorder_sequence(memory_bank, reorder_ind, batch_first=True)
        encoder_final_state = reorder_gru_states(encoder_final_state, reorder_ind)
        """
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        first_layer_memory_bank = reorder_sequence(first_layer_memory_bank, reorder_ind, batch_first=True)
        second_layer_memory_bank = reorder_sequence(second_layer_memory_bank, reorder_ind, batch_first=True)
        first_layer_encoder_final_state = reorder_gru_states(first_layer_encoder_final_state, reorder_ind)
        second_layer_encoder_final_state = reorder_gru_states(second_layer_encoder_final_state, reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_second_layer_final_state = torch.cat((second_layer_encoder_final_state[-1,:,:], second_layer_encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
            encoder_first_layer_final_state = torch.cat((first_layer_encoder_final_state[-1,:,:], first_layer_encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
        else:
            encoder_second_layer_final_state = second_layer_encoder_final_state[-1, :, :] # [batch, hidden_size]
            encoder_first_layer_final_state = first_layer_encoder_final_state[-1, :, :]  # [batch, hidden_size]

        if self.separate_mode == 0: # only return the second layer
            return (second_layer_memory_bank.contiguous(), None), (
            encoder_second_layer_final_state, None)
        elif self.separate_mode == 1: # return second layer for generation, first layer for classification
            return (second_layer_memory_bank.contiguous(), first_layer_memory_bank.contiguous()), (
            encoder_second_layer_final_state, encoder_first_layer_final_state)
        elif self.separate_mode == 2: # return first layer for generation, second layer for classification
            return (first_layer_memory_bank.contiguous(), second_layer_memory_bank.contiguous()), (encoder_first_layer_final_state, encoder_second_layer_final_state)
        else:
            raise ValueError


class CatHirRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(CatHirRNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        #self.dropout = nn.Dropout(dropout)
        num_directions = 1 if not bidirectional else 2
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, self.pad_token)
        self.word_rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.sent_rnn = nn.GRU(input_size=hidden_size * num_directions, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_mask: [batch, src_seq_len], a sequential mask for src
        :param sent_positions: [batch, sent_num, 2], an integer tensor storing the forward and backward ending positions for each sentence of src
        :param sent_nums: a list containing the numbers of src sentences for each batch, with len=batch

        :return: the word-level memory bank [batch, src_seq_len, 2 * hidden_size] and the sentence-level memory bank [batch, sent_num, 2 * hidden_size]
        """
        # Debug
        #if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        src_embed = self.embedding(src) # [batch, src_len, embed_size]
        #src_embed_np = src_embed.detach().cpu().numpy()[:, 0, :]

        # sort src_embed according to its length
        batch_size = src.size(0)
        # 1. Word-level encoding
        assert len(src_lens) == batch_size
        sort_ind = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens_sorted = [src_lens[i] for i in sort_ind]
        src_embed = reorder_sequence(src_embed, sort_ind, batch_first=True)
        #src_embed_sorted_np = src_embed.detach().cpu().numpy()[:, 0, :]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens_sorted, batch_first=True)
        word_memory_bank, encoder_final_state = self.word_rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        word_memory_bank, _ = nn.utils.rnn.pad_packed_sequence(word_memory_bank, batch_first=True)  # unpack (back to padded)

        # restore the order of memory_bank
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        word_memory_bank = reorder_sequence(word_memory_bank, reorder_ind, batch_first=True)
        encoder_final_state = reorder_gru_states(encoder_final_state, reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1,:,:], encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :] # [batch, hidden_size]

        # 2. Sent-level encoding
        f_index = sent_positions[:, :, 0].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        b_index = sent_positions[:, :, 1].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        gather_index = torch.cat([f_index, b_index], dim=-1)
        # [batch, sent_num, 2 * hidden_size]
        sent_vectors = word_memory_bank.gather(dim=1, index=gather_index)
        sort_sent_ind = sorted(range(batch_size), key=lambda i: sent_nums[i], reverse=True)
        sent_nums_sorted = [sent_nums[i] for i in sort_sent_ind]

        sent_vectors = reorder_sequence(sent_vectors, sort_sent_ind, batch_first=True)
        packed_input_sent = nn.utils.rnn.pack_padded_sequence(sent_vectors, sent_nums_sorted, batch_first=True)

        # ([batch, sent_num, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        sent_memory_bank, sent_enc_final_state = self.sent_rnn(packed_input_sent)
        sent_memory_bank, _ = nn.utils.rnn.pad_packed_sequence(sent_memory_bank, batch_first=True)

        # restore the order of sent_memory_bank
        sent_back_map = {ind: i for i, ind in enumerate(sort_sent_ind)}
        sent_reorder_ind = [sent_back_map[i] for i in range(batch_size)]
        sent_memory_bank = reorder_sequence(sent_memory_bank, sent_reorder_ind, batch_first=True)
        sent_enc_final_state = reorder_gru_states(sent_enc_final_state, sent_reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            sent_enc_ll_final_state = torch.cat((sent_enc_final_state[-1, :, :], sent_enc_final_state[-2, :, :]), 1)
        else:
            sent_enc_ll_final_state = sent_enc_final_state[-1, :, :]

        word_memory_bank = word_memory_bank.contiguous()
        sent_memory_bank = sent_memory_bank.contiguous()
        return (word_memory_bank, sent_memory_bank), (encoder_last_layer_final_state, sent_enc_ll_final_state)


def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first, [B, T, D] if batch first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]
    device = sequence_emb.device
    order = torch.LongTensor(order)
    order = order.to(device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_


def reorder_gru_states(gru_state, order):
    """
    gru_state: [num_layer * num_directions, batch, hidden_size]
    order: list of sequence length
    """
    assert len(order) == gru_state.size(1)
    order = torch.LongTensor(order).to(gru_state.device)
    sorted_state = gru_state.index_select(index=order, dim=1)
    return sorted_state


def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states