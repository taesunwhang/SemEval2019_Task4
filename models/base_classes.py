from models.model_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo


class RNNInputDropout(nn.Dropout):
  """
  Dropout layer for the inputs of RNNs.

  Apply the same dropout mask to all the elements of the same sequence in
  a batch of sequences of size (batch, sequences_length, embedding_dim).
  """

  def forward(self, batch_sequence):
    """
    Apply dropout to the input batch of sequences.

    Args:
        batch_sequence: A batch of sequences of vectors that will serve
            as input to an RNN. Tensor of size (batch, sequences_length, emebdding_dim).
    Returns:
        A new tensor on which dropout has been applied.
    """
    # batch, sequence_len, embedding_dim
    ones = batch_sequence.data.new_ones(batch_sequence.shape[0], batch_sequence.shape[-1])

    # dropout = nn.dropout(inputs, drop_out_rate, training, inplace)
    dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)

    return dropout_mask.unsqueeze(1) * batch_sequence

class RNNEncoder(nn.Module):
  def __init__(self,
               rnn_type=nn.LSTM,
               input_size=300,
               hidden_size=256,
               num_layers=1,
               bias=True,
               dropout=0.0,
               bidirectional=False):

    assert issubclass(rnn_type, nn.RNNBase), \
      "rnn_type must be a class inheriting from torch.nn.RNNBase"
    "nn.LSTM, nn.RNN, nn.GRU"
    super(RNNEncoder, self).__init__()

    self.rnn_type = rnn_type
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.dropout = dropout
    self.bidirectional = bidirectional

    self._encoder = rnn_type(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bias=bias,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=bidirectional) # batch_first : batch, seq_len, embedding_dim

  def forward(self, batch_sequence, sequence_lengths):
    """
    :param batch_sequence:
                A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
    :param sequence_lengths:
                A 1D tensor containing the sizes of the
                sequences in the input batch.
    :return:
    """

    sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(batch_sequence, sequence_lengths)
    packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
    packed_outputs, _ = self._encoder(packed_batch, None) # packed_outputs

    outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
    # packed_outputs -> batch_size, max_seq_len, hidden_dim

    reordered_outputs = outputs.index_select(0, restoration_idx)
    # print("reordered_shape", reordered_outputs.shape)

    return reordered_outputs

class TextCNN(nn.Module):
  def __init__(self,
               input_embedding_dim=300,
               channel_in=1,
               channel_out=100,
               filter_sizes=[3,4,5],
               ):
    super(TextCNN, self).__init__()
    self.filter_sizes = filter_sizes
    self.conv = nn.ModuleList([nn.Conv2d(channel_in, channel_out,
                                         (window, input_embedding_dim)) for window in filter_sizes])

    '''
    self.conv13 = nn.Conv2d(Ci, Co, (3, D))
    self.conv14 = nn.Conv2d(Ci, Co, (4, D))
    self.conv15 = nn.Conv2d(Ci, Co, (5, D))
    '''

  def forward(self, batch_sequence):
    """
    :param batch_sequence:
                A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
    :param batch_sequence:
    :return:
    """
    batch_sequence = batch_sequence.unsqueeze(1) #batch, channel_out, max_seq_len, embedding_dim
    # [(batch, channel_out, max_seq_len), ...]*len(Ks)
    pad1d = [F.pad(batch_sequence, (0,0, window-1, window -1), "constant", 0) for window in self.filter_sizes]
    convolved_features = [F.relu(conv(pad.data)).squeeze(3) for conv, pad in zip(self.conv, pad1d)]

    # [(batch, channel_out), ...]*len(Ks)
    max_pooled = [F.max_pool1d(feature, feature.size(2)).squeeze(2) for feature in convolved_features]
    cnn_out = torch.cat(max_pooled, -1) # e.g. batch, 300

    return cnn_out


class ELMoEmbeddings(nn.Module):

  def __init__(self, options_file, weight_file, vocab:list, dropout=0):
    super(ELMoEmbeddings, self).__init__()
    self._elmo = Elmo(options_file=options_file,
                      weight_file=weight_file,
                      num_output_representations=1,
                      vocab_to_cache=vocab,
                      dropout=dropout)

  def forward(self, batch_char_sequence):
    embeddings = self._elmo(batch_char_sequence)
    elmo_representations = embeddings['elmo_representations'][0]

    return elmo_representations

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
