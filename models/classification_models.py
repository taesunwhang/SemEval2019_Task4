from models.base_classes import *

class SemEvalLSTM(nn.Module):
  def __init__(self, hparams):
    super(SemEvalLSTM, self).__init__()
    self.hparams = hparams
    embedding_dim = self.hparams.elmo_embedding_dim if self.hparams.do_use_elmo else self.hparams.sentbert_embedding_dim

    self._rnn_encoder = RNNEncoder(
      rnn_type=nn.LSTM,
      input_size=embedding_dim,
      hidden_size=self.hparams.rnn_hidden_dim,
      num_layers=self.hparams.rnn_depth,
      bias=True,
      dropout=1-self.hparams.dropout_keep_prob,
      bidirectional=True
    )

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.rnn_hidden_dim * 2, self.hparams.rnn_hidden_dim),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.rnn_hidden_dim, len(self.hparams.output_classes))
    )

  def forward(self, batch_elmo_embedded, article_lengths):
    # batch, article_len, embedding_dim
    # # article_len : [200, 32, 15, 62, 77, ...]

    rnn_outputs = self._rnn_encoder(batch_elmo_embedded, article_lengths.long())

    last_tok_reps = [rnn_out[article_len-1,:] for rnn_out, article_len in zip(rnn_outputs, article_lengths.long())]
    last_tok_rep = torch.stack(last_tok_reps, dim=0)
    forward_last = torch.split(last_tok_rep, split_size_or_sections=self.hparams.rnn_hidden_dim, dim=-1)[0]
    first_tok_rep = rnn_outputs[:,0,:]
    backward_last = torch.split(first_tok_rep, split_size_or_sections=self.hparams.rnn_hidden_dim, dim=-1)[1]

    bilstm_concat = torch.cat([forward_last, backward_last], dim=-1)
    logits = self._classification(bilstm_concat)

    return logits

class SemEvalCNN(nn.Module):
  def __init__(self, hparams, vocab=None):
    super(SemEvalCNN, self).__init__()
    self.hparams = hparams

    embedding_dim = self.hparams.elmo_embedding_dim if self.hparams.do_use_elmo else self.hparams.sentbert_embedding_dim

    self._text_cnn = TextCNN(
      input_embedding_dim=embedding_dim,
      channel_in=1,
      channel_out=self.hparams.num_filters,
      filter_sizes=self.hparams.filter_sizes,
    )

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(len(self.hparams.filter_sizes)*self.hparams.num_filters,
                int(len(self.hparams.filter_sizes)*self.hparams.num_filters/ 2)),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(int(len(self.hparams.filter_sizes)*self.hparams.num_filters / 2), len(self.hparams.output_classes))
    )

  def forward(self, batch_elmo_embedded, article_lengths):
    # batch, article_len, embedding_dim
    # # article_len : [200, 32, 15, 62, 77, ...]

    cnn_outputs = self._text_cnn(batch_elmo_embedded)
    logits = self._classification(cnn_outputs)

    return logits

