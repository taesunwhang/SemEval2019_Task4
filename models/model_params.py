from collections import defaultdict
from models.classification_models import *

BASE_PARAMS = defaultdict(
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_num = [0],

  # Input params
  train_batch_size=16,
  eval_batch_size=16,

  # Training params
  learning_rate=3e-4,
  training_shuffle_num=50,
  dropout_keep_prob=0.8,
  num_epochs=20,
  embedding_dim=300,
  elmo_embedding_dim = 1024,
  sentbert_embedding_dim=768,
  class_distribution = {0:273, 1:159},

  rnn_hidden_dim=256,
  rnn_depth=2,
  output_classes = [0, 1],
  max_gradient_norm=10.0,

  max_position_embeddings=512,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  attention_probs_dropout_prob=0.1,
  layer_norm_eps=1e-12,

  # Input Config
  # semeval_class_dist = {"<MET>": 173, "<LIT>" : 737},

  # Train Model Config
  task_name="semeval",
  do_use_elmo=False,
  do_bert=True,

  # Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
  root_dir="./runs/",
  data_dir="data/%s/",
  vocab_dir="data/news_vocab.txt",
  pad_idx=0,

  train_pkl="news_train_elmo.pkl",
  test_pkl="news_test_elmo.pkl",
)

BILSTM_ELMO_PARAMS = BASE_PARAMS.copy()
BILSTM_ELMO_PARAMS.update(
  do_use_elmo=True,
  elmo_options_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
  elmo_weight_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
  model=SemEvalLSTM,
  eval_batch_size=16
)

CNN_ELMO_PARAMS = BASE_PARAMS.copy()
CNN_ELMO_PARAMS.update(
  do_use_elmo=True,
  elmo_options_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
  elmo_weight_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
  model=SemEvalCNN,
  train_batch_size=16,
  eval_batch_size=50,
  learning_rate=1e-3,

  # cnn
  num_filters=100,
  filter_sizes=[3,4,5]
)

BILSTM_SENTBERT_PARAMS = BASE_PARAMS.copy()
BILSTM_SENTBERT_PARAMS.update(
  do_use_elmo=False,
  model=SemEvalLSTM,
  train_batch_size=16,
  eval_batch_size=16,
  learning_rate=3e-5,
  train_pkl="news_sentbert_final_train.pkl",
  test_pkl="news_sentbert_final_test.pkl",
)

CNN_SENTBERT_PARAMS = BASE_PARAMS.copy()
CNN_SENTBERT_PARAMS.update(
  do_use_elmo=False,
  model=SemEvalCNN,
  train_batch_size=16,
  eval_batch_size=16,
  learning_rate=3e-5,

  # cnn
  num_filters=100,
  filter_sizes=[3,4,5],

  train_pkl="news_sentbert_final_train.pkl",
  test_pkl="news_sentbert_final_test.pkl",
)

BILSTM_BERT_PARAMS = BASE_PARAMS.copy()
BILSTM_BERT_PARAMS.update(
  do_use_elmo=False,
  model=SemEvalLSTM,
  train_batch_size=8,
  eval_batch_size=16,
  learning_rate=3e-4,
  train_pkl="news_test_bert.pkl",
  test_pkl="news_train_bert.pkl",
)

CNN_BERT_PARAMS = BASE_PARAMS.copy()
CNN_BERT_PARAMS.update(
  do_use_elmo=False,
  model=SemEvalCNN,
  train_batch_size=8,
  eval_batch_size=16,
  learning_rate=3e-5,

  # cnn
  num_filters=100,
  filter_sizes=[3,4,5],

  train_pkl="news_test_bert.pkl",
  test_pkl="news_train_bert.pkl",
)