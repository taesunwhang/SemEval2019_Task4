import os
import pickle
import random
import numpy as np

from data.data_utils import InputExamples
from nltk.tag.perceptron import PerceptronTagger
from allennlp.modules.elmo import batch_to_ids

from models.bert import tokenization_bert


# Code is widely inspired from:
# https://github.com/google-research/bert
class SemEval2019Task4Processor(object):
  def __init__(self, hparams, dataset_type="relocar"):
    self.hparams = hparams
    self.dataset_type = dataset_type

    self._get_word_dict()
    self._pad_idx = self.hparams.pad_idx

    if self.hparams.do_bert:
      self._bert_tokenizer_init()

  def _bert_tokenizer_init(self, bert_pretrained='bert-base-cased'):
    bert_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % bert_pretrained
    vocab_file_path = "%s-vocab.txt" % bert_pretrained

    self._tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
      do_lower_case=False
    )
    print("bert_tokenizer")

  # data_dir
  def get_train_examples(self):
    self.train_example = self._read_pkl(os.path.join("data/%s" % self.hparams.train_pkl), do_shuffle=True)

    return self.train_example

  def get_test_examples(self):
    self.test_example = self._read_pkl(os.path.join("data/%s" % self.hparams.test_pkl), do_shuffle=False)
    return self.test_example

  def _read_pkl(self, data_dir, do_shuffle=False):
    print("[Reading %s]" % data_dir)
    with open(data_dir, "rb") as frb_handle:
      total_examples = pickle.load(frb_handle)

      if do_shuffle and self.hparams.training_shuffle_num > 1:
        total_examples = self._data_shuffling(total_examples, self.hparams.training_shuffle_num)

      return total_examples

  def _data_shuffling(self, inputs, shuffle_num):
    for i in range(shuffle_num):
      random_seed = random.sample(list(range(0, 1000)), 1)[0]
      random.seed(random_seed)
      random.shuffle(inputs)
    # print("Shuffling total %d process is done! Total dialog context : %d" % (shuffle_num, len(inputs)))

    return inputs

  def _get_word_dict(self):
    with open(self.hparams.vocab_dir, "r", encoding="utf-8") as vocab_handle:
      self.vocab = [word.strip() for word in vocab_handle if len(word.strip()) > 0]

    self.word2id = dict()
    for idx, word in enumerate(self.vocab):
      self.word2id[word] = idx

  def get_batch_data(self, curr_index, batch_size, set_type="train"):
    article_embeddings = []
    labels_id = []
    article_lengths = []

    examples = {
      "train": self.train_example,
      "test": self.test_example,
    }
    example = examples[set_type]

    for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
      # print(curr_index, np.array(each_example.embeddings).shape[0], each_example.article_len, each_example.article_id)
      assert np.array(each_example.embeddings).shape[0] == each_example.article_len

      article_embeddings.append(list(each_example.embeddings)) # article_len, embedding_dim
      labels_id.append(each_example.label)
      article_lengths.append(each_example.article_len)

    article_input_embeddings = rank_3_pad_process(article_embeddings)
    # print(np.array(article_input_embeddings).shape)
    # print(np.array(article_input_embeddings).shape)

    return article_input_embeddings, labels_id, article_lengths

  # def get_word_embeddings(self):
  #   with np.load(self.hparams.glove_embedding_path % (self.dataset_type, self.dataset_type)) as data:
  #     print("glove embedding shape", np.shape(data["embeddings"]))
  #     return data["embeddings"]

def rank_2_pad_process(inputs, pad_idx=0):

  max_sent_len = 0
  for sent in inputs:
    max_sent_len = max(len(sent), max_sent_len)

  padded_result = []
  sent_buffer = []
  for sent in inputs:
    for i in range(max_sent_len - len(sent)):
      sent_buffer.append(pad_idx)
    sent.extend(sent_buffer)
    padded_result.append(sent)
    sent_buffer = []

  return padded_result

def rank_3_pad_process(inputs, pad_idx=0):

  max_sent_len = 0
  max_word_len = 0

  for sent in inputs:
    # print(len(sent))
    max_sent_len = max(len(sent), max_sent_len)
    for idx, word in enumerate(sent):
      if type(word) is not list:
        sent[idx] = list(word)
      max_word_len = max(len(word), max_word_len)

  padded_results = []
  for sent in inputs:
    sent_buffer = [[pad_idx]] * (max_sent_len - len(sent))
    sent.extend(sent_buffer)

    for word in sent:
      word_buffer = [pad_idx] * (max_word_len - len(word))
      word.extend(word_buffer)
    padded_results.append(sent)

  return padded_results