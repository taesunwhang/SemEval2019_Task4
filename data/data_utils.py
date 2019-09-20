import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
import pickle
from nltk import word_tokenize
import numpy as np

import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids
from models.base_classes import ELMoEmbeddings
from models.bert import modeling_bert
from models.bert import tokenization_bert

import gzip

class InputExamples(object):
  def __init__(self, article_id, article_len=None, label=None, sentences=None, embeddings=None):

    self.article_id = article_id
    self.article_len = article_len
    self.label = label
    self.sentences = sentences
    self.embeddings = embeddings

class HyperpartisanDataUtils(object):
  def __init__(self, data_dir, data_type="train"):
    # CoNLL2003, ReLocaR, SemEval2010
    self.data_dir = data_dir
    self.pkl_dir = "news_%s.pkl" % (data_type)
    self.vocab_dir = "news_vocab.txt"
    self.vocab = set()

    examples = self._read_dataset()
    self._make_dataset_pkl(examples)

    if data_type == "train":
      self._make_vocab_file()

  def _read_dataset(self):
    examples = []
    with open(self.data_dir, "r", encoding="utf-8") as fr_handle:
      data = [line for line in fr_handle if len(line.rstrip()) > 0]

      prev_article_id = 0
      prev_label_id = None
      article_sentences = []
      for idx, line in enumerate(data):
        curr_label_id, curr_article_id, sentence = line.split('\t')
        sentence_split = word_tokenize(sentence.strip())
        self.vocab.update(sentence_split)
        article_sentences.append(sentence_split)

        if prev_label_id == None:
          prev_label_id = int(curr_label_id)

        if int(curr_article_id) != prev_article_id or (idx + 1) == len(data):
          examples.append(
            InputExamples(
              article_id=prev_article_id,
              article_len=len(article_sentences),
              sentences=article_sentences,
              label=prev_label_id
            )
          )
          print(prev_article_id, prev_label_id, len(examples))

          assert prev_label_id in [0, 1]
          article_sentences = []
          prev_article_id = int(curr_article_id)
          prev_label_id = int(curr_label_id)

      # assert prev_article_id == len(examples)

    return examples

  def _make_dataset_pkl(self, examples):
    with open(self.pkl_dir, "wb") as pkl_handle:
      pickle.dump(examples, pkl_handle)

  def _make_vocab_file(self):
    with open(self.vocab_dir, "w", encoding="utf8") as vocab_handle:
      vocab = list(self.vocab)
      vocab.insert(0, "<PAD>")
      vocab.append("<UNK>")
      for word_token in vocab:
        vocab_handle.write(word_token + "\n")
      print("total vocab size", len(vocab))

class ELMODataUtils(object):
  def __init__(self, read_pkl_dir, vocab_dir, write_pkl_dir, data_type="train"):
    self.vocab_dir = vocab_dir
    self.read_pkl(read_pkl_dir)
    self._get_word_dict()
    self.data_type = data_type

    self.elmo_options_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    self.elmo_weight_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.elmo_model = ELMoEmbeddings(self.elmo_options_file, self.elmo_weight_file, self.vocab).to(self.device)
    article_elmo_embeddings = self._get_elmo_embeddings()

    embedding_examples = self.create_embedding_examples(article_elmo_embeddings)
    self.write_pkl(write_pkl_dir, embedding_examples)

  def read_pkl(self, read_pkl_dir):
    with open(read_pkl_dir, "rb") as frb_handle:
      self.examples = pickle.load(frb_handle)
      print(len(self.examples))

  def _get_word_dict(self):
    with open(self.vocab_dir, "r", encoding="utf-8") as vocab_handle:
      self.vocab = [word.strip() for word in vocab_handle if len(word.strip()) > 0]

    self.word2id = dict()
    for idx, word in enumerate(self.vocab):
      self.word2id[word] = idx

  def _get_elmo_embeddings(self):
    def convert_single_example(example: InputExamples, word2id):
      input_id = []
      length = []
      # example.sentences : tokenized sentences
      for sent in example.sentences:
        input_id_buffer = []
        for word_token in sent:
          try:
            input_id_buffer.append(word2id[word_token])
          except KeyError:
            input_id_buffer.append(word2id["<UNK>"])

        input_id.append(input_id_buffer)  # tokenized converted ids (each article, several sentences)
        length.append(len(input_id_buffer))

      return input_id, length

    inputs, inputs_len = [], []
    article_elmo_embeddings = []
    one_article_embeddings = []
    batch_size = 50

    self.elmo_model.eval()
    with torch.no_grad():
      for index, each_example in enumerate(self.examples):
        for sent in each_example.sentences:
          for idx, word_tok in enumerate(sent):
            if not word_tok in self.word2id.keys():
              sent[idx] = "<UNK>"
          inputs.append(sent)
          inputs_len.append(len(sent))

          if len(inputs) == batch_size or sent == each_example.sentences[-1]:
            batch_inputs_id = batch_to_ids(inputs)
            batch_len = torch.tensor(inputs_len, dtype=torch.float32).to(self.device)
            batch_elmo_embeddings = self.elmo_model(torch.tensor(batch_inputs_id).to(self.device))

            elmo_emb_sum = torch.sum(batch_elmo_embeddings, dim=1)
            elmo_avg_emb = torch.div(elmo_emb_sum, batch_len.unsqueeze(1)) # batch, 1024
            one_article_embeddings.append(elmo_avg_emb)
            inputs, inputs_len = [], []

        one_article_emb = torch.cat(one_article_embeddings, dim=0).to(torch.device("cpu"))
        article_elmo_embeddings.append(one_article_emb)

        assert len(each_example.sentences) == np.array(one_article_emb).shape[0]
        one_article_embeddings = []
        print(index)

    return article_elmo_embeddings

  def create_embedding_examples(self, article_elmo_embeddings):
    embedding_examples = []
    self.label1_cnt = 0
    self.label2_cnt = 0


    for index, (each_example, elmo_emb) in enumerate(zip(self.examples, article_elmo_embeddings)):
      embedding_examples.append(
        InputExamples(
          article_id=each_example.article_id,
          article_len=each_example.article_len,
          label=each_example.label,
          sentences=each_example.sentences,
          embeddings=np.array(elmo_emb)
        )
      )
      if int(each_example.label) == 0:
        self.label1_cnt += 1
      elif int(each_example.label) == 1:
        self.label2_cnt += 1
      print(index, each_example.label)

      assert np.array(elmo_emb).shape[0] == len(each_example.sentences)
      print(self.data_type, "0 :", self.label1_cnt, "/ 1 :", self.label2_cnt)

    return embedding_examples

  def write_pkl(self, write_pkl_dir, embedding_examples):
    with open(write_pkl_dir, "wb") as fwb_handle:
      pickle.dump(embedding_examples, fwb_handle)

class BERTDataUtils(object):
  def __init__(self, read_pkl_dir, write_pkl_dir, data_type="train"):
    self.read_pkl(read_pkl_dir)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.data_type = data_type
    self._bert_tokenizer_init()
    self._bert_model = modeling_bert.BertModel.from_pretrained("bert-base-cased")
    article_bert_embeddings = self._get_bert_embeddings()

    embedding_examples = self.create_embedding_examples(article_bert_embeddings)
    self.write_pkl(write_pkl_dir, embedding_examples)

  def read_pkl(self, read_pkl_dir):
    with open(read_pkl_dir, "rb") as frb_handle:
      self.examples = pickle.load(frb_handle)
      print(len(self.examples))

  def _bert_tokenizer_init(self, bert_pretrained='bert-base-cased'):
    bert_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % bert_pretrained
    vocab_file_path = "%s-vocab.txt" % bert_pretrained

    self._tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
      do_lower_case=False
    )
    print("bert_tokenizer")

  def _get_bert_embeddings(self):

    article_bert_embeddings = []
    one_article_embeddings = []
    batch_size = 50

    self._bert_model.eval()
    with torch.no_grad():
      for index, each_example in enumerate(self.examples):
        batch_inputs_id, inputs_len = [], []
        for sent in each_example.sentences:
          tokenized_sent = self._tokenizer.convert_tokens_to_ids(sent)
          if len(tokenized_sent) > 512:
            tokenized_sent = tokenized_sent[:512]

          batch_inputs_id.append(tokenized_sent)
          # print(np.array(self._tokenizer.convert_tokens_to_ids(sent)).shape)
          inputs_len.append(len(sent))

          if len(batch_inputs_id) == batch_size or sent == each_example.sentences[-1]:
            batch_inputs_id = self.rank_2_pad_process(batch_inputs_id)
            print(np.array(batch_inputs_id).shape)
            batch_len = torch.tensor(inputs_len).to(self.device)
            batch_inputs_id = torch.tensor(batch_inputs_id).to(self.device)
            segment_tensors = torch.zeros(batch_inputs_id.size(), dtype=torch.long).to(self.device)

            batch_bert_embeddings, _ = self._bert_model(batch_inputs_id, segment_tensors)

            bert_emb_sum = torch.sum(batch_bert_embeddings, dim=1)
            bert_avg_emb = torch.div(bert_emb_sum, batch_len.float().unsqueeze(1)) # batch
            one_article_embeddings.append(bert_avg_emb)
            batch_inputs_id, inputs_len = [], []

        one_article_emb = torch.cat(one_article_embeddings, dim=0).to(torch.device("cpu"))
        article_bert_embeddings.append(one_article_emb)

        assert len(each_example.sentences) == np.array(one_article_emb).shape[0]
        one_article_embeddings = []
        print(index)

    return article_bert_embeddings

  def create_embedding_examples(self, article_bert_embeddings):
    embedding_examples = []

    for index, (each_example, elmo_emb) in enumerate(zip(self.examples, article_bert_embeddings)):
      embedding_examples.append(
        InputExamples(
          article_id=each_example.article_id,
          article_len=each_example.article_len,
          label=each_example.label,
          sentences=each_example.sentences,
          embeddings=np.array(elmo_emb)
        )
      )
      assert np.array(elmo_emb).shape[0] == len(each_example.sentences)

    return embedding_examples

  def write_pkl(self, write_pkl_dir, embedding_examples):
    with open(write_pkl_dir, "wb") as fwb_handle:
      pickle.dump(embedding_examples, fwb_handle)

  def rank_2_pad_process(self, inputs, pad_idx=0):

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

class SentBERTUtils(object):
  def __init__(self, read_pkl_dir, write_pkl_dir, data_type="train"):
    self.label1_cnt = 0
    self.label2_cnt = 0
    examples = self.read_pkl(read_pkl_dir)

    self.write_pkl(write_pkl_dir, examples)

  def read_pkl(self, read_pkl_dir):
    examples = []
    with gzip.open(read_pkl_dir, "rb") as frb_handle:
      data = pickle.load(frb_handle)

    for idx in range(len(data)):

      article_id = idx
      label_id = data.iloc[idx,1]
      sentfbert_embeddings = data.iloc[idx,2]
      # print(sentfbert_embeddings)

      examples.append(
        InputExamples(
          article_id=article_id,
          label=label_id,
          embeddings=sentfbert_embeddings,
          article_len=len(sentfbert_embeddings)
        )
      )
      # print(idx, len(sentfbert_embeddings))
      assert int(article_id) + 1 == len(examples)

      if int(label_id) == 0:
        self.label1_cnt += 1
      elif int(label_id) == 1:
        self.label2_cnt += 1

    return examples

  def write_pkl(self, write_pkl_dir, embedding_examples):
    with open(write_pkl_dir, "wb") as fwb_handle:
      pickle.dump(embedding_examples, fwb_handle)

if __name__ == '__main__':
  # HyperpartisanDataUtils("new_train_data.tsv", "train")
  # HyperpartisanDataUtils("new_test_data.tsv", "test")

  # ELMODataUtils("news_train.pkl", "news_vocab.txt", "news_train_elmo.pkl")
  # ELMODataUtils("news_test.pkl", "news_vocab.txt", "news_test_elmo.pkl")

  BERTDataUtils("news_test.pkl", "news_test_bert.pkl")
  BERTDataUtils("news_train.pkl", "news_train_bert.pkl")

  # SentBERTUtils("final_data_train_ts.pickle", "news_sentbert_final_train.pkl", "train")
  # SentBERTUtils("final_data_test_ts.pickle", "news_sentbert_final_test.pkl", "test")