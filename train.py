import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import pickle
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random
import time

import torch
import torch.nn as nn

from data_process import SemEval2019Task4Processor
from data.data_utils import InputExamples
from sklearn.metrics import f1_score, classification_report



class MetonymyModel(object):
  def __init__(self, hparams, dataset_type):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)
    self.dataset_type = dataset_type

  def _build_data_process(self):
    print("\t* Loading training data...")
    processors = {
      "semeval": SemEval2019Task4Processor,
    }

    self.processor = processors[self.hparams.task_name](self.hparams, self.dataset_type)
    self.train_examples = self.processor.get_train_examples()
    self.test_examples = self.processor.get_test_examples()

    # self.word_embeddings = self.processor.get_word_embeddings()

  def _build_model(self):
    # -------------------- Model definition ------------------- #
    print('\t* Building model...')

    self.model = self.hparams.model(self.hparams)
    # to device
    self.model = self.model.to(self.device)

    # -------------------- Preparation for training  ------------------- #
    # self.criterion = nn.CrossEntropyLoss()

    print("Weighted Cross Entropy Loss")
    class_dist_dict = self.hparams.class_distribution
    print(class_dist_dict.items())
    class_weights = [sum(class_dist_dict.values()) / class_dist_dict[key] for key in class_dist_dict.keys()]
    # class_weights = [2 * max(class_weights), min(class_weights)]
    print("class_weights", class_weights)

    # sf_class_weights = [math.exp(weights) / sum([math.exp(w) for w in class_weights]) for weights in class_weights]
    # print("sf_class_weights", sf_class_weights)
    self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode="max",
                                                                factor=0.5,
                                                                patience=0)

  def _batch_data_to_device(self, batch_data):
    batch_article_embeddings, batch_labels, article_lengths = batch_data

    batch_article_embeddings = torch.Tensor(batch_article_embeddings).to(self.device)
    batch_labels = torch.Tensor(batch_labels).to(self.device)
    batch_article_len = torch.Tensor(article_lengths).to(self.device)

    return batch_article_embeddings, batch_labels, batch_article_len

  def train(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self._build_data_process()
    self._build_model()

    # total_examples
    train_data_len = int(math.ceil(len(self.train_examples)/self.hparams.train_batch_size))
    self._logger.info("Batch iteration per epoch is %d" % train_data_len)

    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)

    max_eval_acc = 0
    max_eval_cls_report = None
    for epoch_completed in range(self.hparams.num_epochs):
      self.model.train()
      loss_sum, correct_preds_sum = 0, 0

      if epoch_completed > 0:
        self.train_examples = self.processor.get_train_examples()

      tqdm_batch_iterator = tqdm(range(train_data_len))
      for batch_idx in tqdm_batch_iterator:
        time.sleep(0.1)
        batch_data = self.processor.get_batch_data(batch_idx, self.hparams.train_batch_size, "train")
        batch_article_embeddings, batch_labels, batch_article_len = self._batch_data_to_device(batch_data)

        logits = self.model(batch_article_embeddings, batch_article_len)
        predictions = torch.argmax(logits, dim=-1)
        correct_preds = torch.sum(torch.eq(predictions.float(), batch_labels).int())
        loss = self.criterion(logits, batch_labels.long())

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
        self.optimizer.step()

        loss_sum += loss.item()
        correct_preds_sum += correct_preds.item()

        description = "Avg. batch proc. loss: {:.4f}" \
          .format(loss_sum / (batch_idx + 1))
        tqdm_batch_iterator.set_description(description)

      self._logger.info("-> Training loss = {:.4f}, accuracy: {:.4f}%\n"
            .format(loss_sum / train_data_len, (correct_preds_sum / len(self.train_examples))*100))

      eval_cls_report, eval_acc = self._run_evaluate(epoch_completed)

      if eval_acc > max_eval_acc:
        max_eval_acc = eval_acc
        max_eval_cls_report = eval_cls_report

    print(max_eval_cls_report)

  def _run_evaluate(self, epoch_completed):
    # self.test_examples = self.processor.get_test_examples()
    eval_data_len = int(math.ceil(len(self.test_examples) / self.hparams.eval_batch_size))

    self.model.eval()
    with torch.no_grad():
      self._logger.info("Batch iteration per epoch is %d" % eval_data_len)

      loss_sum, correct_preds_sum = 0, 0
      total_labels, total_preds = [], []

      for batch_idx in range(eval_data_len):

        batch_data = self.processor.get_batch_data(batch_idx, self.hparams.eval_batch_size, "test")
        batch_article_embeddings, batch_labels, batch_article_len = self._batch_data_to_device(batch_data)
        logits = self.model(batch_article_embeddings, batch_article_len)
        predictions = torch.argmax(logits, dim=-1)
        # print(predictions)
        correct_preds = torch.sum(torch.eq(predictions.float(), batch_labels).int())
        # print(batch_labels)
        loss = self.criterion(logits, batch_labels.long())

        loss_sum += loss

        total_labels.extend(batch_labels.to(torch.device("cpu")))
        total_preds.extend(predictions.to(torch.device("cpu")))

        correct_preds_sum += correct_preds.item()

      print(classification_report(total_labels, total_preds, labels=[0,1], target_names=["false", "true"]))
      print("{}->{} loss: {:.4f}, accuracy: {:.4f}%, f1_score : {:.4f}"
            .format(epoch_completed, "test", (loss_sum / eval_data_len),
                    (correct_preds_sum / len(self.test_examples))*100, f1_score(total_labels, total_preds, average='micro')))
      self.scheduler.step((correct_preds_sum / len(self.test_examples)))

    return classification_report(total_labels, total_preds, digits=3), (correct_preds_sum / len(self.test_examples))*100











