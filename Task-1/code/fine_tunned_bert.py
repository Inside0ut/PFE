
"""
create a compiled fine_tunned_bert including the preprocessing layer, the bert encoder layer, the activation function 
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow_hub as hub

import tensorflow as tf
import os
import shutil

import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

def build_classifier_model(dropout=0.1,bert_model_name='small_bert/bert_en_uncased_L-4_H-512_A-8' ):
  
  #get the model specific to the model name
  map_name_to_handle = {
      'bert_en_uncased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
      'bert_en_cased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
      'bert_multi_cased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
      'small_bert/bert_en_uncased_L-2_H-128_A-2':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-2_H-256_A-4':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-2_H-512_A-8':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-2_H-768_A-12':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-4_H-128_A-2':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-4_H-256_A-4':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-4_H-512_A-8':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-4_H-768_A-12':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-6_H-128_A-2':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-6_H-256_A-4':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-6_H-512_A-8':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-6_H-768_A-12':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-8_H-128_A-2':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-8_H-256_A-4':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-8_H-512_A-8':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-8_H-768_A-12':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-10_H-128_A-2':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-10_H-256_A-4':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-10_H-512_A-8':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-10_H-768_A-12':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-12_H-128_A-2':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-12_H-256_A-4':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-12_H-512_A-8':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
      'albert_en_base':
          'https://tfhub.dev/tensorflow/albert_en_base/2',
      'electra_small':
          'https://tfhub.dev/google/electra_small/2',
      'electra_base':
          'https://tfhub.dev/google/electra_base/2',
      'experts_pubmed':
          'https://tfhub.dev/google/experts/bert/pubmed/2',
      'experts_wiki_books':
          'https://tfhub.dev/google/experts/bert/wiki_books/2',
      'talking-heads_base':
          'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
  }
  # get the preprocessor specific to the model 
  map_model_to_preprocess = {
      'bert_en_uncased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'bert_en_cased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'bert_multi_cased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
      'albert_en_base':
          'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
      'electra_small':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'electra_base':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'experts_pubmed':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'experts_wiki_books':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'talking-heads_base':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
  }

  #get the model specific to the model name
  tfhub_handle_encoder = map_name_to_handle[bert_model_name]
  # get the preprocessor specific to the model 
  tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
  #define layers
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(dropout)(net)
  #define the activation function
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
  return tf.keras.Model(text_input, net)

def create_fine_tunned_bert(training_data,bert_model_name,num_epochs=5,learning_rate=3e-5,dropout=0.1):

  classifier_model = build_classifier_model()
  epochs =num_epochs
  steps_per_epoch = tf.data.experimental.cardinality(training_data).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)
  #define the learning rate 
  init_lr = learning_rate
  #define the optimizer
  optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
  # define the loss
  loss = tf.keras.losses.BinaryCrossentropy()
  metrics = tf.metrics.BinaryAccuracy()
  #compile the model
  classifier_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', 
                         tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall(), 
                         tf.keras.metrics.AUC()
                        ])
  



  return classifier_model