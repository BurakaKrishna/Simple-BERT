import collections
import csv
import os
import pickle

import bert
import data_processor
import optimization
import tokenization
import tensorflow as tf

# Task name
task_name = 'mrda'  # TODO Needs arg?

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output/'

# Training parameters
max_seq_length = 128  # TODO Needs args?
train_batch_size = 32
learning_rate = 2e-5
num_train_epochs = 1.0  # Default 3

print("------------------------------------")
print("Using parameters...")
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", train_batch_size)
print("Learning Rate: ", learning_rate)
print("Epochs: ", num_train_epochs)

# Configure BERT
bert_model_type = 'BERT_Base'  # TODO Needs arg?
do_lower_case = True  # TODO Needs arg?
# vocab_file = bert_model_type + '/vocab.txt'
# bert_config_file = bert_model_type + '/bert_config.json'
bert_config = bert.BertConfig.from_json_file(bert_model_type + '/bert_config.json')
if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

# BERT model checkpoint
init_checkpoint = 'BERT_Base/bert_model.ckpt'  # TODO Needs arg?

print("------------------------------------")
print("Configured BERT Model...")
print("Model type: ", bert_model_type)
print("Configuration: ", bert_config.to_json_string())

# Prepare data and data processors
data_processor = data_processor.processors[task_name.lower()]()
tokenizer = tokenization.FullTokenizer(vocab_file=bert_model_type + '/vocab.txt', do_lower_case=do_lower_case)

label_list = data_processor.get_labels()
print("------------------------------------")
print("Define Graph...")
# Define Tensorflow Graph
print("------------------------------------")
print("Define Graph...")

