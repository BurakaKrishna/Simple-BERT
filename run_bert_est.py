import collections
import csv
import os
# Disable GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

import bert
import bert_utilities as utils
import data_processor
import optimization
import tokenization
import tensorflow as tf


# Task name
task_name = 'mrda'  # TODO Needs arg?
processors = {
    "swda": data_processor.SwdaProcessor(),
    "mrda": data_processor.MrdaProcessor(),
}  # TODO add the others included with BERT

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output/'

# Training parameters
max_seq_length = 128  # TODO Needs args?
train_batch_size = 32
learning_rate = 2e-5
num_train_epochs = 2.0  # Default 3

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
init_checkpoint = bert_model_type + '/bert_model.ckpt'  # TODO Needs arg?

print("------------------------------------")
print("Configured BERT Model...")
print("Model type: ", bert_model_type)
print("Configuration: ", bert_config.to_json_string())

print("------------------------------------")
print("Preparing data...")
# Prepare data and data processors
dataset_processor = processors[task_name.lower()]
tokenizer = tokenization.FullTokenizer(vocab_file=bert_model_type + '/vocab.txt', do_lower_case=do_lower_case)

# Get labels
label_list = dataset_processor.get_labels(data_dir)

train_data_dir = output_dir + "/train"
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
# Get training data and input function
training_examples = dataset_processor.get_train_examples(data_dir)
num_training_steps = int(len(training_examples) / train_batch_size * num_train_epochs)
train_data_file = os.path.join(train_data_dir, "train.tf_record")
data_processor.convert_examples_to_features(training_examples, label_list, max_seq_length, tokenizer, train_data_file)
train_input_fn = utils.input_fn_builder(
    input_file=train_data_file,
    seq_length=max_seq_length,
    is_training=True,
    drop_remainder=True)

eval_data_dir = output_dir + "/eval"
if not os.path.exists(eval_data_dir):
    os.makedirs(eval_data_dir)
eval_examples = dataset_processor.get_eval_examples(data_dir)
eval_data_file = os.path.join(eval_data_dir, "eval.tf_record")
data_processor.convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_data_file)
eval_input_fn = utils.input_fn_builder(
    input_file=eval_data_file,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False)

# Train BERT
print("------------------------------------")
print("Train Model...")
model_fn = utils.model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list),
    init_checkpoint=init_checkpoint,
    learning_rate=learning_rate,
    num_train_steps=num_training_steps,
    num_warmup_steps=0,
    use_tpu=False,
    use_one_hot_embeddings=False)

run_config = tf.estimator.RunConfig(model_dir=output_dir, save_checkpoints_steps=5, save_summary_steps=5)
estimator = tf.estimator.Estimator(model_fn, model_dir=output_dir, config=run_config, params={'batch_size': train_batch_size})
# estimator = tf.contrib.estimator.add_metrics(estimator, utils.metric_fn)
tf.logging.set_verbosity(tf.logging.INFO)

num_training_steps = int(len(training_examples) / train_batch_size) * num_train_epochs
num_eval_steps = int(len(eval_examples) / train_batch_size) * num_train_epochs

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_training_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=num_eval_steps, start_delay_secs=60)
eval_results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
# print(eval_results)


# epochs = 2
# for epoch in range(epochs):
#     print("epochs-{}".format(epoch))
#     estimator.train(input_fn=train_input_fn, steps=num_training_steps)
#     eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_steps)
#     print(eval_results)



